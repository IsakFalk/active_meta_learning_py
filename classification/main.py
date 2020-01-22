import pickle
import logging
from random import shuffle
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import numpy as np
from tqdm import tqdm

import utils
from torchmeta_utils import update_parameters, get_accuracy
import optimisation
import kernels
from models import ConvolutionalNeuralNetwork
import arguments


def get_active_learning_batches(sampled_batches_train, kernel_mat_func, fw_class):
    fw_batches_train = utils.convert_batches_to_fw_form(sampled_batches_train)
    K = kernel_mat_func(fw_batches_train)
    fw = fw_class(K)
    fw.run()
    sampled_batches_train_fw = [sampled_batches_train[i] for i in fw.sampled_order]
    return sampled_batches_train_fw


def get_outer_loss(batch, model, lr_inner, device, first_order, test=False):
    if test:
        _model = pickle.loads(pickle.dumps(model))
    else:
        _model = model

    train_inputs, train_targets = batch["train"]
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device)
    batch_size = train_inputs.shape[0]

    test_inputs, test_targets = batch["test"]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device)

    outer_loss = torch.tensor(0.0, device=device)
    accuracy = torch.tensor(0.0, device=device)
    for task_idx, (train_input, train_target, test_input, test_target) in enumerate(
        zip(train_inputs, train_targets, test_inputs, test_targets)
    ):
        train_logit = _model(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target, reduction="mean")

        _model.zero_grad()
        params = update_parameters(
            _model, inner_loss, step_size=lr_inner, first_order=first_order
        )

        test_logit = _model(test_input, params=params)
        outer_loss += F.cross_entropy(test_logit, test_target, reduction="mean")

        with torch.no_grad():
            accuracy += get_accuracy(test_logit, test_target)

    return outer_loss.div_(batch_size), accuracy.div_(batch_size)


def generate_dataloaders(args):
    # Create dataset
    if args.dataset == "omniglot":
        dataset_train = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_train,
            seed=args.seed,
            download=True,  # Only downloads if not in data_path
        )
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )

        dataset_val = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_val,
            seed=args.seed,
            download=True,
        )
        dataloader_val = BatchMetaDataLoader(
            dataset_val,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )

        dataset_test = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_test,
            download=True,
        )
        dataloader_test = BatchMetaDataLoader(
            dataset_test,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )
    elif args.dataset == "miniimagenet":
        dataset_train = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_train,
            seed=args.seed,
            download=True,  # Only downloads if not in data_path
        )
        dataloader_train = BatchMetaDataLoader(
            dataset_train,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )

        dataset_val = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_val,
            seed=args.seed,
            download=True,
        )
        dataloader_val = BatchMetaDataLoader(
            dataset_val,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )

        dataset_test = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_test,
            download=True,
        )
        dataloader_test = BatchMetaDataLoader(
            dataset_test,
            batch_size=args.tasks_per_metaupdate,
            shuffle=True,
            num_workers=args.n_workers,
        )
    else:
        raise Exception("Dataset {} not implemented".format(args.dataset))

    return dataloader_train, dataloader_val, dataloader_test


def run_training_loop(
    sampled_batches_train,
    sampled_batches_test,
    model_func,
    lr_meta,
    lr_inner,
    first_order,
    evaluate_every,
    num_epochs,
    device,
):
    T = len(sampled_batches_train)
    test_loss = np.zeros(((T // evaluate_every) + 1, num_epochs))
    test_acc = np.zeros(((T // evaluate_every) + 1, num_epochs))

    def evaluate_model(model):
        temp_loss = []
        temp_acc = []
        for batch in sampled_batches_test:
            model.zero_grad()
            outer_loss, accuracy = get_outer_loss(
                batch, model, lr_inner, device, first_order, test=True
            )
            temp_loss.append(outer_loss.squeeze().item())
            temp_acc.append(accuracy.squeeze().item())
        return np.array(temp_loss).mean(), np.array(temp_acc).mean()

    # Every evaluate_every we train the model and evaluate it
    # This means that we will use train_data[1:t] for each of these t to train
    for i, t in enumerate(range(0, len(sampled_batches_train) + 1, evaluate_every)):
        logging.info(
            "Evaluating model with metatrain instance until index {}".format(t)
        )
        if t == 0:
            model = model_func()
            model.to(device=device)
            # Get loss for uninitialised model
            # This should have accuracy 1/n_way
            for epoch in range(num_epochs):
                temp_loss, temp_acc = evaluate_model(model)
                test_loss[i, epoch] = temp_loss
                test_acc[i, epoch] = temp_acc
        else:
            cur_train_batches = sampled_batches_train[:t]
            model = model_func()
            model.to(device=device)
            model.train()
            meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_meta)

            for epoch in range(num_epochs):
                logging.info("Running epoch {}".format(epoch))
                shuffle(cur_train_batches)
                with tqdm(cur_train_batches) as pbar:
                    for idx, batch in enumerate(pbar):
                        model.zero_grad()

                        outer_loss, accuracy = get_outer_loss(
                            batch, model, lr_inner, device, first_order, test=False
                        )

                        outer_loss.backward()
                        meta_optimizer.step()

                # Get average test loss / acc
                temp_loss, temp_acc = evaluate_model(model)
                test_loss[i, epoch] = temp_loss
                test_acc[i, epoch] = temp_acc

    return test_loss, test_acc


def run(args):
    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # get save path
    # and create directory if it doesn't exist
    st = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir = "{}-{}-{}-{}-{}".format(
        args.kernel_function, args.frank_wolfe, args.dataset, args.model, st
    )
    save_path = Path(args.save_path) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    utils.dump_args_to_json(args, save_path)

    if args.kernel_function == "mean_linear":
        kernel_mat_func = kernels.mean_embedding_linear_kernel_matrix
    else:
        raise Exception("Kernel {} not implemented".format(args.kernel_function))

    if args.frank_wolfe == "kernel_herding":
        fw_class = optimisation.KernelHerding
    elif args.frank_wolfe == "line_search":
        fw_class = optimisation.FrankWolfeLineSearch
    else:
        raise Exception(
            "Frank Wolfe algorithm {} not implemented".format(args.frank_wolfe)
        )

    dataloader_train, dataloader_val, dataloader_test = generate_dataloaders(args)

    # And initialise the model we will be using
    if args.model == "cnn":
        if args.dataset == "miniimagenet":
            in_channels = 3
        elif args.dataset == "omniglot":
            in_channels = 1

        def get_new_model_instance():
            model = ConvolutionalNeuralNetwork(
                in_channels=in_channels,
                out_features=args.n_way,
                hidden_size=args.num_filters,
            )
            return model

    elif args.model == "mlp":
        raise NotImplementedError
    else:
        raise Exception("Please choose cnn or mlp")

    # NOTE: This means we save t=0 and t=end in addition to end of each t
    _shape = (
        args.n_runs,
        (args.n_train_batches // args.evaluate_every) + 1,
        args.num_epochs,
    )
    all_test_loss_uniform = np.zeros(_shape)
    all_test_acc_uniform = np.zeros(_shape)
    all_test_loss_fw = np.zeros(_shape)
    all_test_acc_fw = np.zeros(_shape)
    for run in range(args.n_runs):
        sampled_batches_train = utils.aggregate_sampled_task_batches(
            dataloader_train, args.n_train_batches
        )
        sampled_batches_train_fw = get_active_learning_batches(
            sampled_batches_train, kernel_mat_func, fw_class
        )
        sampled_batches_test = utils.aggregate_sampled_task_batches(
            dataloader_test, args.n_test_batches
        )
        test_loss, test_acc = run_training_loop(
            sampled_batches_train,
            sampled_batches_test,
            get_new_model_instance,
            args.lr_meta,
            args.lr_inner,
            args.first_order,
            args.evaluate_every,
            args.num_epochs,
            args.device,
        )
        test_loss_fw, test_acc_fw = run_training_loop(
            sampled_batches_train_fw,
            sampled_batches_test,
            get_new_model_instance,
            args.lr_meta,
            args.lr_inner,
            args.first_order,
            args.evaluate_every,
            args.num_epochs,
            args.device,
        )
        all_test_loss_uniform[run, :, :] = test_loss
        all_test_acc_uniform[run, :, :] = test_acc
        all_test_loss_fw[run, :, :] = test_loss_fw
        all_test_acc_fw[run, :, :] = test_acc_fw
        logging.info("run {} out of {} completed".format(run + 1, args.n_runs))

    utils.dump_runs_to_npy(
        all_test_loss_uniform,
        all_test_acc_uniform,
        all_test_loss_fw,
        all_test_acc_fw,
        save_path,
    )


if __name__ == "__main__":
    logging.info("Reading command line arguments")
    args = arguments.parse_args()
    logging.info("Entering run()")
    run(args)
    logging.info("Finished successfully")
