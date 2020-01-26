import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import numpy as np

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

def get_outer_loss(batch, model, lr_inner, device, first_order, test=False):
    if test:
        _model = pickle.loads(pickle.dumps(model))
    else:
        _model = model
    _model.to(device)

    train_inputs, train_targets = batch["train"]
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device)
    batch_size = train_inputs.shape[0]

    test_inputs, test_targets = batch["test"]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device)
    if test:
        with torch.no_grad():
            outer_loss = torch.tensor(0.0, device=device)
            accuracy = torch.tensor(0.0, device=device)
    else:
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
        if test:
            with torch.no_grad():
                test_logit = _model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target, reduction="mean")
        else:
            test_logit = _model(test_input, params=params)
            outer_loss += F.cross_entropy(test_logit, test_target, reduction="mean")
        with torch.no_grad():
            accuracy += get_accuracy(test_logit, test_target)

    # clean up if in evaluation mode
    return_loss = outer_loss.div_(batch_size).item()
    return_acc = accuracy.div_(batch_size).item()
    if test:
        del _model
        del outer_loss
        del accuracy

    return return_loss, return_acc

def run_training_loop(
    sampled_batches_train,
    sampled_batches_test,
    model,
    lr_meta,
    lr_inner,
    first_order,
    evaluate_every,
    device
):
    test_loss = []
    test_acc = []
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

    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_meta)
    for idx, batch in enumerate(sampled_batches_train):
        logging.info("Train task batch: {}".format(idx))
        model.zero_grad()
        outer_loss, accuracy = get_outer_loss(
            batch, model, lr_inner, device, first_order, test=False
        )

        outer_loss.backward()
        meta_optimizer.step()
                
        if idx % evaluate_every == 0:
            # Get average test loss / acc
            temp_loss, temp_acc = evaluate_model(model)
            test_loss.append(temp_loss)
            test_acc.append(temp_acc)
            logging.info("Test loss / acc: {:.4f} / {:.4f}".format(temp_loss, temp_acc))

    return np.array(test_loss), np.array(test_acc)


def run(args):
    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # NOTE: Will use scheduler to output start and stopping
    # Just save seed as this is the only unique identifier for this run
    # data is where the data of the run will go
    save_dir = "seed_{}".format(
        args.seed
    )
    save_path = Path(args.save_path)
    save_path_data = save_path / save_dir
    # seed_{seed} need to be unique, crash if not, don't overwrite
    save_path_data.mkdir(parents=True, exist_ok=False)
    if args.write_config == "yes":
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
    
    logging.info("Generating dataloaders for dataset {}".format(args.dataset))
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

    logging.info("Sampling batched train datasets")
    sampled_batches_train = utils.aggregate_sampled_task_batches(
        dataloader_train, args.n_train_batches
    )
    logging.info("Generating fw train batch order")
    with torch.no_grad():
        sampled_batches_train_fw = get_active_learning_batches(
            sampled_batches_train, kernel_mat_func, fw_class
        )
    logging.info("Sampling batched test datasets")
    sampled_batches_test = utils.aggregate_sampled_task_batches(
        dataloader_test, args.n_test_batches
    )
    logging.info("Run training: Uniform")
    test_loss_uniform, test_acc_uniform = run_training_loop(
        sampled_batches_train,
        sampled_batches_test,
        get_new_model_instance(),
        args.lr_meta,
        args.lr_inner,
        args.first_order,
        args.evaluate_every,
        args.device
    )
    logging.info("Run training: FW")
    test_loss_fw, test_acc_fw = run_training_loop(
        sampled_batches_train_fw,
        sampled_batches_test,
        get_new_model_instance(),
        args.lr_meta,
        args.lr_inner,
        args.first_order,
        args.evaluate_every,
        args.device
    )
    logging.info("Run completed")

    utils.dump_runs_to_npy(
        test_loss_uniform,
        test_acc_uniform,
        test_loss_fw,
        test_acc_fw,
        save_path_data,
    )
    logging.info("Loss and accuracy dumped to npy")


if __name__ == "__main__":
    logging.info("Reading command line arguments")
    args = arguments.parse_args()
    logging.info("Entering run()")
    run(args)
    logging.info("Finished successfully")
