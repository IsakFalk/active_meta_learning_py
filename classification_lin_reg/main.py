import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F
import numpy as np

import utils
from models import MultiLayerPerceptron
from torchmeta_utils import update_parameters, get_accuracy
import optimisation
import arguments


def get_outer_loss(batch, model, args, test=False):
    lr_inner = args.lr_inner
    device = args.device
    first_order = args.first_order
    num_grad_steps_inner = args.num_grad_steps_inner

    if test:
        _model = pickle.loads(pickle.dumps(model))
    else:
        _model = model
    _model.to(device)

    train_inputs, train_targets = batch["train"]
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device).long()
    batch_size = train_inputs.shape[0]

    test_inputs, test_targets = batch["test"]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device).long()
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
        inner_loss = F.cross_entropy(
            train_logit, train_target.squeeze(), reduction="mean"
        )

        _model.zero_grad()
        for _ in range(num_grad_steps_inner):
            params = update_parameters(
                _model, inner_loss, step_size=lr_inner, first_order=first_order
            )
        if test:
            with torch.no_grad():
                test_logit = _model(test_input, params=params)
                outer_loss += F.cross_entropy(
                    test_logit, test_target.squeeze(), reduction="mean"
                )
        else:
            test_logit = _model(test_input, params=params)
            outer_loss += F.cross_entropy(
                test_logit, test_target.squeeze(), reduction="mean"
            )
        with torch.no_grad():
            accuracy += get_accuracy(test_logit, test_target.squeeze())

    # clean up if in evaluation mode
    return_loss = outer_loss.div_(batch_size)
    return_acc = accuracy.div_(batch_size)
    if test:
        del _model

    return return_loss, return_acc


def run_training_loop(sampled_batches_train, sampled_batches_test, model, args):
    lr_inner = args.lr_inner
    lr_meta = args.lr_meta
    device = args.device
    first_order = args.first_order
    evaluate_every = args.evaluate_every
    num_grad_steps_inner = args.num_grad_steps_inner
    num_grad_steps_meta = args.num_grad_steps_meta

    test_loss = []
    test_acc = []

    def evaluate_model(model):
        temp_loss = []
        temp_acc = []
        for batch in sampled_batches_test:
            model.zero_grad()
            outer_loss, accuracy = get_outer_loss(batch, model, args, test=True)
            temp_loss.append(outer_loss.squeeze().item())
            temp_acc.append(accuracy.squeeze().item())
        return np.array(temp_loss).mean(), np.array(temp_acc).mean()

    model.to(device=device)
    model.train()
    if args.meta_optimizer == "adam":
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_meta)
    elif args.meta_optimizer == "sgd":
        meta_optimizer = torch.optim.SGD(model.parameters(), lr=lr_meta, momentum=0.0)
    else:
        raise Exception(
            "Optimizer {} not implemented".format(args.kernel_meta_optimizer)
        )
    for idx, batch in enumerate(sampled_batches_train):
        for _ in range(num_grad_steps_meta):
            model.zero_grad()
            outer_loss, accuracy = get_outer_loss(batch, model, args, test=False)
            outer_loss.backward()
            meta_optimizer.step()

        del outer_loss
        del accuracy

        if idx % evaluate_every == 0:
            # Get average test loss / acc
            temp_loss, temp_acc = evaluate_model(model)
            test_loss.append(temp_loss)
            test_acc.append(temp_acc)
            logging.info(
                "Iteration: {}, test loss: {:.4f}, test acc {:.4f}".format(
                    idx, temp_loss, temp_acc
                )
            )

    return np.array(test_loss), np.array(test_acc)


def run(args):
    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # NOTE: Will use scheduler to output start and stopping
    # Just save seed as this is the only unique identifier for this run
    # data is where the data of the run will go
    save_dir = "seed_{}".format(args.seed)
    save_path = Path(args.save_path)
    save_path_data = save_path / save_dir
    # seed_{seed} need to be unique, crash if not, don't overwrite
    save_path_data.mkdir(parents=True, exist_ok=False)
    if args.write_config == "yes":
        utils.dump_args_to_json(args, save_path)

    if args.frank_wolfe == "kernel_herding":
        fw_class = optimisation.KernelHerding
    elif args.frank_wolfe == "line_search":
        fw_class = optimisation.FrankWolfeLineSearch
    else:
        raise Exception(
            "Frank Wolfe algorithm {} not implemented".format(args.frank_wolfe)
        )

    logging.info("Generating dataloaders")
    dataloader_train, dataloader_val, dataloader_test = utils.generate_dataloaders_leo(
        args
    )

    # We'll be using MLP
    def get_new_model_instance():
        model = MultiLayerPerceptron(in_dim=640, out_dim=5, hidden_dim=40)
        return model

    logging.info("Sampling batched train datasets")
    sampled_batches_train = utils.aggregate_sampled_task_batches(
        dataloader_train, args.n_train_batches
    )
    logging.info("Generating fw train batch order")
    sampled_batches_train_fw = utils.get_active_learning_batches_lin_reg(
        sampled_batches_train, fw_class
    )
    logging.info("Sampling batched test datasets")
    sampled_batches_test = utils.aggregate_sampled_task_batches(
        dataloader_test, args.n_test_batches
    )
    logging.info("Run training: Uniform")
    test_loss_uniform, test_acc_uniform = run_training_loop(
        sampled_batches_train, sampled_batches_test, get_new_model_instance(), args
    )
    logging.info("Run training: FW")
    test_loss_fw, test_acc_fw = run_training_loop(
        sampled_batches_train_fw, sampled_batches_test, get_new_model_instance(), args
    )
    logging.info("Run completed")

    utils.dump_runs_to_npy(
        test_loss_uniform, test_acc_uniform, test_loss_fw, test_acc_fw, save_path_data,
    )
    logging.info("Loss and accuracy dumped to npy")


if __name__ == "__main__":
    logging.info("Reading command line arguments")
    args = arguments.parse_args()
    logging.info("Entering run()")
    run(args)
    logging.info("Finished successfully")
