import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn.functional as F
import numpy as np

import utils
from torchmeta_utils import update_parameters
from models import MultiLayerPerceptron
import optimisation
import kernels
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
    train_targets = train_targets.to(device=device)
    batch_size = train_inputs.shape[0]

    test_inputs, test_targets = batch["test"]
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device)
    if test:
        with torch.no_grad():
            outer_loss = torch.tensor(0.0, device=device)
    else:
        outer_loss = torch.tensor(0.0, device=device)

    for task_idx, (train_input, train_target, test_input, test_target) in enumerate(
        zip(train_inputs, train_targets, test_inputs, test_targets)
    ):
        train_output = _model(train_input.float())
        inner_loss = F.mse_loss(train_output, train_target.float(), reduction="mean")

        _model.zero_grad()
        for _ in range(num_grad_steps_inner):
            params = update_parameters(
                _model, inner_loss, step_size=lr_inner, first_order=first_order
            )
        if test:
            with torch.no_grad():
                test_output = _model(test_input.float(), params=params)
                outer_loss += F.mse_loss(
                    test_output, test_target.float(), reduction="mean"
                )
        else:
            test_output = _model(test_input.float(), params=params)
            outer_loss += F.mse_loss(test_output, test_target.float(), reduction="mean")

    # clean up if in evaluation mode
    return_loss = outer_loss.div_(batch_size)
    if test:
        del _model

    return return_loss


def run_training_loop(sampled_batches_train, sampled_batches_test, model, args):
    lr_inner = args.lr_inner
    lr_meta = args.lr_meta
    device = args.device
    first_order = args.first_order
    evaluate_every = args.evaluate_every
    num_grad_steps_inner = args.num_grad_steps_inner
    num_grad_steps_meta = args.num_grad_steps_meta

    test_loss = []

    def evaluate_model(model):
        temp_loss = []
        for batch in sampled_batches_test:
            model.zero_grad()
            outer_loss = get_outer_loss(batch, model, args, test=True)
            temp_loss.append(outer_loss.squeeze().item())
        return np.array(temp_loss).mean()

    model.to(device=device)
    model.train()
    if args.meta_optimizer == "adam":
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_meta)
    elif args.meta_optimizer == "sgd":
        meta_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    else:
        raise Exception(
            "Optimizer {} not implemented".format(args.kernel_meta_optimizer)
        )
    for idx, batch in enumerate(sampled_batches_train):
        for _ in range(num_grad_steps_meta):
            model.zero_grad()
            outer_loss = get_outer_loss(batch, model, args, test=False)
            outer_loss.backward()
            meta_optimizer.step()

        del outer_loss

        if idx % evaluate_every == 0:
            # Get average test loss / acc
            temp_loss = evaluate_model(model)
            test_loss.append(temp_loss)
            logging.info("Iteration: {}, test loss: {:.4f}".format(idx, temp_loss))

    return np.array(test_loss)


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

    if args.kernel_function == "double_gaussian_kernel":
        kernel_mat_func = kernels.gaussian_kernel_mmd2_matrix
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
    dataloader_train, dataloader_val, dataloader_test = utils.generate_dataloaders(args)

    def get_new_model_instance():
        model = MultiLayerPerceptron(in_dim=1, out_dim=1, hidden_dim=args.hidden_dim,)
        return model

    logging.info("Sampling batched train datasets")
    sampled_batches_train = utils.aggregate_sampled_task_batches(dataloader_train)
    nn_sampled_batches_train = utils.convert_batches_to_nn_dict_train_test_form(
        sampled_batches_train, args.k_shot
    )
    logging.info("Generating fw train batch order")
    nn_sampled_batches_train_fw = utils.get_active_learning_batches(
        sampled_batches_train, args.k_shot, kernel_mat_func, fw_class
    )
    logging.info("Sampling batched test datasets")
    sampled_batches_test = utils.aggregate_sampled_task_batches(dataloader_test)
    nn_sampled_batches_test = utils.convert_batches_to_nn_dict_train_test_form(
        sampled_batches_test, args.k_shot
    )
    logging.info("Run training: Uniform")
    test_loss_uniform = run_training_loop(
        nn_sampled_batches_train,
        nn_sampled_batches_test,
        get_new_model_instance(),
        args,
    )
    logging.info("Run training: FW")
    test_loss_fw = run_training_loop(
        nn_sampled_batches_train_fw,
        nn_sampled_batches_test,
        get_new_model_instance(),
        args,
    )
    logging.info("Run completed")

    utils.dump_runs_to_npy(
        test_loss_uniform, test_loss_fw, save_path_data,
    )
    logging.info("Loss dumped to npy")


if __name__ == "__main__":
    logging.info("Reading command line arguments")
    args = arguments.parse_args()
    logging.info("Entering run()")
    run(args)
    logging.info("Finished successfully")
