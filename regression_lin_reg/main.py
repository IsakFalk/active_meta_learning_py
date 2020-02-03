import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import torch
import numpy as np

import utils
from models import MultiLayerPerceptron
import optimisation
import kernels
import arguments


def get_outer_loss(batch, model, args, test=False):
    inner_lambda = args.inner_regularization
    device = args.device
    batch_size = args.tasks_per_metaupdate
    batch_size = args.tasks_per_metaupdate
    n_train = args.k_shot
    n_test = args.k_query
    feature_dim = args.feature_dim

    def get_lin_reg_test_loss(
        model, train_phi_input, train_target, test_phi_input, test_target, outer_loss
    ):
        C = train_phi_input.t() @ train_phi_input + inner_lambda * n_train * torch.eye(
            feature_dim
        ).to(device=device)
        w = torch.inverse(C) @ train_phi_input.t() @ train_target
        pred_target = test_phi_input @ w
        outer_loss += torch.sum((pred_target - test_target) ** 2) / n_test
        return outer_loss

    if test:
        _model = pickle.loads(pickle.dumps(model))
    else:
        _model = model
    _model.to(device)

    train_inputs, train_targets = batch["train"]
    train_inputs = train_inputs.to(device=device).float()
    train_phi_inputs = _model(train_inputs)
    train_targets = train_targets.to(device=device).float()
    batch_size = train_inputs.shape[0]

    test_inputs, test_targets = batch["test"]
    test_inputs = test_inputs.to(device=device).float()
    test_phi_inputs = _model(test_inputs)
    test_targets = test_targets.to(device=device).float()
    if test:
        with torch.no_grad():
            outer_loss = torch.tensor(0.0, device=device)
    else:
        outer_loss = torch.tensor(0.0, device=device)

    for (
        task_idx,
        (train_phi_input, train_target, test_phi_input, test_target),
    ) in enumerate(zip(train_phi_inputs, train_targets, test_phi_inputs, test_targets)):

        if test:
            with torch.no_grad():
                return_loss = get_lin_reg_test_loss(
                    model.to(device=device),
                    train_phi_input.to(device=device),
                    train_target.to(device=device),
                    test_phi_input.to(device=device),
                    test_target.to(device=device),
                    outer_loss.to(device=device),
                )
        else:
            return_loss = get_lin_reg_test_loss(
                model.to(device=device),
                train_phi_input.to(device=device),
                train_target.to(device=device),
                test_phi_input.to(device=device),
                test_target.to(device=device),
                outer_loss.to(device=device),
            )

    # clean up if in evaluation mode
    return_loss = outer_loss.div_(batch_size)
    if test:
        del _model

    return return_loss


def run_training_loop(sampled_batches_train, sampled_batches_test, model, args):
    lr_meta = args.lr_meta
    device = args.device
    evaluate_every = args.evaluate_every
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
        model = MultiLayerPerceptron(
            in_dim=1, out_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        )
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
