"""
Experiment: We let the meta-train and meta-validation generating distributions be different

We want to see if when using the hypercube setup with a Gaussian at k vertices out of
d^2, does it matter that we sample these vertices when we train, or is it more important to
just get into the hypercube? (note: we start with theta_0 at a gaussian at [0, ..., 0], due
to pytorch parameter initialisation strategy, so any sampled train set should push theta_0 inwards).

This would show if the experimental setup is too easy with respect to maml, and thus if
using MMD to find the generating distribution does not really add anything beyond getting
theta_0 within some neighbourhood of the optimal initial value theta_0^*.
"""


import copy
import logging
from functools import partial

import hickle as hkl
import learn2learn as l2l
import numpy as np
import torch as th
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.utils.data import DataLoader

from active_meta_learning.data import (EnvironmentDataSet,
                                       HypercubeWithKVertexGaussian)
from active_meta_learning.data_utils import (aggregate_sampled_task_batches,
                                             convert_batches_to_fw_form,
                                             get_task_parameters)
from active_meta_learning.kernels import (gaussian_kernel_matrix,
                                          gaussian_kernel_mmd2_matrix,
                                          median_heuristic)
from active_meta_learning.optimisation import KernelHerding
from active_meta_learning.project_parameters import PROCESSED_DATA_DIR

# To use on server
import matplotlib as mpl  # isort:skip

mpl.use("Agg")  # isort:skip
import matplotlib.pyplot as plt  # isort:skip


logging.basicConfig(level=logging.INFO)


th.set_default_dtype(th.double)

GET_PARAMETERS_EVERY = 10


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_features, out_features, bias).double()

    def forward(self, x=None):
        return self.l1(x)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def create_environment_dataloader(k_shot, k_query, env, noise_w, noise_y, batch_size):
    # Create dataset and dataloader
    env_dataset = EnvironmentDataSet(
        k_shot, k_query, env, noise_w=noise_w, noise_y=noise_y
    )
    dataloader = DataLoader(
        env_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=env_dataset.collate_fn,
    )
    return dataloader


def sample_tasks(dataloader, N):
    # Sample metainstances
    sampled_batches = aggregate_sampled_task_batches(dataloader, N)
    return sampled_batches


def reorder_train_batches(train_batches, new_order):
    return [train_batches[i] for i in new_order]


def fast_adapt(
    train_input,
    train_target,
    val_input,
    val_target,
    learner,
    loss,
    adaptation_steps,
    device,
):
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    val_input = val_input.to(device)
    val_target = val_target.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(train_input), train_target)
        train_error /= len(train_input)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(val_input)
    val_error = loss(predictions, val_target)
    val_error /= len(val_input)
    return val_error


def unpack_batch(batch, t):
    """Return t'th datasets in batch"""
    # Unpack batch
    train_input, train_target = map(lambda x: x[t], batch["train"])
    test_input, test_target = map(lambda x: x[t], batch["test"])
    return train_input, train_target, test_input, test_target


def run(
    train_batches,
    d,
    env,
    k_shot,
    k_query,
    meta_lr,
    meta_optimiser,
    fast_lr,
    meta_train_batch_size,
    meta_val_batch_size,
    collect_val_data_every,
    adaptation_steps,
    device,
):
    meta_train_errors = []
    meta_val_errors = []
    model_parameters = []

    val_dataset = EnvironmentDataSet(k_shot, k_query, env)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, batch_size=meta_val_batch_size
    )
    val_batches = sample_tasks(val_dataloader, len(train_batches))

    model = Model(in_features=d, out_features=1, bias=False)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = meta_optimiser(params=maml.parameters(), lr=meta_lr)
    loss = nn.MSELoss(reduction="mean")

    for i, (train_batch, val_batch) in enumerate(zip(train_batches, val_batches)):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_val_error = 0.0
        if i % GET_PARAMETERS_EVERY == 0:
            model_parameters.append(copy.deepcopy(model.l1.weight.data.numpy()))
        for t in range(meta_train_batch_size):
            # Meta-train
            train_input, train_target, val_input, val_target = unpack_batch(
                train_batch, t
            )
            assert train_input.shape == (k_shot, d)
            assert train_target.shape == (k_shot, 1)
            assert val_input.shape == (k_query, d)
            assert val_target.shape == (k_query, 1)
            learner = maml.clone()
            val_error = fast_adapt(
                train_input,
                train_target,
                val_input,
                val_target,
                learner,
                loss,
                adaptation_steps,
                device,
            )
            # Gradients accumulate, so don't need to sum errors
            val_error.backward()
            meta_train_error += val_error.item()
        meta_train_errors.append(meta_train_error / meta_train_batch_size)

        if i % collect_val_data_every == 0:
            for t in range(meta_val_batch_size):
                # Meta-validation
                train_input, train_target, val_input, val_target = unpack_batch(
                    val_batch, t
                )
                assert train_input.shape == (k_shot, d)
                assert train_target.shape == (k_shot, 1)
                assert val_input.shape == (k_query, d)
                assert val_target.shape == (k_query, 1)
                learner = maml.clone()
                val_error = fast_adapt(
                    train_input,
                    train_target,
                    val_input,
                    val_target,
                    learner,
                    loss,
                    adaptation_steps,
                    device,
                )
                meta_val_error += val_error.item()
            meta_val_errors.append(meta_val_error / meta_val_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.mul_(1.0 / meta_train_batch_size)
        opt.step()

    return meta_train_errors, meta_val_errors, model_parameters


def experiment(
    d,
    env,
    k_shot,
    k_query,
    meta_lr,
    meta_optimiser,
    fast_lr,
    num_train_batches,
    meta_train_batch_size,
    meta_val_batch_size,
    collect_val_data_every,
    median_heuristic_n_subsamples,
    adaptation_steps,
    device,
    save_path,
):

    logging.info("Generating meta train data batches")
    # Split this into dumb and non-dumb training
    # Dumb
    dumb_train_env = HypercubeWithKVertexGaussian(
        env.d, 1, 0.0, mixture_vertices=[0.5 * np.ones(env.d)]
    )
    dumb_train_dataset = EnvironmentDataSet(k_shot, k_query, dumb_train_env)
    dumb_train_dataloader = DataLoader(
        dumb_train_dataset,
        collate_fn=dumb_train_dataset.collate_fn,
        batch_size=meta_train_batch_size,
    )
    dumb_train_batches = aggregate_sampled_task_batches(
        dumb_train_dataloader, num_train_batches
    )
    # Non-dumb
    train_env = env
    train_dataset = EnvironmentDataSet(k_shot, k_query, train_env)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=meta_train_batch_size,
    )
    train_batches = aggregate_sampled_task_batches(train_dataloader, num_train_batches)
    logging.info("Done!")

    # Save experimental data
    experiment_data = dict()

    # Dumb training
    meta_train_errors, meta_val_errors, model_parameters = run(
        dumb_train_batches,
        d,
        env,
        k_shot,
        k_query,
        meta_lr,
        meta_optimiser,
        fast_lr,
        meta_train_batch_size,
        meta_val_batch_size,
        collect_val_data_every,
        adaptation_steps,
        device,
    )
    experiment_data["dumb"] = {
        "meta_train_errors": meta_train_errors,
        "meta_val_errors": meta_val_errors,
        "theta_0": model_parameters,
        "norm_theta_0": [np.linalg.norm(w) for w in model_parameters],
    }
    logging.info("Done!")

    logging.info("Run training for uniform sampling")
    # Uniform sampling
    meta_train_errors, meta_val_errors, model_parameters = run(
        train_batches,
        d,
        env,
        k_shot,
        k_query,
        meta_lr,
        meta_optimiser,
        fast_lr,
        meta_train_batch_size,
        meta_val_batch_size,
        collect_val_data_every,
        adaptation_steps,
        device,
    )
    experiment_data["uniform"] = {
        "meta_train_errors": meta_train_errors,
        "meta_val_errors": meta_val_errors,
        "theta_0": model_parameters,
        "norm_theta_0": [np.linalg.norm(w) for w in model_parameters],
    }
    logging.info("Done!")

    # KH sampling
    logging.info("Generating KH batches")
    kh_train_batches = convert_batches_to_fw_form(train_batches)
    logging.info("Generating kernel matrix")
    K_D = gaussian_kernel_mmd2_matrix(kh_train_batches, median_heuristic_n_subsamples)
    kh_D = KernelHerding(K_D)
    logging.info("Running Kernel Herding")
    kh_D.run()
    logging.info("Run training for KH sampling")
    meta_train_errors, meta_val_errors, model_parameters = run(
        reorder_train_batches(train_batches, kh_D.sampled_order),
        d,
        env,
        k_shot,
        k_query,
        meta_lr,
        meta_optimiser,
        fast_lr,
        meta_train_batch_size,
        meta_val_batch_size,
        collect_val_data_every,
        adaptation_steps,
        device,
    )
    experiment_data["kh"] = {
        "meta_train_errors": meta_train_errors,
        "meta_val_errors": meta_val_errors,
        "theta_0": model_parameters,
        "norm_theta_0": [np.linalg.norm(w) for w in model_parameters],
    }
    logging.info("Done!")

    ### MMD experiment
    experiment_data["mmd"] = dict()
    # Process sampled tasks
    train_task_ws = get_task_parameters(train_batches)

    # Get KH (weight) ordering
    s2_w = median_heuristic(squareform(pdist(train_task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(train_task_ws, s2_w)
    # Run herding
    kh_w = KernelHerding(K_w)
    kh_w.run()
    # Dump data in experiment dict
    experiment_data["mmd"]["K_w"] = K_w
    experiment_data["mmd"]["s2_w"] = s2_w
    experiment_data["mmd"]["kh_w_ordering"] = kh_w.sampled_order

    # Get KH (data) ordering
    # Computation already done for learning curve experiment
    # Dump data in experiment dict
    experiment_data["mmd"]["K_D"] = K_D
    experiment_data["mmd"]["s2_D"] = None
    experiment_data["mmd"]["kh_D_ordering"] = kh_D.sampled_order

    logging.info("Done!")

    return experiment_data


def main(
    d,
    env,
    k_shot,
    k_query,
    meta_lr,
    meta_optimiser,
    fast_lr,
    num_train_batches,
    meta_train_batch_size,
    meta_val_batch_size,
    collect_val_data_every,
    median_heuristic_n_subsamples,
    adaptation_steps,
    device,
    save_path,
):
    experiment_data = experiment(
        d,
        env,
        k_shot,
        k_query,
        meta_lr,
        meta_optimiser,
        fast_lr,
        num_train_batches,
        meta_train_batch_size,
        meta_val_batch_size,
        collect_val_data_every,
        median_heuristic_n_subsamples,
        adaptation_steps,
        device="cpu",
        save_path=save_path,
    )

    logging.info("Plotting")
    # learning curves
    fig, ax = plt.subplots()
    uniform_data = experiment_data["uniform"]
    val_t = np.arange(0, num_train_batches, collect_val_data_every)
    ax.plot(
        uniform_data["meta_train_errors"],
        color="blue",
        label="meta train error (U)",
        alpha=0.1,
    )
    ax.plot(
        val_t,
        uniform_data["meta_val_errors"],
        color="blue",
        label="meta val error (U)",
        linestyle="--",
    )
    kh_data = experiment_data["kh"]
    ax.plot(
        kh_data["meta_train_errors"],
        color="red",
        alpha=0.1,
        label="meta train error (KH)",
    )
    ax.plot(
        val_t,
        kh_data["meta_val_errors"],
        color="red",
        linestyle="--",
        label="meta val error (KH)",
    )
    dumb_data = experiment_data["dumb"]
    ax.plot(
        dumb_data["meta_train_errors"],
        color="green",
        alpha=0.1,
        label="meta train error (DUMB)",
    )
    ax.plot(
        val_t,
        dumb_data["meta_val_errors"],
        color="green",
        linestyle="--",
        label="meta val error (DUMB)",
    )
    ax.legend()
    ax.set_title("Uniform vs KH vs dumb")
    fig.savefig(save_path / "learning_curves.png")
    # Weight norms
    fig, ax = plt.subplots()
    uniform_data = experiment_data["uniform"]
    val_t = np.arange(0, num_train_batches, GET_PARAMETERS_EVERY)
    ax.plot(
        uniform_data["norm_theta_0"], color="blue", label="norm of theta_0 (U)",
    )
    kh_data = experiment_data["kh"]
    ax.plot(
        kh_data["norm_theta_0"], color="red", label="norm of theta_0 (KH)",
    )
    dumb_data = experiment_data["dumb"]
    ax.plot(
        dumb_data["norm_theta_0"], color="green", label="norm of theta_0 (DUMB)",
    )
    ax.legend()
    ax.set_title("Uniform vs KH vs dumb (weights)")
    fig.savefig(save_path / "weight_norm_curves.png")
    logging.info("Done!")

    logging.info("Dumping experimental data")
    # dump params
    hkl.dump(experiment_data, save_path / "experiment_data.hkl")
    # with open(save_path / "experiment_data.pkl", "wb") as f:
    #     pickle.dump(experiment_data, f)
    logging.info("Done!")


if __name__ == "__main__":
    # read in line of parameters
    d = 20
    k = 3
    s2 = 0.01
    # Note: this is the meta-test / validation environment
    # the meta-train environment is set inside the
    # experiment function
    param_dict = {"d": d, "k": k, "s2": s2}
    output_dir = (
        PROCESSED_DATA_DIR / "train_on_same_task_maml_hypercube_with_k_vertex_gaussians"
    )
    output_dir.mkdir(exist_ok=True)
    env = HypercubeWithKVertexGaussian(d, k, s2)
    sgd = partial(optim.SGD, momentum=0.0)
    main(
        d,
        env,
        k_shot=10,
        k_query=25,
        meta_lr=0.4,
        meta_optimiser=sgd,
        fast_lr=0.4,
        num_train_batches=300,
        meta_train_batch_size=4,
        meta_val_batch_size=64,
        collect_val_data_every=5,
        median_heuristic_n_subsamples=500,
        adaptation_steps=1,
        device="cpu",
        save_path=output_dir,
    )

    logging.info("Dumping parameters")
    # dump params
    hkl.dump(param_dict, output_dir / "parameters.hkl")
    # with open(args.output_dir / "parameters.pkl", "wb") as f:
    #     pickle.dump(param_dict, f)
    logging.info("Done!")
