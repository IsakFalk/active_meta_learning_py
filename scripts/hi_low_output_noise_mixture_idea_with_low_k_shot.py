import argparse
import copy
import logging
from functools import partial
from pathlib import Path

import hickle as hkl
import learn2learn as l2l
import numpy as np
import torch as th
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.utils.data import DataLoader

from active_meta_learning.data import (
    EnvironmentDataSet,
    GaussianNoiseMixture,
    HypercubeWithKVertexGaussian,
)
from active_meta_learning.data_utils import (
    aggregate_sampled_task_batches,
    convert_batches_to_fw_form,
    get_task_parameters,
    reorder_list,
)
from active_meta_learning.experiment_utils import save_mmd_experiment_plots
from active_meta_learning.kernels import (
    gaussian_kernel_matrix,
    gaussian_kernel_mmd2_matrix,
    median_heuristic,
)
from active_meta_learning.optimisation import KernelHerding
from hpc_cluster.utils import extract_csv_to_dict

# To use on server
import matplotlib as mpl  # isort:skip

mpl.use("Agg")  # isort:skip
import matplotlib.pyplot as plt  # isort:skip


logging.basicConfig(level=logging.INFO)


th.set_default_dtype(th.double)

GET_TASK_PARAMETERS_EVERY = 10
GET_PARAMETERS_EVERY = 10


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_features, out_features, bias).double()

    def forward(self, x=None):
        return self.l1(x)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


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
    noise_y_distribution,
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

    val_dataset = EnvironmentDataSet(
        k_shot, k_query, env, noise_w=0.0, noise_y=noise_y_distribution.sample
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, batch_size=meta_val_batch_size
    )
    val_batches = sample_tasks(val_dataloader, len(train_batches))

    model = Model(in_features=d, out_features=1, bias=False)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = meta_optimiser(params=maml.parameters(), lr=meta_lr)
    loss = nn.MSELoss(reduction="sum")

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
    noise_y_distribution,
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
    noise_y_distribution.reset_history()
    train_dataset = EnvironmentDataSet(
        k_shot, k_query, env, noise_w=0.0, noise_y=noise_y_distribution.sample
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=meta_train_batch_size,
    )
    train_batches = aggregate_sampled_task_batches(train_dataloader, num_train_batches)
    logging.info("Done!")

    # Save experimental data
    experiment_data = dict()

    ### Save train task data
    # Batch the mixture history properly
    # And save in numpy array
    _mixture_history = []
    for i in range(num_train_batches):
        _mixture_history.append(
            noise_y_distribution.mixture_history[i : i + meta_train_batch_size]
        )
    _mixture_history = np.array(_mixture_history)
    assert _mixture_history.shape == (
        num_train_batches,
        meta_train_batch_size,
    ), "mixture_history should have shape {}, has shape {}".format(
        (num_train_batches, meta_train_batch_size), _mixture_history.shape
    )
    experiment_data["train_tasks"] = {
        "task_ws": get_task_parameters(train_batches),
        "mixture_history": _mixture_history.copy(),
    }

    logging.info("Run training for uniform sampling")
    # Uniform sampling
    meta_train_errors, meta_val_errors, model_parameters = run(
        train_batches,
        d,
        noise_y_distribution,
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
        "sampled_order": np.arange(0, num_train_batches),
    }
    experiment_data["uniform"]["mixture_history"] = _mixture_history
    logging.info("Done!")

    # KH sampling
    logging.info("Generating KH batches (data space)")
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
        noise_y_distribution,
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
    experiment_data["kh_D"] = {
        "meta_train_errors": meta_train_errors,
        "meta_val_errors": meta_val_errors,
        "theta_0": model_parameters,
        "sampled_order": kh_D.sampled_order,
    }
    experiment_data["kh_D"]["mixture_history"] = np.array(
        reorder_list(_mixture_history, kh_D.sampled_order)
    )
    # KH sampling (in weight space)
    # Process sampled tasks
    train_task_ws = get_task_parameters(train_batches)
    # Get KH (weight) ordering
    logging.info("Generating KH batches (weight space)")
    s2_w = median_heuristic(squareform(pdist(train_task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(train_task_ws, s2_w)
    # Run herding
    kh_w = KernelHerding(K_w)
    kh_w.run()
    logging.info("Run training for KH sampling")
    meta_train_errors, meta_val_errors, model_parameters = run(
        reorder_train_batches(train_batches, kh_w.sampled_order),
        d,
        noise_y_distribution,
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
    experiment_data["kh_w"] = {
        "meta_train_errors": meta_train_errors,
        "meta_val_errors": meta_val_errors,
        "theta_0": model_parameters,
        "sampled_order": kh_w.sampled_order,
    }
    experiment_data["kh_w"]["mixture_history"] = np.array(
        reorder_list(_mixture_history, kh_D.sampled_order)
    )
    ### MMD experiment
    experiment_data["mmd"] = dict()
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

    save_mmd_experiment_plots(
        env, train_batches, K_w, K_D, kh_w.sampled_order, kh_D.sampled_order, save_path
    )

    logging.info("Done!")

    return experiment_data


def main(
    d,
    env,
    noise_y_distribution,
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
        noise_y_distribution,
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

    train_task_ws = experiment_data["train_tasks"]["task_ws"]
    noise_mixture_history = experiment_data["kh_D"]["mixture_history"]
    mean_task_w = train_task_ws.mean(axis=0)

    logging.info("Plotting")
    # Learning curves
    fig, ax = plt.subplots(2, 1, sharex=True)
    uniform_data = experiment_data["uniform"]
    val_t = np.arange(0, num_train_batches, collect_val_data_every)
    ax[0].plot(
        uniform_data["meta_train_errors"],
        color="blue",
        label="meta train error (U)",
        alpha=0.1,
    )
    ax[0].plot(
        val_t,
        uniform_data["meta_val_errors"],
        color="blue",
        label="meta val error (U)",
        linestyle="--",
    )
    kh_D_data = experiment_data["kh_D"]
    ax[0].plot(
        kh_D_data["meta_train_errors"],
        color="red",
        alpha=0.1,
        label="meta train error (KH on data)",
    )
    ax[0].plot(
        val_t,
        kh_D_data["meta_val_errors"],
        color="red",
        linestyle="--",
        label="meta val error (KH on data)",
    )
    kh_w_data = experiment_data["kh_w"]
    ax[0].plot(
        kh_w_data["meta_train_errors"],
        color="orange",
        alpha=0.1,
        label="meta train error (KH on weights)",
    )
    ax[0].plot(
        val_t,
        kh_w_data["meta_val_errors"],
        color="orange",
        linestyle="--",
        label="meta val error (KH on weights)",
    )
    ax[0].legend()
    ax[0].set_title("learning curves")
    ax[0].set_ylabel("MSE")
    # Plot number of hi-noise in batch per timestep
    ax[1].plot(
        np.arange(0, num_train_batches),
        np.sum(uniform_data["mixture_history"], axis=1),
        color="blue",
        linestyle="",
        marker="o"
    )
    ax[1].plot(
        np.arange(0, num_train_batches),
        np.sum(kh_D_data["mixture_history"], axis=1),
        color="red",
        linestyle="",
        marker="x"
    )
    ax[1].set_title("number of low-noise dataset in chosen batch at time t (moving average)")
    ax[1].set_xlabel('timestep')
    ax[1].set_ylabel('No. low-noise datsets in batch')
    ax[1].set_ylim([0.0, meta_train_batch_size])
    ax[1].axhline(meta_train_batch_size / 2, color="black", label="Average number of low-noise tasks in one batch")
    ax[1].legend()

    fig.savefig(save_path / "learning_curves.png")

    # Euclidean distance
    def norm_cummean_to_mean_task_w(reordered_task_ws, mean_task_w):
        cumulative_means = reordered_task_ws.cumsum(axis=0)
        cumulative_means = cumulative_means / np.arange(
            1, 1 + cumulative_means.shape[0]
        ).reshape(-1, 1)
        dist_reordered_task_means_to_task_mean = np.linalg.norm(
            cumulative_means - mean_task_w, axis=1
        )
        return dist_reordered_task_means_to_task_mean

    mmd_data = experiment_data["mmd"]
    fig, ax = plt.subplots()
    uniform_data = experiment_data["uniform"]
    uniform_params = np.concatenate(uniform_data["theta_0"])
    val_t = np.arange(0, num_train_batches, GET_TASK_PARAMETERS_EVERY)
    ax.plot(
        val_t,
        np.linalg.norm(uniform_params - mean_task_w, axis=1),
        color="blue",
        label="norm of parameters from meta_train mean (Uniform)",
    )
    kh_D_data = experiment_data["kh_D"]
    kh_D_params = np.concatenate(kh_D_data["theta_0"])
    val_t = np.arange(0, num_train_batches, GET_TASK_PARAMETERS_EVERY)
    ax.plot(
        val_t,
        np.linalg.norm(kh_D_params - mean_task_w, axis=1),
        color="red",
        label="norm of parameters from meta_train mean (KH data)",
    )
    fig.savefig(save_path / "distance_to_mean.png")
    logging.info("Done!")

    logging.info("Dumping experimental data")
    # dump params
    hkl.dump(experiment_data, save_path / "experiment_data.hkl")
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster parameters (see hpc_cluster library docs)"
    )
    parser.add_argument(
        "--csv_path", type=str,
    )
    parser.add_argument(
        "--extract_line", type=int,
    )
    parser.add_argument(
        "--output_dir", type=str,
    )

    args = parser.parse_args()
    args.csv_path = Path(args.csv_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # read in line of parameters
    param_dict = extract_csv_to_dict(args.csv_path, args.extract_line)
    d = param_dict["d"]
    k = param_dict["k"]
    noise_y_hi = param_dict["noise_y_hi"]
    # number of train examples should be very low
    # compared to actual dimensionality of problem
    k_shot = 4
    noise_y_low = 0.001
    # Form noise distribution
    p = np.array([0.95, 0.05])
    mus = np.array([0.0, 0.0])
    s2s = np.array([noise_y_hi, noise_y_low])
    noise_y_distribution = GaussianNoiseMixture(p, mus, s2s)
    noise_w = param_dict["s2_w"] / d
    env = HypercubeWithKVertexGaussian(d, k, noise_w)
    sgd = partial(optim.SGD, momentum=0.0)
    main(
        d,
        env,
        noise_y_distribution,
        k_shot=k_shot,
        k_query=25,
        meta_lr=0.1,
        meta_optimiser=sgd,
        fast_lr=0.1,
        num_train_batches=300,
        meta_train_batch_size=4,
        meta_val_batch_size=64,
        collect_val_data_every=5,
        median_heuristic_n_subsamples=300,
        adaptation_steps=1,
        device="cpu",
        save_path=args.output_dir,
    )

    logging.info("Dumping parameters")
    # dump params
    hkl.dump(param_dict, args.output_dir / "parameters.hkl")
    logging.info("Done!")
