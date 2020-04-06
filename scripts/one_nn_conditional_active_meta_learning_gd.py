import argparse
from pathlib import Path
import logging

from tqdm import tqdm
import numpy as np
import torch as th
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from active_meta_learning.data import (
    EnvironmentDataSet,
    GaussianNoiseMixture,
    HypercubeWithKVertexGaussian,
)
from active_meta_learning.data_utils import (
    aggregate_sampled_task_batches,
    convert_batches_to_fw_form,
    get_task_parameters,
    remove_batched_dimension_in_D,
    convert_batches_to_np,
    coalesce_train_and_test_in_dicts,
    reorder_list,
)
from active_meta_learning.kernels import (
    gaussian_kernel_matrix,
    gaussian_kernel_mmd2_matrix,
    median_heuristic,
)
from active_meta_learning.project_parameters import SCRIPTS_DIR
from active_meta_learning.optimisation import KernelHerding
from hpc_cluster.utils import extract_csv_to_dict


def stringify_parameter_dictionary(d, joiner="-"):
    l = []
    for key, val in d.items():
        if type(val) == float:
            l.append("{!s}={:.2f}".format(key, val))
        elif type(val) == int:
            l.append("{!s}={}".format(key, val))
        else:
            l.append("{!s}={}".format(key, val))
    return joiner.join(l)


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


def unpack_batch(batch, t):
    """Return t'th datasets in batch"""
    # Unpack batch
    train_input, train_target = map(lambda x: x[t], batch["train"])
    test_input, test_target = map(lambda x: x[t], batch["test"])
    return train_input, train_target, test_input, test_target


def npfy_batches(batches):
    new_batches = []
    for batch in batches:
        new_dict = {}
        new_dict["train"] = (
            batch["train"][0].numpy().squeeze(),
            batch["train"][1].numpy().squeeze(),
        )
        new_dict["test"] = (
            batch["test"][0].numpy().squeeze(),
            batch["test"][1].numpy().squeeze(),
        )
        new_dict["w"] = batch["w"].squeeze()
        new_batches.append(new_dict)
    return new_batches


class MetaKNNGD:
    """Meta k_nn on weights, prediction on data"""

    def __init__(
        self, prototypes, distance, k=1, adaptation_steps=1, learning_rate=0.01
    ):
        self.prototypes = prototypes
        self.distance = distance
        self.k = k
        self.adaptation_steps = adaptation_steps
        self.learning_rate = learning_rate

    def fit(self, task):
        # Unpack task
        X_tr, y_tr = task["train"]
        w = task["w"]
        # Find best k prototypes
        self.closest_prototypes = []
        distances_to_w = []
        for prototype in self.prototypes:
            distances_to_w.append(distance(prototype, w))
        distances_to_w = np.array(distances_to_w)
        self.closest_prototypes = self.prototypes[np.argsort(distances_to_w)[: self.k]]
        # Find w_hat running GD
        self.w_0 = np.mean(self.closest_prototypes, axis=0)
        w = self.w_0
        for _ in range(self.adaptation_steps):
            w -= self.learning_rate * 2 * X_tr.T @ (X_tr @ w - y_tr)
        self.w_hat = w

    def predict(self, task):
        # Unpack task
        test_input, _ = task["test"]
        return test_input @ self.w_hat

    def transfer_risk(self, task, loss=mean_squared_error):
        self.fit(task)
        _, test_target = task["test"]
        y_test_hat = self.predict(task)
        return loss(y_test_hat, test_target)


def cross_validate(model, lrs, val_batches):
    opt_lr = None
    opt_loss = np.inf
    for lr in lrs:
        model.learning_rate = lr
        for val_task in val_batches:
            current_loss = model.transfer_risk(val_task)
            if current_loss < opt_loss:
                opt_loss = current_loss
                opt_lr = lr
                return opt_lr, opt_loss


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

    logging.info("Reading args")
    args = parser.parse_args()
    args.csv_path = Path(args.csv_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Done!")
    logging.info("Extracting args")
    # read in line of parameters
    param_dict = extract_csv_to_dict(args.csv_path, args.extract_line)
    d = param_dict["d"]
    k = param_dict["k"]
    noise_w = param_dict["noise_w"]
    num_prototypes = param_dict["num_prototypes"]

    k_shot = 10
    k_query = 15
    median_heuristic_n_subsamples = 300
    meta_train_batches = 300
    meta_train_batch_size = 1  # Hardcoded
    meta_val_batches = 300
    meta_val_batch_size = 1  # Hardcoded
    meta_test_batches = 3000
    meta_test_batch_size = 1  # Hardcoded
    env = HypercubeWithKVertexGaussian(d, k=k, s2=noise_w / d)

    logging.info("Generating meta-train batches")
    # Sample meta-train
    train_dataset = EnvironmentDataSet(k_shot, k_query, env, noise_w=0.0, noise_y=0.0)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=meta_train_batch_size,
    )
    train_batches = aggregate_sampled_task_batches(train_dataloader, meta_train_batches)
    train_batches_kh = convert_batches_to_fw_form(train_batches)
    train_batches = npfy_batches(train_batches)
    train_task_ws = get_task_parameters(train_batches)
    logging.info("Done")

    logging.info("Generating meta-val batches")
    # Sample meta-val
    val_dataset = EnvironmentDataSet(k_shot, k_query, env, noise_w=0.0, noise_y=0.0)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, batch_size=meta_val_batch_size
    )
    val_batches = aggregate_sampled_task_batches(val_dataloader, meta_val_batches)
    val_batches_kh = convert_batches_to_fw_form(val_batches)
    val_batches = npfy_batches(val_batches)
    val_task_ws = get_task_parameters(val_batches)
    logging.info("Done")

    logging.info("Generating meta-test batches")
    # Sample meta-test
    test_dataset = EnvironmentDataSet(k_shot, k_query, env, noise_w=0.0, noise_y=0.0)
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        batch_size=meta_test_batch_size,
    )
    test_batches = aggregate_sampled_task_batches(test_dataloader, meta_test_batches)
    test_batches_kh = convert_batches_to_fw_form(test_batches)
    test_batches = npfy_batches(test_batches)
    test_task_ws = get_task_parameters(test_batches)
    logging.info("Done")

    logging.info("Generating prototypes for:")
    dataset_indices = dict()
    # Sample k from meta-train using
    # Uniform
    logging.info("Uniform")
    dataset_indices["uniform"] = np.arange(num_prototypes)
    logging.info("Done")
    # KH weights
    logging.info("KH weights")
    s2_w = median_heuristic(squareform(pdist(train_task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(train_task_ws, s2_w)
    kh_w = KernelHerding(K_w, stop_t=num_prototypes)
    kh_w.run()
    dataset_indices["kh_weights"] = kh_w.sampled_order
    logging.info("Done")
    # KH data
    logging.info("KH Data")
    K_D = gaussian_kernel_mmd2_matrix(train_batches_kh, median_heuristic_n_subsamples)
    kh_D = KernelHerding(K_D, stop_t=num_prototypes)
    kh_D.run()
    dataset_indices["kh_data"] = kh_D.sampled_order
    logging.info("Done")

    logging.info("Defining models")
    distance = lambda x, y: ((x - y) ** 2).sum()
    model_u = MetaKNNGD(train_task_ws[dataset_indices["uniform"]], distance=distance)
    model_kh_w = MetaKNNGD(
        train_task_ws[dataset_indices["kh_weights"]], distance=distance
    )
    model_kh_D = MetaKNNGD(train_task_ws[dataset_indices["kh_data"]], distance=distance)
    logging.info("Done")
    logging.info("Cross validation against learning rate")
    # Hyperparams
    lrs = np.geomspace(1e-3, 1e0, 4)
    lr_u, u_cross_val_loss = cross_validate(model_u, lrs, val_batches)
    lr_kh_w, kh_w_cross_val_loss = cross_validate(model_kh_w, lrs, val_batches)
    lr_kh_D, kh_D_cross_val_loss = cross_validate(model_kh_D, lrs, val_batches)
    logging.info("Optimal learning rates and losses found:")
    logging.info("U: lr={}, loss={}".format(lr_u, u_cross_val_loss))
    logging.info("KH_W: lr={}, loss={}".format(lr_kh_w, kh_w_cross_val_loss))
    logging.info("KH_D: lr={}, loss={}".format(lr_kh_D, kh_D_cross_val_loss))
    # Set optimal learning rate
    model_u.learning_rate = lr_u
    model_kh_w.learning_rate = lr_kh_w
    model_kh_D.learning_rate = lr_kh_D

    logging.info("Getting meta test errors")
    meta_test_error = dict()
    meta_test_error["uniform"] = []
    meta_test_error["kh_weights"] = []
    meta_test_error["kh_data"] = []
    for test_task in test_batches:
        # uniform
        meta_test_error["uniform"].append(model_u.transfer_risk(test_task))
        # kh weights
        meta_test_error["kh_weights"].append(model_kh_w.transfer_risk(test_task))
        # kh data
        meta_test_error["kh_data"].append(model_kh_D.transfer_risk(test_task))
    logging.info("Done")

    logging.info("Plotting")
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    df = pd.DataFrame(meta_test_error)
    # Historgram
    bins = int(meta_test_batches / 50)
    ax[0].hist(
        meta_test_error["uniform"],
        color="blue",
        alpha=0.4,
        label="uniform",
        bins=bins,
        density=True,
    )
    ax[0].axvline(np.mean(meta_test_error["uniform"]), color="blue", linestyle="--")
    ax[0].hist(
        meta_test_error["kh_weights"],
        color="orange",
        alpha=0.4,
        label="KH (weights)",
        bins=bins,
        density=True,
    )
    ax[0].axvline(
        np.mean(meta_test_error["kh_weights"]), color="orange", linestyle="--"
    )
    ax[0].hist(
        meta_test_error["kh_data"],
        color="red",
        alpha=0.4,
        label="KH (data)",
        bins=bins,
        density=True,
    )
    ax[0].axvline(np.mean(meta_test_error["kh_data"]), color="red", linestyle="--")
    ax[0].set_xlabel("MSE")
    ax[0].set_ylabel("Density")
    ax[0].set_title(
        "Histogram (meta-test MSE), setting: {}".format(
            stringify_parameter_dictionary(param_dict, joiner=", ")
        )
    )
    ax[0].legend()
    # Box plot
    df.boxplot(grid=False, rot=45, fontsize=15, ax=ax[1], showfliers=False)
    ax[1].set_title("Boxplot (outliers removed)")
    ax[1].set_ylabel("MSE")
    plt.tight_layout()

    fig.savefig(
        args.output_dir
        / "performance_plot-{}.png".format(stringify_parameter_dictionary(param_dict))
    )
    logging.info("Done")
    logging.info("Good bye!")
