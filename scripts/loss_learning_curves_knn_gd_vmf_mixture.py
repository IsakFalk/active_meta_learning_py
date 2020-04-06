import argparse
from pathlib import Path
import logging


import hickle as hkl
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

from active_meta_learning.data import EnvironmentDataSet, VonMisesFisherMixture
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

logging.basicConfig(level=logging.INFO)


GET_LOSS_EVERY = 1


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


def create_vmf_params(d, k_mixtures, kappa):
    mus = np.random.randn(k_mixtures, d)
    mus /= np.linalg.norm(mus, axis=1).reshape(-1, 1)
    kappas = np.ones(k_mixtures) * kappa
    ps = np.ones(k_mixtures) / k_mixtures
    return mus, kappas, ps


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
    k_mixtures = param_dict["k_mixtures"]
    # kappa = s2**-1
    kappa = param_dict["kappa"]
    k_nn = param_dict["k_nn"]

    k_shot = 10
    k_query = 15
    until_t = 50
    median_heuristic_n_subsamples = 300
    meta_train_batches = 400
    meta_train_batch_size = 1  # Hardcoded
    meta_val_batches = 300
    meta_val_batch_size = 1  # Hardcoded
    meta_test_batches = 1000
    meta_test_batch_size = 1  # Hardcoded
    mus, kappas, ps = create_vmf_params(d, k_mixtures, kappa)
    env = VonMisesFisherMixture(mus, kappas, ps)

    assert (
        until_t % GET_LOSS_EVERY == 0
    ), "meta_train_batches needs to be divisible by GET_LOSS_EVERY"

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
    dataset_indices["uniform"] = np.arange(meta_train_batches)
    logging.info("Done")
    # KH weights
    logging.info("KH weights")
    s2_w = median_heuristic(squareform(pdist(train_task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(train_task_ws, s2_w)
    kh_w = KernelHerding(K_w)
    kh_w.run()
    dataset_indices["kh_weights"] = kh_w.sampled_order
    logging.info("Done")
    # KH data
    logging.info("KH Data")
    K_D = gaussian_kernel_mmd2_matrix(train_batches_kh, median_heuristic_n_subsamples)
    kh_D = KernelHerding(K_D)
    kh_D.run()
    dataset_indices["kh_data"] = kh_D.sampled_order
    logging.info("Done")

    logging.info("Defining models")

    def get_task_ws(task_ws, indices):
        return task_ws[indices]

    distance = lambda x, y: ((x - y) ** 2).sum()
    # We use some reasonable number of prototypes
    # to run cross-validation.
    # Later we will generate learning curves for a large range of number of
    # prototypes ordered by KH or uniform
    cross_val_prototypes = 50
    model_u = MetaKNNGD(
        prototypes=get_task_ws(
            train_task_ws, dataset_indices["uniform"][:cross_val_prototypes]
        ),
        k=k_nn,
        distance=distance,
    )
    model_kh_w = MetaKNNGD(
        prototypes=get_task_ws(
            train_task_ws, dataset_indices["kh_weights"][:cross_val_prototypes]
        ),
        k=k_nn,
        distance=distance,
    )
    model_kh_D = MetaKNNGD(
        prototypes=get_task_ws(
            train_task_ws, dataset_indices["kh_data"][:cross_val_prototypes]
        ),
        k=k_nn,
        distance=distance,
    )
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

    logging.info("Getting learning curves for meta test error")
    meta_test_error = dict()
    # shape: (n_runs, meta_val_batches)
    n_runs = until_t // GET_LOSS_EVERY
    meta_test_error["uniform"] = np.zeros((n_runs, meta_test_batches))
    meta_test_error["kh_weights"] = np.zeros((n_runs, meta_test_batches))
    meta_test_error["kh_data"] = np.zeros((n_runs, meta_test_batches))
    for i, num_prototypes in enumerate(range(0, until_t, GET_LOSS_EVERY)):
        model_u.prototypes = get_task_ws(
            train_task_ws, dataset_indices["uniform"][: num_prototypes + 1]
        )
        model_kh_w.prototypes = get_task_ws(
            train_task_ws, dataset_indices["kh_weights"][: num_prototypes + 1]
        )
        model_kh_D.prototypes = get_task_ws(
            train_task_ws, dataset_indices["kh_data"][: num_prototypes + 1]
        )
        for j, test_task in enumerate(test_batches):
            # uniform
            meta_test_error["uniform"][i, j] = model_u.transfer_risk(test_task)
            # kh weights
            meta_test_error["kh_weights"][i, j] = model_kh_w.transfer_risk(test_task)
            # kh data
            meta_test_error["kh_data"][i, j] = model_kh_D.transfer_risk(test_task)
    logging.info("Done")

    t = np.arange(0, until_t, GET_LOSS_EVERY)
    # We just plot a learning curve with error bars of 1 std
    def plot_sns_tsplot(meta_test_error, t, ax):
        """mean_test_error is dict with keys being the algorithm and values (n_runs, meta_val_batches)"""
        df_list = []
        for algorithm in meta_test_error.keys():
            df = pd.DataFrame(data=meta_test_error[algorithm])
            test_batch_columns = ["meta_test_batch{}".format(col) for col in df.columns]
            df.columns = test_batch_columns
            df["t"] = t
            df = pd.melt(
                df,
                id_vars=["t"],
                value_vars=test_batch_columns,
                var_name="meta_test_batch",
                value_name="MSE",
            )
            df["algorithm"] = algorithm
            df.columns
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        sns.lineplot(x="t", y="MSE", hue="algorithm", data=df, ax=ax)
        return ax

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_sns_tsplot(meta_test_error, t, ax)
    fig.savefig(
        args.output_dir
        / "learning_curves-{}.png".format(stringify_parameter_dictionary(param_dict))
    )

    # Dump data
    # Params
    hkl.dump(param_dict, args.output_dir / "parameters.hkl")
    # Experiment data
    hkl.dump(meta_test_error, args.output_dir / "experiment_data.hkl")

    logging.info("Done")
    logging.info("Good bye!")
