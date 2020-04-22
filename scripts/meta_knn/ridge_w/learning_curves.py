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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from active_meta_learning.data import EnvironmentDataSet, UniformSphere
from active_meta_learning.data_utils import (
    aggregate_sampled_task_batches,
    convert_batches_to_fw_form,
    get_task_parameters,
    remove_batched_dimension_in_D,
    convert_batches_to_np,
    coalesce_train_and_test_in_dicts,
    reorder_list,
    set_random_seeds,
)
from active_meta_learning.kernels import (
    mmd2,
    gaussian_kernel_matrix,
    gaussian_kernel_mmd2_matrix,
    median_heuristic,
)
from active_meta_learning.project_parameters import SCRIPTS_DIR, SETTINGS_DATA_DIR
from active_meta_learning.optimisation import KernelHerding
from hpc_cluster.utils import extract_csv_to_dict

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def stringify_parameter_dictionary(d, joiner="-"):
    l = []
    for key, val in d.items():
        if type(val) == float:
            l.append("{!s}={:.4f}".format(key, val))
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


def _mmd2_matrix(A, B, base_s2):
    assert len(A.shape) == 3
    m, n, d = A.shape
    assert len(B.shape) == 3
    p, q, l = B.shape
    M2 = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            K_xx = gaussian_kernel_matrix(A[i], s2=base_s2)
            K_yy = gaussian_kernel_matrix(B[j], s2=base_s2)
            K_xy = gaussian_kernel_matrix(A[i], B[j], s2=base_s2)
            M2[i, j] = mmd2(K_xx, K_yy, K_xy)
    return M2


def _gaussian_kernel_mmd2_matrix(A, B, base_s2, meta_s2):
    """Calculate the double gaussian kernel

    Calculate the double gaussian kernel matrix between A and B
    using base_s2 for the inner and meta_s2 for the outer level
    """
    M2 = _mmd2_matrix(A, B, base_s2)
    return np.exp(-0.5 * M2 / meta_s2)


def calculate_double_gaussian_median_heuristics(
    A, n_base_subsamples=None, n_meta_subsamples=None
):
    """A.shape = (m, n, d), m is number of datasets, n is the size, d is the dimension"""
    assert len(A.shape) == 3
    m, n, d = A.shape
    if n_base_subsamples is None:
        vec_A = A.reshape(-1, d)
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    else:
        vec_A = A.reshape(-1, d)
        subsample_indices = np.random.permutation(vec_A.shape[0])[
            :median_heuristic_n_subsamples
        ]
        vec_A = vec_A[subsample_indices]
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    base_s2 = median_heuristic(pairwise_square_dists)
    M2 = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            K_xx = gaussian_kernel_matrix(A[i], s2=base_s2)
            # K_xx = K_xx + np.eye(n)*eps
            K_yy = gaussian_kernel_matrix(A[j], s2=base_s2)
            # K_yy = K_yy + np.eye(n)*eps
            K_xy = gaussian_kernel_matrix(A[i], A[j], s2=base_s2)
            M2[i, j] = mmd2(K_xx, K_yy, K_xy)
    # Only have lower diagonal entries and diag=0
    # this way we avoid computing m(m-1)/2 entries
    M2 = M2 + M2.T
    meta_s2 = median_heuristic(M2, n_meta_subsamples)

    return base_s2, meta_s2


def form_datasets_from_tasks(tasks):
    datasets = []
    for task in tasks:
        X_tr, y_tr = task["train"]
        X_te, y_te = task["test"]
        X = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)
        D = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        datasets.append(D)
    # adds new axis
    datasets = np.stack(datasets, axis=0)
    return datasets


def cross_validate(model, alphas, lrs):
    opt_loss = np.inf
    for alpha in alphas:
        for lr in lrs:
            model.ridge_alpha = alpha
            model.learning_rate = lr
            model.calculate_ridge_regression_prototype_weights()
            model.calculate_transfer_risk()
            current_loss = np.nanmean(model.loss_matrix)
            logging.info("Cross validating (alpha, lr): {}".format((alpha, lr)))
            logging.info("Current loss: {}".format(current_loss))
            # Keep best hyperparams
            if current_loss < opt_loss:
                logging.info(
                    "Best (alpha, lr) so far {}, with loss {:.4f}".format(
                        (alpha, lr), current_loss
                    )
                )
                opt_loss = current_loss
                opt_alpha = alpha
                opt_lr = lr
    return opt_alpha, opt_lr, opt_loss


class MetaKNNGDExperiment:
    """Full experiment optimised for speed

    This does not adhere to the sklearn like
    fit/transform/predict framework."""

    def __init__(
        self,
        train_tasks,
        test_tasks,
        M_tr_te,
        k_nn,
        adaptation_steps,
        ridge_alpha,
        learning_rate,
        train_task_reordering,
    ):
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        # MMD matrix (T_tr, T_te) where T_te and T_te
        # is the number of train and test tasks respectively
        self.M_tr_te_original = M_tr_te
        self.T_tr, self.T_te = len(train_tasks), len(test_tasks)
        assert self.M_tr_te_original.shape == (self.T_tr, self.T_te)
        self.k_nn = k_nn
        self.adaptation_steps = adaptation_steps
        self.ridge_alpha = ridge_alpha
        self.learning_rate = learning_rate
        self.train_task_reordering = train_task_reordering
        self._reorder()
        self._form_datasets_from_tasks()
        assert self.M_tr_te.shape == (self.T_tr, self.T_te)

    def _reorder(self):
        self.train_tasks = [self.train_tasks[i] for i in self.train_task_reordering]
        self.M_tr_te = self.M_tr_te_original[
            np.ix_(self.train_task_reordering, np.arange(self.T_te))
        ]

    def _form_datasets_from_tasks(self):
        self.train_datasets = form_datasets_from_tasks(self.train_tasks)
        self.test_datasets = form_datasets_from_tasks(self.test_tasks)
        self.datasets = np.concatenate(
            [self.train_datasets, self.test_datasets], axis=0
        )

    def _get_ridge_regression_prototype_weights(self):
        """Get the ridge regression estimates of dataset prototypes"""

        def get_weights(dataset):
            X, y = dataset[:, :-1], dataset[:, -1:]
            w_hat = Ridge(alpha=self.ridge_alpha, fit_intercept=False).fit(X, y).coef_
            return w_hat

        self.prototype_ridge_weights = []
        for dataset in self.train_datasets:
            self.prototype_ridge_weights.append(get_weights(dataset))
        self.prototype_ridge_weights = np.array(self.prototype_ridge_weights).squeeze()
        assert len(self.prototype_ridge_weights) == self.T_tr

    def calculate_ridge_regression_prototype_weights(self):
        self._get_ridge_regression_prototype_weights()

    def _adapt(self, i, t):
        """Adapt to one task with index i when meta-train set is of size t"""
        # Unpack task
        test_task = self.test_tasks[i]
        X_tr, y_tr = test_task["train"]
        # Find K nearest tasks using M_tr_te
        distances = self.M_tr_te[:t, i]

        ridge_weights_of_knn = self.prototype_ridge_weights[
            np.argsort(distances)[: self.k_nn]
        ]
        # Find w_hat running GD
        w_0 = np.mean(ridge_weights_of_knn, axis=0)
        w_hat = w_0
        for _ in range(self.adaptation_steps):
            w_hat -= self.learning_rate * 2 * X_tr.T @ (X_tr @ w_hat - y_tr)
        return w_hat

    def _loss(self, i, t):
        test_task = self.test_tasks[i]
        X_te, y_te = test_task["test"]
        w_hat = self._adapt(i, t)
        y_hat_te = X_te @ w_hat
        return mean_squared_error(y_te, y_hat_te)

    def calculate_transfer_risk(self):
        """Calculate the transfer risk"""
        # The loss matrix is a matrix of size (T_tr, T_te)
        # Note that the first self.k_nn columns are nans selfince
        # we do not fill them in as not eno ugh prototypes are available
        self.loss_matrix = np.zeros((self.T_tr, self.T_te))
        self.loss_matrix[: self.k_nn, :] = np.nan
        # Each t represents using meta-train instances up until t according to
        # ordering as prototypes. We need to start at k_nn to be able to find
        # k_nn neighbours
        for t in range(self.k_nn, self.T_tr):
            for i in range(self.T_te):
                self.loss_matrix[t, i] = self._loss(i, t)


class GDLeastSquares:
    def __init__(self, learning_rate, adaptation_steps):
        self.learning_rate = learning_rate
        self.adaptation_steps = adaptation_steps

    def fit(self, X_tr, y_tr):
        n, d = X_tr.shape
        self.w_hat = np.zeros(d)
        for _ in range(self.adaptation_steps):
            self.w_hat -= self.learning_rate * 2 * X_tr.T @ (X_tr @ self.w_hat - y_tr)
        return self.w_hat

    def predict(self, X_te):
        return X_te @ self.w_hat


class IndependentTaskLearning:
    def __init__(self, tasks, algorithm, loss=mean_squared_error):
        """
        :param tasks: tasks that ITL will be performed over by algorithm
        :type tasks: list of tasks, where each task is a dict with keys ("train", "test")
            and values (X_tr, y_tr), tuple of numpy arrays
        :param algorithm: algorithm implementing sklearn fit / predict framework
        :type algorithm: instance of predictor class with fit / predict defined
        :param loss: loss function taking loss(y, y_pred)
        :type loss: loss(y: np.ndarray, y_pred: np.ndarray) -> float
        """
        self.tasks = tasks
        self.algorithm = algorithm
        self.loss = loss

    def _fit(self, task):
        """Fit `algorithm` to the i'th task"""
        X_tr, y_tr = task["train"]
        return self.algorithm.fit(X_tr, y_tr)

    def _predict(self, task):
        X_te, y_te = task["test"]
        return self.algorithm.predict(X_te)

    def _loss(self, task):
        _, y_te = task["test"]
        self._fit(task)
        y_pred = self._predict(task)
        return self.loss(y_pred, y_te)

    def calculate_transfer_risk(self):
        # Collect loss for each task in tasks
        self.losses = []
        for task in self.tasks:
            self.losses.append(self._loss(task))
        self.losses = np.array(self.losses)


def cross_validate_itl(model, lrs):
    opt_loss = np.inf
    for lr in lrs:
        model.algorithm.learning_rate = lr
        model.calculate_transfer_risk()
        current_loss = np.mean(model.losses)
        logging.info("Cross validating (lr): {}".format(lr))
        logging.info("Current loss: {}".format(current_loss))
        # Keep best hyperparams
        if current_loss < opt_loss:
            logging.info(
                "Best (lr) so far {}, with loss {:.4f}".format(lr, current_loss)
            )
            opt_loss = current_loss
            opt_lr = lr
    return opt_lr, opt_loss


def cross_validate(model, alphas, lrs):
    opt_loss = np.inf
    for alpha in alphas:
        for lr in lrs:
            model.ridge_alpha = alpha
            model.learning_rate = lr
            model.calculate_ridge_regression_prototype_weights()
            model.calculate_transfer_risk()
            current_loss = np.nanmean(model.loss_matrix)
            logging.info("Cross validating (alpha, lr): {}".format((alpha, lr)))
            logging.info("Current loss: {}".format(current_loss))
            # Keep best hyperparams
            if current_loss < opt_loss:
                logging.info(
                    "Best (alpha, lr) so far {}, with loss {:.4f}".format(
                        (alpha, lr), current_loss
                    )
                )
                opt_loss = current_loss
                opt_alpha = alpha
                opt_lr = lr
    return opt_alpha, opt_lr, opt_loss


def calculate_double_gaussian_median_heuristics(
    A, n_base_subsamples=None, n_meta_subsamples=None
):
    """A.shape = (m, n, d), m is number of datasets, n is the size, d is the dimension"""
    assert len(A.shape) == 3
    m, n, d = A.shape
    if n_base_subsamples is None:
        vec_A = A.reshape(-1, d)
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    else:
        vec_A = A.reshape(-1, d)
        subsample_indices = np.random.permutation(vec_A.shape[0])[
            :median_heuristic_n_subsamples
        ]
        vec_A = vec_A[subsample_indices]
        pairwise_square_dists = squareform(pdist(vec_A, "sqeuclidean"))
    base_s2 = median_heuristic(pairwise_square_dists)
    M2 = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            K_xx = gaussian_kernel_matrix(A[i], s2=base_s2)
            # K_xx = K_xx + np.eye(n)*eps
            K_yy = gaussian_kernel_matrix(A[j], s2=base_s2)
            # K_yy = K_yy + np.eye(n)*eps
            K_xy = gaussian_kernel_matrix(A[i], A[j], s2=base_s2)
            M2[i, j] = mmd2(K_xx, K_yy, K_xy)
    # Only have lower diagonal entries and diag=0
    # this way we avoid computing m(m-1)/2 entries
    M2 = M2 + M2.T
    meta_s2 = median_heuristic(M2, n_meta_subsamples)

    return base_s2, meta_s2


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
    seed = param_dict["seed"]
    set_random_seeds(seed)
    logging.info("random seeds set to {}".format(seed))
    k_nn = param_dict["k_nn"]
    env_name = param_dict["env_name"]

    k_shot = 10
    k_query = 15
    median_heuristic_n_subsamples = 300
    meta_train_batches = 400
    meta_train_batch_size = 1  # Hardcoded
    meta_val_batches = 500
    meta_val_batch_size = 1  # Hardcoded
    meta_test_batches = 500
    meta_test_batch_size = 1  # Hardcoded
    env = hkl.load(SETTINGS_DATA_DIR / env_name)
    param_dict["env_attributes"] = vars(env)

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

    # Calculate base_s2 and meta_s2 from train set
    train_datasets = form_datasets_from_tasks(train_batches)
    val_datasets = form_datasets_from_tasks(val_batches)
    test_datasets = form_datasets_from_tasks(test_batches)
    base_s2_D, meta_s2_D = calculate_double_gaussian_median_heuristics(
        train_datasets, n_base_subsamples=median_heuristic_n_subsamples
    )

    M_tr_val = np.sqrt(_mmd2_matrix(train_datasets, val_datasets, base_s2_D))
    M_tr_te = np.sqrt(_mmd2_matrix(train_datasets, test_datasets, base_s2_D))

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
    K_D = _gaussian_kernel_mmd2_matrix(
        train_datasets, train_datasets, base_s2_D, meta_s2_D
    )
    kh_D = KernelHerding(K_D)
    kh_D.run()
    dataset_indices["kh_data"] = kh_D.sampled_order
    logging.info("Done")

    # Will cross validate anyway
    # Better to fail if it doesn't work
    ridge_alpha = None
    learning_rate = None
    logging.info("Defining models")
    model_u = MetaKNNGDExperiment(
        train_batches,
        val_batches,
        M_tr_val,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=ridge_alpha,
        learning_rate=learning_rate,
        train_task_reordering=dataset_indices["uniform"],
    )
    model_kh_w = MetaKNNGDExperiment(
        train_batches,
        val_batches,
        M_tr_val,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=ridge_alpha,
        learning_rate=learning_rate,
        train_task_reordering=dataset_indices["kh_weights"],
    )
    model_kh_D = MetaKNNGDExperiment(
        train_batches,
        val_batches,
        M_tr_val,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=ridge_alpha,
        learning_rate=learning_rate,
        train_task_reordering=dataset_indices["kh_data"],
    )
    logging.info("Done")
    logging.info("Cross validation against learning rate")
    # Hyperparams
    lrs = np.geomspace(1e-4, 1e0, 3)
    alphas = np.geomspace(1e-4, 1e4, 5)
    logging.info("Uniform")
    alpha_u, lr_u, u_cross_val_loss = cross_validate(model_u, alphas, lrs)
    logging.info("KH weights")
    alpha_kh_w, lr_kh_w, kh_w_cross_val_loss = cross_validate(model_kh_w, alphas, lrs)
    logging.info("KH data")
    alpha_kh_D, lr_kh_D, kh_D_cross_val_loss = cross_validate(model_kh_D, alphas, lrs)
    logging.info("Optimal learning rates and losses found:")
    logging.info(
        "U: alpha={}, lr={}, loss={:.4f}".format(alpha_u, lr_u, u_cross_val_loss)
    )
    logging.info(
        "KH_W: alpha={}, lr={}, loss={:.4f}".format(
            alpha_kh_w, lr_kh_w, kh_w_cross_val_loss
        )
    )
    logging.info(
        "KH_D: alpha={}, lr={}, loss={:.4f}".format(
            alpha_kh_D, lr_kh_D, kh_D_cross_val_loss
        )
    )
    # Optimising for ITL
    logging.info("Cross validation for ITL")
    one_step_gd = GDLeastSquares(learning_rate=None, adaptation_steps=1)
    itl = IndependentTaskLearning(val_batches, one_step_gd)
    lr_itl, itl_cross_val_loss = cross_validate_itl(itl, lrs)
    logging.info("Optimal learning rate and loss found:")
    logging.info("lr={}, loss={:.4f}".format(lr_itl, itl_cross_val_loss))
    itl.algorithm.learning_rate = lr_itl

    # Set optimal hyperparameters
    experiment_data = dict()
    experiment_data["uniform"] = {
        "optimal_parameters": {"ridge_alpha": alpha_u, "learning_rate": lr_u}
    }
    experiment_data["kh_weights"] = {
        "optimal_parameters": {"ridge_alpha": alpha_kh_w, "learning_rate": lr_kh_w}
    }
    experiment_data["kh_data"] = {
        "optimal_parameters": {"ridge_alpha": alpha_kh_D, "learning_rate": lr_kh_D}
    }
    experiment_data["itl"] = {
        "optimal_parameters": {"ridge_alpha": None, "learning_rate": lr_itl}
    }

    logging.info("Getting learning curves for meta test error")
    meta_test_error = dict()
    model_u = MetaKNNGDExperiment(
        train_batches,
        test_batches,
        M_tr_te,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=alpha_u,
        learning_rate=lr_u,
        train_task_reordering=dataset_indices["uniform"],
    )
    model_kh_w = MetaKNNGDExperiment(
        train_batches,
        test_batches,
        M_tr_te,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=alpha_kh_w,
        learning_rate=lr_kh_w,
        train_task_reordering=dataset_indices["kh_weights"],
    )
    model_kh_D = MetaKNNGDExperiment(
        train_batches,
        test_batches,
        M_tr_te,
        k_nn,
        adaptation_steps=1,
        ridge_alpha=alpha_kh_D,
        learning_rate=lr_kh_D,
        train_task_reordering=dataset_indices["kh_data"],
    )

    logging.info("Uniform")
    model_u.calculate_ridge_regression_prototype_weights()
    model_u.calculate_transfer_risk()
    meta_test_error["uniform"] = model_u.loss_matrix
    logging.info("KH weights")
    model_kh_w.calculate_ridge_regression_prototype_weights()
    model_kh_w.calculate_transfer_risk()
    meta_test_error["kh_weights"] = model_kh_w.loss_matrix
    logging.info("KH data")
    model_kh_D.calculate_ridge_regression_prototype_weights()
    model_kh_D.calculate_transfer_risk()
    meta_test_error["kh_data"] = model_kh_D.loss_matrix
    logging.info("ITL")
    itl.tasks = test_batches
    itl.calculate_transfer_risk()
    meta_test_error["itl"] = itl.losses

    logging.info("Done")

    experiment_data["uniform"]["test_error"] = meta_test_error["uniform"]
    experiment_data["kh_weights"]["test_error"] = meta_test_error["kh_weights"]
    experiment_data["kh_data"]["test_error"] = meta_test_error["kh_data"]
    experiment_data["itl"]["test_error"] = meta_test_error["itl"]

    # Dump data
    # Params
    hkl.dump(param_dict, args.output_dir / "parameters.hkl")
    # Experiment data
    hkl.dump(experiment_data, args.output_dir / "experiment_data.hkl")

    logging.info("Done")
    logging.info("Good bye!")
