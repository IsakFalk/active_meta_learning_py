import argparse
import logging
from pathlib import Path


import hickle as hkl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from active_meta_learning.data import EnvironmentDataSet
from active_meta_learning.data_utils import (
    aggregate_sampled_task_batches,
    coalesce_train_and_test_in_dicts,
    convert_batches_to_fw_form,
    convert_batches_to_np,
    get_task_parameters,
    remove_batched_dimension_in_D,
    reorder_list,
    set_random_seeds,
    form_datasets_from_tasks,
    npfy_batches,
)
from active_meta_learning.kernels import (
    gaussian_kernel_matrix,
    gaussian_kernel_mmd2_matrix,
    median_heuristic,
    mmd2,
)
from active_meta_learning.optimisation import KernelHerding
from active_meta_learning.estimators import (
    RidgeRegression,
    BiasedRidgeRegression,
    RidgeRegPrototypeEstimator,
    TrueWeightPrototypeEstimator,
    GDLeastSquares,
)
from active_meta_learning.project_parameters import SCRIPTS_DIR, SETTINGS_DATA_DIR
from hpc_cluster.utils import extract_csv_to_dict

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
import warnings

warnings.filterwarnings("ignore")


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


class MetaKNNExperiment:
    """Full experiment optimised for speed

    This does not adhere to the sklearn like
    fit/transform/predict framework."""

    def __init__(
        self,
        train_tasks,
        test_tasks,
        dist_tr_te,
        base_algorithm,
        k_nn,
        train_task_reordering,
        prototype_estimator,  # transformer prototype
    ):
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.dist_tr_te = dist_tr_te
        self.base_algorithm = base_algorithm
        self.k_nn = k_nn
        self.train_task_reordering = train_task_reordering
        self.prototype_estimator = prototype_estimator

        self.T_tr, self.T_te = len(train_tasks), len(test_tasks)
        assert self.dist_tr_te.shape == (self.T_tr, self.T_te)
        self._reorder()
        self._form_datasets_from_tasks()
        self.calculate_prototypes()

    def _reorder(self):
        self.train_tasks = [self.train_tasks[i] for i in self.train_task_reordering]
        self.dist_tr_te = self.dist_tr_te[
            np.ix_(self.train_task_reordering, np.arange(self.T_te))
        ]

    def _form_datasets_from_tasks(self):
        self.train_datasets = form_datasets_from_tasks(self.train_tasks)
        self.test_datasets = form_datasets_from_tasks(self.test_tasks)
        self.datasets = np.concatenate(
            [self.train_datasets, self.test_datasets], axis=0
        )

    def calculate_prototypes(self):
        """Recalculate prototypes using prototype_estimator

        This allows us to cross-validate after we change the prototype
        estimator parameters"""
        self.prototypes = self.prototype_estimator.transform(self.train_tasks)

    def _adapt(self, i, t):
        """Adapt to one task with index i when meta-train set is of size t"""
        test_task = self.test_tasks[i]
        X_tr, y_tr = test_task["train"]
        distances = self.dist_tr_te[:t, i]

        knn_prototypes = self.prototypes[np.argsort(distances)[: self.k_nn], :]

        w_0 = np.mean(knn_prototypes, axis=0)
        self.base_algorithm.fit(X_tr, y_tr, w_0=w_0)

    def _loss(self, i, t):
        test_task = self.test_tasks[i]
        X_te, y_te = test_task["test"]
        self._adapt(i, t)
        return mean_squared_error(y_te, self.base_algorithm.predict(X_te))

    def calculate_transfer_risk(self):
        """Calculate the transfer risk"""
        # The loss matrix is a matrix of size (T_tr, T_te)
        # Note that the first self.k_nn columns are nans selfince
        # we do not fill them in as not eno ugh prototypes are available
        self.loss_matrix_ = np.zeros((self.T_tr, self.T_te))
        self.loss_matrix_[: self.k_nn, :] = np.nan
        # Each t represents using meta-train instances up until t according to
        # ordering as prototypes. We need to start at k_nn to be able to find
        # k_nn neighbours
        for t in range(self.k_nn, self.T_tr):
            for i in range(self.T_te):
                self.loss_matrix_[t, i] = self._loss(i, t)

    def set_base_algorithm_params(self, params):
        self.base_algorithm.set_params(**params)

    def get_base_algorithm_params(self):
        return self.base_algorithm.get_params()

    def set_prototype_estimator_params(self, params):
        self.prototype_estimator.set_params(**params)

    def get_prototype_estimator_params(self):
        return self.prototype_estimator.get_params()

    def set_params(self, params):
        # Need to catch estimator having no params
        self.set_base_algorithm_params(params["base_algorithm"])
        self.set_prototype_estimator_params(params["prototype_estimator"])

    def get_params(self):
        return {
            "base_algorithm": self.get_base_algorithm_params(),
            "prototype_estimator": self.get_prototype_estimator_params(),
        }


def cross_validate_aml(aml, cv_base_algorithm_params, cv_prototype_estimator_params):
    """Cross validate aml over cv_params

    aml is an object of class MetaKNNExperiment and cv_base_algorithm_params and
    cv_prototype_estimator_params are dictionaries of the values for each
    parameter (base_algorithm and prototype_estimator respectively).

    Note that the params need to be non-empty. If the base_algorithm or
    prototype_estimator does not take any parameters, pass None as key"""
    base_algorithm_param_grid = ParameterGrid(cv_base_algorithm_params)
    prototype_estimator_param_grid = ParameterGrid(cv_prototype_estimator_params)
    opt_loss = np.inf
    opt_params = {"base_algorithm": None, "prototype_estimator": None}
    for base_params in base_algorithm_param_grid:
        for prototype_params in prototype_estimator_param_grid:
            logging.info(
                "Cross validating base / prototype params: {} / {}".format(
                    base_params, prototype_params
                )
            )
            aml.set_params(
                {"base_algorithm": base_params, "prototype_estimator": prototype_params}
            )
            aml.calculate_prototypes()
            aml.calculate_transfer_risk()
            current_loss = np.nanmean(aml.loss_matrix_)
            logging.info("Current loss: {}".format(current_loss))
            if current_loss < opt_loss:
                opt_loss = current_loss
                opt_params["base_algorithm"] = base_params
                opt_params["prototype_estimator"] = prototype_params
                logging.info(
                    "Best params so far {} / {}, with loss {:.4f}".format(
                        base_params, prototype_params, opt_loss
                    )
                )
    return opt_params, opt_loss


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

    def fit(self, task):
        """Fit `algorithm` to the i'th task"""
        X_tr, y_tr = task["train"]
        self.algorithm.fit(X_tr, y_tr)

    def predict(self, task):
        X_te, _ = task["test"]
        return self.algorithm.predict(X_te)

    def get_loss(self, task):
        _, y_te = task["test"]
        self.fit(task)
        y_pred = self.predict(task)
        return self.loss(y_pred, y_te)

    def calculate_transfer_risk(self):
        # Collect loss for each task in tasks
        self.losses_ = []
        for task in self.tasks:
            self.losses_.append(self.get_loss(task))
        self.losses_ = np.array(self.losses_)

    def set_params(self, params):
        """Update paramaters of algorithm"""
        self.algorithm.set_params(**params)

    def get_params(self):
        """Update paramaters of algorithm"""
        return self.algorithm.get_params()


def cross_validate_itl(itl, cv_params):
    """Cross validate itl over cv_params

    itl is an object of class IndependentTaskLearning
    and cv_params is a dictionary of the values for each
    parameter."""
    param_grid = ParameterGrid(cv_params)
    opt_loss = np.inf
    opt_params = None
    for params in param_grid:
        logging.info("Cross validating params: {}".format(params))
        itl.set_params(params)
        itl.calculate_transfer_risk()
        current_loss = np.mean(itl.losses_)
        logging.info("Current loss: {}".format(current_loss))
        if current_loss < opt_loss:
            opt_loss = current_loss
            opt_params = params
            logging.info(
                "Best params so far {}, with loss {:.4f}".format(params, opt_loss)
            )
    return opt_params, opt_loss


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
        subsample_indices = np.random.permutation(vec_A.shape[0])[:n_base_subsamples]
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


def run_aml(
    aml_order,
    train_batches,
    val_batches,
    test_batches,
    dist_tr_te,
    dist_tr_val,
    learning_rates,
    alphas,
):
    """Run AML experiment"""
    data = {"ridge": {}, "weights": {}}
    one_step_gd = GDLeastSquares(learning_rate=None, adaptation_steps=1)

    #
    # Prototypes: ridge regression
    #
    model = MetaKNNExperiment(
        train_batches,
        val_batches,
        dist_tr_val,
        one_step_gd,
        k_nn,
        aml_order,
        RidgeRegPrototypeEstimator(alpha=0.1),
    )
    logging.info("Cross validating AML: RidgeReg")
    opt_params, opt_loss = cross_validate_aml(
        model, {"learning_rate": learning_rates}, {"alpha": alphas}
    )
    logging.info("Optimal parameters: {}, loss {}".format(opt_params, opt_loss))
    # Reset model with optimal parameters
    # and tr-te
    model = MetaKNNExperiment(
        train_batches,
        test_batches,
        dist_tr_te,
        one_step_gd,
        k_nn,
        aml_order,
        RidgeRegPrototypeEstimator(alpha=0.1),
    )
    model.set_params(opt_params)
    model.calculate_prototypes()
    model.calculate_transfer_risk()
    logging.info("Mean test error: {}".format(np.nanmean(model.loss_matrix_)))
    data["ridge"] = {"optimal_parameters": opt_params, "test_error": model.loss_matrix_}

    #
    # Prototypes: true weights
    #
    model = MetaKNNExperiment(
        train_batches,
        val_batches,
        dist_tr_val,
        one_step_gd,
        k_nn,
        aml_order,
        TrueWeightPrototypeEstimator(),
    )
    logging.info("Cross validating AML: TrueWeights")
    opt_params, opt_loss = cross_validate_aml(
        model, {"learning_rate": learning_rates}, {}
    )
    logging.info("Optimal parameters: {}, loss {}".format(opt_params, opt_loss))
    # Reset model with optimal parameters
    # and tr-te
    model = MetaKNNExperiment(
        train_batches,
        test_batches,
        dist_tr_te,
        one_step_gd,
        k_nn,
        aml_order,
        TrueWeightPrototypeEstimator(),
    )
    model.set_params(opt_params)
    model.calculate_prototypes()
    model.calculate_transfer_risk()
    logging.info("Mean test error: {}".format(np.nanmean(model.loss_matrix_)))
    data["weights"] = {
        "optimal_parameters": opt_params,
        "test_error": model.loss_matrix_,
    }
    return data


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
    parser.add_argument(
        "--plot", action="store_true",
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

    # Experiment data
    # A dictionary that holds the data related to the
    # experiment. This is a dictionary which looks as follows
    # Level 1:
    # experiment_data.keys() == ("aml", "itl")
    # For "aml" and "itl"
    # Level 2:
    # experiment_data["aml"].keys() == ("data", "weights", "uniform")
    # experiment_data["itl"].keys() == ("ridge", "one_step_gd")
    # Level 3:
    # experiment_data["itl"][{"ridge", "one_step_gd"}].keys() == ("optimal_parameters", "test_error")
    # experiment_data["aml"][{"data", "weights", "uniform"}].keys() == ("ridge", "weights")
    # Level 4:
    # experiment_data["aml"][{"data", "weights", "uniform"}][{"ridge", "weights"}].keys() == ("optimal_parameters", "test_error")
    experiment_data = {"aml": {}, "itl": {}}
    # extra_info
    # Extra dictionary containing the kh algorithm object
    # including the sampled order and so on.
    # Also contain the full datasets for train, val and test
    extra_info = {"datasets": {}, "kh_objects": {}}

    env = hkl.load(SETTINGS_DATA_DIR / env_name)
    d = env.d
    param_dict["env_attributes"] = vars(env)
    logging.info("Environment: {}".format(env_name))
    k_shot = 25
    k_query = 20
    logging.info("d: {}, k_shot: {}, k_query: {}".format(d, k_shot, k_query))

    median_heuristic_n_subsamples = 300
    num_meta_train_batches = 400
    meta_train_batch_size = 1  # Hardcoded
    num_meta_val_batches = 500
    meta_val_batch_size = 1  # Hardcoded
    num_meta_test_batches = 500
    meta_test_batch_size = 1  # Hardcoded

    logging.info("Generating meta-train batches")
    # Sample meta-train
    train_dataset = EnvironmentDataSet(k_shot, k_query, env, noise_w=0.0, noise_y=0.0)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=meta_train_batch_size,
    )
    train_batches = aggregate_sampled_task_batches(
        train_dataloader, num_meta_train_batches
    )
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
    val_batches = aggregate_sampled_task_batches(val_dataloader, num_meta_val_batches)
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
    test_batches = aggregate_sampled_task_batches(
        test_dataloader, num_meta_test_batches
    )
    test_batches_kh = convert_batches_to_fw_form(test_batches)
    test_batches = npfy_batches(test_batches)
    test_task_ws = get_task_parameters(test_batches)
    logging.info("Done")

    extra_info["datasets"] = {
        "train": train_batches,
        "val": val_batches,
        "test": test_batches,
    }

    train_datasets = form_datasets_from_tasks(train_batches)
    # Don't cheat, can only use support/train set for meta-test and meta-val
    tr_val_datasets = form_datasets_from_tasks(val_batches, use_only_train=True)
    tr_test_datasets = form_datasets_from_tasks(test_batches, use_only_train=True)

    logging.info("Calculating sigmas for data MMD")
    # Calculate base_s2 and meta_s2 from train set
    base_s2_D, meta_s2_D = calculate_double_gaussian_median_heuristics(
        train_datasets, n_base_subsamples=median_heuristic_n_subsamples
    )
    logging.info("Done")

    logging.info("Building MMD matrices for tr, val, test")
    M_tr_val = np.sqrt(_mmd2_matrix(train_datasets, tr_val_datasets, base_s2_D))
    M_tr_te = np.sqrt(_mmd2_matrix(train_datasets, tr_test_datasets, base_s2_D))
    logging.info("Done")

    logging.info("Generating learning curves")

    # Base algorithm
    one_step_gd = GDLeastSquares(learning_rate=None, adaptation_steps=1)
    rr = RidgeRegression(alpha=None)
    # Parameter grids
    learning_rates = np.geomspace(1e-3, 1e0, 5)
    alphas = np.geomspace(1e-8, 1e3, 5)

    experiment_data["aml"] = {}
    ###
    ### Baseline: Uniform sampling
    ###
    logging.info("AML: uniform")
    experiment_data["aml"]["uniform"] = {}
    logging.info("Getting optimal parameter and test error")
    aml_order = np.arange(num_meta_train_batches)
    data = run_aml(
        aml_order,
        train_batches,
        val_batches,
        test_batches,
        M_tr_te,
        M_tr_val,
        learning_rates,
        alphas,
    )
    experiment_data["aml"]["uniform"] = data

    ###
    ### AML: KH on data
    ###
    logging.info("AML: KH data")
    experiment_data["aml"]["data"] = {}
    logging.info("Generating active train order")
    K_D = _gaussian_kernel_mmd2_matrix(
        train_datasets, train_datasets, base_s2_D, meta_s2_D
    )
    kh_D = KernelHerding(K_D)
    kh_D.run()
    extra_info["kh_objects"]["data"] = kh_D
    logging.info("Done")
    logging.info("Getting optimal parameter and test error")
    aml_order = kh_D.sampled_order
    data = run_aml(
        aml_order,
        train_batches,
        val_batches,
        test_batches,
        M_tr_te,
        M_tr_val,
        learning_rates,
        alphas,
    )
    experiment_data["aml"]["data"] = data

    ###
    ### AML: KH on weights
    ###
    logging.info("AML: KH weights")
    experiment_data["aml"]["weights"] = {}
    logging.info("Generating active train order")
    s2_w = median_heuristic(squareform(pdist(train_task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(train_task_ws, s2_w)
    kh_w = KernelHerding(K_w)
    kh_w.run()
    extra_info["kh_objects"]["weights"] = kh_w
    logging.info("Done")
    logging.info("Getting optimal parameter and test error")
    aml_order = kh_w.sampled_order
    data = run_aml(
        aml_order,
        train_batches,
        val_batches,
        test_batches,
        M_tr_te,
        M_tr_val,
        learning_rates,
        alphas,
    )
    experiment_data["aml"]["weights"] = data

    ###
    ### ITL
    ###

    logging.info("ITL")
    experiment_data["itl"] = {}
    #
    # Ridge regression
    #
    logging.info("Ridge Reg")
    logging.info("Cross Validating")
    rr = RidgeRegression(alpha=None)
    itl_rr = IndependentTaskLearning(val_batches, rr)
    itl_rr_opt_params, itl_rr_cross_val_loss = cross_validate_itl(
        itl_rr, {"alpha": alphas.tolist()}
    )
    logging.info("Optimal learning rate and loss found:")
    logging.info(
        "alpha={}, loss={:.4f}".format(
            itl_rr_opt_params["alpha"], itl_rr_cross_val_loss
        )
    )
    itl_rr.set_params(itl_rr_opt_params)

    logging.info("Calculating meta-test error")
    itl_rr.tasks = test_batches
    itl_rr.calculate_transfer_risk()

    experiment_data["itl"]["ridge"] = {
        "optimal_parameters": itl_rr_opt_params,
        "test_error": itl_rr.losses_,
    }
    logging.info("Done")

    #
    # One step gd
    #
    logging.info("One-step GD")
    itl_one_step_gd = IndependentTaskLearning(val_batches, one_step_gd)
    itl_one_step_gd_opt_params, itl_one_step_gd_cross_val_loss = cross_validate_itl(
        itl_one_step_gd,
        {"learning_rate": learning_rates.tolist(), "adaptation_steps": [1]},
    )
    logging.info("Optimal learning rate and loss found:")
    logging.info(
        "lr={}, loss={:.4f}".format(
            itl_one_step_gd_opt_params["learning_rate"], itl_one_step_gd_cross_val_loss
        )
    )
    itl_one_step_gd.set_params(itl_one_step_gd_opt_params)

    logging.info("Calculating meta-test error")
    itl_one_step_gd.tasks = test_batches
    itl_one_step_gd.calculate_transfer_risk()

    experiment_data["itl"]["one_step_gd"] = {
        "optimal_parameters": itl_one_step_gd_opt_params,
        "test_error": itl_one_step_gd.losses_,
    }
    logging.info("Done")

    # Dump data
    logging.info("Dumping parameters and experiment_data")
    # Params
    hkl.dump(param_dict, args.output_dir / "parameters.hkl")
    # Experiment data
    hkl.dump(experiment_data, args.output_dir / "experiment_data.hkl")
    # Extra info
    hkl.dump(extra_info, args.output_dir / "extra_info.hkl")

    if args.plot:
        import json

        def plot_aml_ci(ax, error, color, label, until_t):
            mean = error.mean(axis=1)
            std = np.std(error, axis=1)
            ax.plot(mean[:until_t], label=label, color=color)
            upper_ci = mean + std
            lower_ci = mean - std
            ax.fill_between(
                np.arange(until_t),
                lower_ci[:until_t],
                upper_ci[:until_t],
                color=color,
                alpha=0.2,
            )

        def plot_itl_ci(ax, error, color, label, until_t):
            mean = np.mean(error)
            std = np.std(error)
            ax.axhline(mean, label=label, color=color, linestyle="--")
            upper_ci = mean + std
            lower_ci = mean - std
            ax.fill_between(
                np.arange(until_t), lower_ci, upper_ci, color=color, alpha=0.2
            )

        def plot_mean_and_ci(experiment_data, until_t, plot_itl=True):
            itl_error = experiment_data["itl"]
            data_error = experiment_data["aml"]["data"]
            weights_error = experiment_data["aml"]["weights"]
            uniform_error = experiment_data["aml"]["uniform"]

            # cmap: {"kh_data": red, "kh_weights": yellow, "uniform": blue, "itl_ridge": black, "itl_one_step_gd":gray}
            fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)

            # prototype: ridge
            ax[0].set_title("prototype: ridge estimates")
            plot_aml_ci(
                ax[0],
                data_error["ridge"]["test_error"],
                color="red",
                label="ordering: kh on data",
                until_t=until_t,
            )
            plot_aml_ci(
                ax[0],
                weights_error["ridge"]["test_error"],
                color="orange",
                label="ordering: kh on task weights",
                until_t=until_t,
            )
            plot_aml_ci(
                ax[0],
                uniform_error["ridge"]["test_error"],
                color="blue",
                label="ordering: random",
                until_t=until_t,
            )
            if plot_itl is True:
                plot_itl_ci(
                    ax[0],
                    itl_error["ridge"]["test_error"],
                    color="black",
                    label="itl ridge",
                    until_t=until_t,
                )
                plot_itl_ci(
                    ax[0],
                    itl_error["one_step_gd"]["test_error"],
                    color="gray",
                    label="itl 1-step gd",
                    until_t=until_t,
                )
            elif plot_itl == "ridge":
                plot_itl_ci(
                    ax[0],
                    itl_error["ridge"]["test_error"],
                    color="black",
                    label="itl ridge",
                    until_t=until_t,
                )
            elif plot_itl == "gd":
                plot_itl_ci(
                    ax[0],
                    itl_error["one_step_gd"]["test_error"],
                    color="gray",
                    label="itl 1-step gd",
                    until_t=until_t,
                )

            ax[0].set_ylabel("mse")
            ax[0].legend()

            # prototype: true weights
            ax[1].set_title("prototype: true weights")
            plot_aml_ci(
                ax[1],
                weights_error["weights"]["test_error"],
                color="orange",
                label="ordering: kh on task weights",
                until_t=until_t,
            )
            plot_aml_ci(
                ax[1],
                uniform_error["weights"]["test_error"],
                color="blue",
                label="ordering: random",
                until_t=until_t,
            )
            if plot_itl is True:
                plot_itl_ci(
                    ax[1],
                    itl_error["ridge"]["test_error"],
                    color="black",
                    label="itl ridge",
                    until_t=until_t,
                )
                plot_itl_ci(
                    ax[1],
                    itl_error["one_step_gd"]["test_error"],
                    color="gray",
                    label="itl 1-step gd",
                    until_t=until_t,
                )
            elif plot_itl == "ridge":
                plot_itl_ci(
                    ax[1],
                    itl_error["ridge"]["test_error"],
                    color="black",
                    label="itl ridge",
                    until_t=until_t,
                )
            elif plot_itl == "gd":
                plot_itl_ci(
                    ax[1],
                    itl_error["one_step_gd"]["test_error"],
                    color="gray",
                    label="itl 1-step gd",
                    until_t=until_t,
                )
            ax[1].set_ylabel("mse")
            ax[1].set_xlabel("t")
            ax[1].legend()

            return fig, ax

        # Plot all of the learning curves
        # Calculate meta_test_error

        fig, ax = plot_mean_and_ci(
            experiment_data, until_t=num_meta_train_batches, plot_itl="ridge"
        )
        fig.savefig(args.output_dir / "learning_curves.png")

    logging.info("Good bye!")
