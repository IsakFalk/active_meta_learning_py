import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from active_meta_learning.data import (EnvironmentDataSet,
                                       HypercubeWithKVertexGaussian)
from active_meta_learning.data_utils import *
from active_meta_learning.kernels import *
from active_meta_learning.optimisation import *
from active_meta_learning.utils import *
from hpc_cluster.utils import extract_csv_to_dict


def fit_env_pca(env):
    X = env.sample(n=3000)
    pca = PCA(n_components=2, whiten=True)
    pca.fit(X)
    return pca


def plot_2d_dist(env, fitted_pca):
    X = env.sample(3000)
    X = fitted_pca.transform(X)
    fig, ax = plt.subplots(1)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.2)
    ax.set_xlabel("v_1")
    ax.set_ylabel("v_2")
    ax.set_title("2d-pca projected task_w sampled from environment")
    return fig, ax


def plot_mmd_kh_vs_uniform(J_kh, J_uniform):
    fig, ax = plt.subplots()
    t = np.arange(J_kh.shape[0])
    ax.plot(t, J_kh, label="Kernel Herding", color="red")
    ax.plot(t, J_uniform, label="Uniform", color="blue")
    ax.legend()
    ax.set_title("MMD between P and Q_t")
    ax.set_xlabel("t")
    ax.set_ylabel("MMD(P, Q_t)")
    return fig, ax


def plot_mmd_kh_D_vs_KH_w_vs_uniform(J_kh_D, J_kh_w, J_uniform_w):
    fig, ax = plt.subplots(figsize=(8, 6))
    t = np.arange(J_kh_D.shape[0])
    ax.plot(t, J_kh_D, label="Kernel Herding (Data space)", color="red")
    ax.plot(t, J_kh_w, label="Kernel Herding (Weight space)", color="orange")
    ax.plot(t, J_uniform_w, label="Uniform", color="blue")
    ax.legend()
    ax.set_title("MMD (kernel on w) between P and Q_t")
    ax.set_xlabel("t")
    ax.set_ylabel("MMD(P, Q_t)")
    return fig, ax


def plot_first_n_task_ws(
    task_ws, kh_order_w, kh_order_D, uniform_order, fitted_pca, n=10
):
    fig, ax = plt.subplots(1, 3, figsize=(8 * 3, 8))
    task_ws = fitted_pca.transform(task_ws)

    task_ws_new = task_ws[kh_order_w]
    ax[0].scatter(
        task_ws[:, 0], task_ws[:, 1], color="black", alpha=0.2, label="all task ws"
    )
    ax[0].scatter(
        task_ws_new[:n, 0],
        task_ws_new[:n, 1],
        color="orange",
        marker="o",
        label="KH (weight) order",
    )
    n_sampled_mean = task_ws_new[:n].mean(axis=0)
    mean_x = n_sampled_mean[0]
    mean_y = n_sampled_mean[1]
    ax[0].scatter(
        mean_x,
        mean_y,
        color="black",
        marker="x",
        s=100,
        label="mean (first n kh (weight) samples)",
    )
    ax[0].legend()
    ax[0].set_title("Sampled task_ws KH (weight)")

    task_ws_new = task_ws[kh_order_D]
    ax[1].scatter(
        task_ws[:, 0], task_ws[:, 1], color="black", alpha=0.2, label="all task ws"
    )
    ax[1].scatter(
        task_ws_new[:n, 0],
        task_ws_new[:n, 1],
        color="red",
        marker="o",
        label="KH (data) order",
    )
    n_sampled_mean = task_ws_new[:n].mean(axis=0)
    mean_x = n_sampled_mean[0]
    mean_y = n_sampled_mean[1]
    ax[1].scatter(
        mean_x,
        mean_y,
        color="black",
        marker="x",
        s=100,
        label="mean (first n kh (data) samples)",
    )
    ax[1].legend()
    ax[1].set_title("Sampled task_ws KH (data)")

    task_ws_new = task_ws[uniform_order]
    ax[2].scatter(
        task_ws[:, 0], task_ws[:, 1], color="black", alpha=0.2, label="all task ws"
    )
    ax[2].scatter(
        task_ws_new[:n, 0],
        task_ws_new[:n, 1],
        color="blue",
        marker="o",
        label="random order",
    )
    n_sampled_mean = task_ws_new[:n].mean(axis=0)
    mean_x = n_sampled_mean[0]
    mean_y = n_sampled_mean[1]
    ax[2].scatter(
        mean_x,
        mean_y,
        color="black",
        marker="x",
        s=100,
        label="mean (first n random samples)",
    )
    ax[2].legend()
    ax[2].set_title("Sampled task_ws random")

    fig.suptitle(
        "Task ws (PCA-projected to 2d), KH (weight and data) vs uniform sampling order of tasks"
    )
    return fig, ax


def run_experiment(
    env, plot_dir, k_shot=10, k_query=15, noise_w=0.0, noise_y=0.0, N=400
):
    # fit pca and save projected scatter plot
    fitted_pca = fit_env_pca(env)
    fig, ax = plot_2d_dist(env, fitted_pca)
    fig.savefig(
        plot_dir
        / "task_w_pdf_on_sphere_scatter_plot-d={}_k={}_s2={}".format(
            env.d, env.k, env.s2
        ),
        format="png",
    )

    # Create dataset and dataloader
    env_ = EnvironmentDataSet(k_shot, k_query, env, noise_w=noise_w, noise_y=noise_y)
    dataloader = DataLoader(
        env_,
        batch_size=1,  # torch IterableDataset reduces to batch_size=1 for any batch_size we pick
        num_workers=0,
        collate_fn=env_.collate_fn,
    )

    # Sample metainstances
    sampled_batches = aggregate_sampled_task_batches(dataloader, N)
    task_ws = get_task_parameters(sampled_batches)
    sampled_batches = convert_batches_to_fw_form(sampled_batches)

    # Run kernel herding directly on weight space
    s2_w = median_heuristic(squareform(pdist(task_ws, "sqeuclidean")))
    K_w = gaussian_kernel_matrix(task_ws, s2_w)
    kh_w = KernelHerding(K_w)
    kh_w.run()
    kh_w_order = kh_w.sampled_order

    # Run kernel herding on data space
    K_D = gaussian_kernel_mmd2_matrix(sampled_batches)
    kh_D = KernelHerding(K_D)
    kh_D.run()
    kh_D_order = kh_D.sampled_order
    uniform_order = np.random.permutation(np.arange(N))

    # Get learning curves in data space
    J_kh_D = mmd2_curve(K_D, kh_D_order) ** 0.5
    J_uniform_D = mmd2_curve(K_D, uniform_order) ** 0.5
    fig, ax = plot_mmd_kh_vs_uniform(J_kh_D, J_uniform_D)
    fig.savefig(
        plot_dir
        / "mmd_kh_vs_uniform_in_D_space-d={}_k={}_s2={}".format(env.d, env.k, env.s2),
        format="png",
    )

    # Look at the chosen instances for each ordering
    fig, ax = plot_first_n_task_ws(
        task_ws, kh_w_order, kh_D_order, uniform_order, fitted_pca, n=25
    )
    fig.savefig(
        plot_dir
        / "n_first_task_ws_chosen_kh_vs_uniform-d={}_k={}_s2={}".format(
            env.d, env.k, env.s2
        ),
        format="png",
    )

    # Get learning curves in weight space
    J_kh_D = mmd2_curve(K_w, kh_D_order) ** 0.5
    J_kh_w = mmd2_curve(K_w, kh_w_order) ** 0.5
    J_uniform_w = mmd2_curve(K_w, uniform_order) ** 0.5
    fig, ax = plot_mmd_kh_D_vs_KH_w_vs_uniform(J_kh_D, J_kh_w, J_uniform_w)
    fig.savefig(
        plot_dir
        / "mmd_kh_vs_uniform_in_w_space-d={}_k={}_s2={}".format(env.d, env.k, env.s2),
        format="png",
    )

    # Clean up left over figure objects
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster paramters (see hpc_cluster library docs)"
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

    # read in line of parameters
    param_dict = extract_csv_to_dict(args.csv_path, args.extract_line)
    print(param_dict)
    env = HypercubeWithKVertexGaussian(**param_dict)
    run_experiment(env, plot_dir=args.output_dir, N=500)
    # dump params
    with open(args.output_dir / "parameters.pkl", "wb") as f:
        pickle.dump(param_dict, f)
