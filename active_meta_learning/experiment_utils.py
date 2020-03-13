"""Snippets used for various experiments, collected and refactored"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from active_meta_learning.data_utils import (convert_batches_to_fw_form,
                                             get_task_parameters)
from active_meta_learning.kernels import (gaussian_kernel_matrix,
                                          gaussian_kernel_mmd2_matrix,
                                          median_heuristic)
from active_meta_learning.utils import mmd2_curve

########################
# MMD curve experiment #
########################


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


def plot_mmd_kh_D_vs_uniform_D(J_kh, J_uniform):
    fig, ax = plt.subplots()
    t = np.arange(J_kh.shape[0])
    ax.plot(t, J_kh, label="Kernel Herding (Data space)", color="red")
    ax.plot(t, J_uniform, label="Uniform", color="blue")
    ax.legend()
    ax.set_title("MMD (kernel on D) between P and Q_t")
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

    def plot_task_ws(
        ax, new_order, color, scatter_label_points, scatter_label_mean, title
    ):
        task_ws_new = task_ws[new_order]
        ax.scatter(
            task_ws[:, 0], task_ws[:, 1], color="black", alpha=0.2, label="all task ws"
        )
        ax.scatter(
            task_ws_new[:n, 0],
            task_ws_new[:n, 1],
            color=color,
            marker="o",
            label=scatter_label_points,
        )
        n_sampled_mean = task_ws_new[:n].mean(axis=0)
        mean_x = n_sampled_mean[0]
        mean_y = n_sampled_mean[1]
        ax.scatter(
            mean_x, mean_y, color="black", marker="x", s=100, label=scatter_label_mean,
        )
        ax.legend()
        ax.set_title(title)

    # Scatter plots of sampled tasks according to order of
    # KH (weight), KH (data), Uniform
    plot_task_ws(
        ax[0],
        kh_order_w,
        "orange",
        "KH (weight) order",
        "mean (first n kh (weight) samples)",
        "Sampled task_ws KH (weight)",
    )
    plot_task_ws(
        ax[1],
        kh_order_D,
        "red",
        "KH (data) order",
        "mean (first n kh (data) samples)",
        "Sampled task_ws KH (data)",
    )
    plot_task_ws(
        ax[2],
        uniform_order,
        "blue",
        "random order",
        "mean (first n random samples)",
        "Sampled task_ws random",
    )

    fig.suptitle(
        "Task ws (PCA-projected to 2d), KH (weight and data) vs uniform sampling order of tasks"
    )
    return fig, ax


def save_mmd_experiment_plots(
    env, sampled_tasks, K_w, K_D, kh_w_order, kh_D_order, plot_dir
):
    # Return data dict
    data_dict = dict()

    # fit pca and save projected scatter plot
    fitted_pca = fit_env_pca(env)
    fig, ax = plot_2d_dist(env, fitted_pca)
    fig.savefig(
        plot_dir / "task_w_pdf_on_sphere_scatter_plot.png", format="png",
    )

    # Get task weights
    task_ws = get_task_parameters(sampled_tasks)

    # Get uniform ordering
    N = K_D.shape[0]
    uniform_order = np.random.permutation(np.arange(N))

    # Get learning curves in data space
    J_kh_D = mmd2_curve(K_D, kh_D_order) ** 0.5
    J_uniform_D = mmd2_curve(K_D, uniform_order) ** 0.5
    data_dict["mmd_in_data_space"] = {"J_kh_D": J_kh_D, "J_uniform_D": J_uniform_D}

    fig, ax = plot_mmd_kh_D_vs_uniform_D(J_kh_D, J_uniform_D)
    fig.savefig(
        plot_dir / "mmd_kh_vs_uniform_in_D_space.png", format="png",
    )

    # Look at the chosen instances for each ordering
    fig, ax = plot_first_n_task_ws(
        task_ws, kh_w_order, kh_D_order, uniform_order, fitted_pca, n=25
    )
    fig.savefig(plot_dir / "n_first_task_ws_chosen_kh_vs_uniform.png", format="png")

    # Get learning curves in weight space
    J_kh_D = mmd2_curve(K_w, kh_D_order) ** 0.5
    J_kh_w = mmd2_curve(K_w, kh_w_order) ** 0.5
    J_uniform_w = mmd2_curve(K_w, uniform_order) ** 0.5
    data_dict["mmd_in_weight_space"] = {
        "J_kh_D": J_kh_D,
        "J_kh_w": J_kh_w,
        "J_uniform_D": J_uniform_D,
    }

    fig, ax = plot_mmd_kh_D_vs_KH_w_vs_uniform(J_kh_D, J_kh_w, J_uniform_w)
    fig.savefig(plot_dir / "mmd_kh_vs_uniform_in_w_space.png", format="png")

    # Clean up left over figure objects
    plt.close("all")

    return data_dict
