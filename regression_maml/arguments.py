import argparse
import logging

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning for Meta-learning using MMD,"
        "Regression experiments."
    )

    parser.add_argument(
        "--n_train_batches",
        type=int,
        default=200,
        help="number of batches of metatrain instances",
    )
    parser.add_argument(
        "--n_val_batches",
        type=int,
        default=50,
        help="number of batches of metaval instances",
    )
    parser.add_argument(
        "--n_test_batches",
        type=int,
        default=100,
        help="number of batches of metatest instances",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tasks_per_metaupdate",
        type=int,
        default=1,
        help="number of tasks in each batch per meta-update",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=5,
        help="evaluate the test loss every evaluate_every t",
    )

    parser.add_argument(
        "--k_shot", type=int, default=5, help="number of points in train set",
    )
    parser.add_argument(
        "--k_query", type=int, default=15, help="number of points in test set",
    )

    parser.add_argument(
        "--lr_inner",
        type=float,
        default=0.01,
        help="inner-loop learning rate (per task)",
    )
    parser.add_argument(
        "--lr_meta", type=float, default=0.001, help="outer-loop learning rate",
    )
    parser.add_argument(
        "--meta_optimizer",
        type=str,
        default="adam",
        help="optimizer to use for meta loop",
    )
    parser.add_argument(
        "--frank_wolfe",
        type=str,
        default="kernel_herding",
        help="what frank wolfe style algorithm to use for active learning the metatrain order",
    )
    parser.add_argument(
        "--kernel_function",
        type=str,
        default="double_gaussian_kernel",
        help="what kernel function (on distributions) to use",
    )

    parser.add_argument(
        "--num_grad_steps_inner",
        type=int,
        default=1,
        help="number of gradient steps in inner loop (during training)",
    )
    parser.add_argument(
        "--num_grad_steps_eval",
        type=int,
        default=1,
        help="number of gradient updates at test time (for evaluation)",
    )
    parser.add_argument(
        "--num_grad_steps_meta",
        type=int,
        default=1,
        help="number of gradient updates for meta optimiser",
    )

    parser.add_argument(
        "--first_order",
        action="store_true",
        default=False,
        help="run first order approximation of MAML",
    )

    # # network architecture
    # MLP
    parser.add_argument(
        "--hidden_dim", type=int, default=40, help="number hidden units in each layer"
    )

    # # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="sine",
        help="dataset to use, current options: sine, harmonic, sinusoidandline, uniformsphere, uniformcube, vmfmixture",
    )
    # For custom datasets
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=4,
        help="number of clusters in mixture dataset",
    )
    parser.add_argument(
        "--data_dim",
        type=int,
        default=1,
        help="dimension of embedded space of the dataset",
    )
    # In order to be able to avoid dataset shift
    parser.add_argument(
        "--save_path", type=str, default=".", help="folder to save resulting data",
    )

    # # Computer
    parser.add_argument(
        "--n_workers", type=int, default=0, help="number of workers to use for CPU"
    )
    parser.add_argument(
        "--write_config",
        type=str,
        default="",
        help="Whether to write config file (used for cluster management)",
    )

    args = parser.parse_args()

    # use the GPU if available
    # This might work a bit iffy on cluster, but not sure how to get this
    # to run on GPU consistently otherwise
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Running on device: {}".format(args.device))

    return args
