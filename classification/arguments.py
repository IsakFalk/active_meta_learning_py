import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning for Meta-learning using MMD,"
        "Clasification experiments."
    )

    parser.add_argument(
        "--n_train_batches",
        type=int,
        default=1000,
        help="number of batches of metatrain instances",
    )
    parser.add_argument(
        "--n_test_batches",
        type=int,
        default=200,
        help="number of batches of metatest instances",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tasks_per_metaupdate",
        type=int,
        default=16,
        help="number of tasks in each batch per meta-update",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=100,
        help="evaluate the test loss / accuracy every evaluate_every t",
    )

    parser.add_argument(
        "--n_way", type=int, default=5, help="number of object classes to learn"
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=1,
        help="number of examples per class to learn from",
    )
    parser.add_argument(
        "--k_query",
        type=int,
        default=15,
        help="number of examples to evaluate on (in outer loop)",
    )

    parser.add_argument(
        "--lr_inner",
        type=float,
        default=0.5,
        help="inner-loop learning rate (per task)",
    )
    parser.add_argument(
        "--lr_meta",
        type=float,
        default=0.001,
        help="outer-loop learning rate (used with Adam optimiser)",
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
        default="mean_linear",
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
        "--first_order",
        action="store_true",
        default=False,
        help="run first order approximation of CAVIA",
    )

    # # network architecture
    parser.add_argument(
        "--model", type=str, default="cnn", help="model to use",
    )
    # CNN
    parser.add_argument(
        "--num_filters", type=int, default=32, help="number of filters per conv-layer"
    )
    # MLP
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="number of hidden units per layer in MLP",
    )

    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of hidden layers"
    )
    # parser.add_argument(
    #     "--nn_initialisation",
    #     type=str,
    #     default="zero",
    #     help="initialisation type (kaiming, xavier, zero)",
    # )

    # # Data
    parser.add_argument(
        "--dataset", type=str, default="omniglot", help="dataset to use"
    )
    # In order to be able to avoid dataset shift
    parser.add_argument(
        "--base_dataset_train",
        type=str,
        default="train",
        help="Base dataset to use to sample train from",
    )
    parser.add_argument(
        "--base_dataset_val",
        type=str,
        default="val",
        help="Base dataset to use to sample val from",
    )
    parser.add_argument(
        "--base_dataset_test",
        type=str,
        default="test",
        help="Base dataset to use to sample test from",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="folder which contains image data",
    )
    parser.add_argument(
        "--save_path", type=str, default=".", help="folder to save resulting data",
    )

    # # Computer
    parser.add_argument(
        "--n_workers", type=int, default=0, help="number of workers to use for CPU"
    )
    parser.add_argument(
        "--write_config", type=str, default="", help="where to write config"
    )
    
    args = parser.parse_args()

    # use the GPU if available
    # This might work a bit iffy on cluster, but not sure how to get this
    # to run on GPU consistently otherwise
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(args.device))

    return args
