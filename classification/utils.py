import random
import json

import torch
import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import numpy as np

##########
# Torch  #
##########


def swish(x):
    return x * F.sigmoid(x)


####################
# Reproducibility  #
####################


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    # if (seed is not None) and cudnn:
    #     torch.backends.cudnn.deterministic = True


####################################
# Get dataloader data into fw form #
####################################


def aggregate_sampled_task_batches(dataloader, num_batches):
    """
    Get num_batches batch of tasks from dataloader.
    :param dataloader: torchmeta dataloader
    :param num_batches: number of batches to sample
    :return sampled_batches: list of samples (dicts)
    """
    sampled_batches = []
    for batch_idx, batch in enumerate(dataloader):
        if num_batches <= batch_idx:
            break
        sampled_batches.append(batch)
    return sampled_batches


def coalesce_train_and_test_in_dicts(sampled_batches):
    """
    Take a list of batches of tasks and coalesce the train/test
    split into one dataset for herding / FW.
    :param sampled_batches: list of dictionaries of train/test tensors from dataloader
    :return new_list: same list but with data in place of train and test,
    tensors concatenated over dimension 1.
    """
    new_list = []
    for batch in sampled_batches:
        X_train, y_train = batch["train"]
        X_test, y_test = batch["test"]
        X_task = torch.cat((X_train, X_test), dim=1)
        y_task = torch.cat((y_train, y_test), dim=1)
        new_list.append({"X": X_task, "y": y_task})
    return new_list


def concatenate_X_and_y(X, y):
    """
    Concatete X and y where y is a integer coded class to be ohe and X will be flattened
    """
    _X = torch.flatten(X, start_dim=2)
    _y = F.one_hot(y).float()
    batch_dim = _X.shape[0]
    examples_dim = _X.shape[1]
    feature_dim = _X.shape[2] + _y.shape[2]
    new_shape = (batch_dim, examples_dim, feature_dim)
    D = torch.zeros(new_shape)
    D[:, :, : _X.shape[2]] = _X
    D[:, :, _X.shape[2] :] = _y
    return D


def coalesce_X_and_y_in_dicts(sampled_batches):
    """
    Coalesce X and y into one dataset D in
    """
    new_list = []
    for batch in sampled_batches:
        X, y = batch["X"], batch["y"]
        D = concatenate_X_and_y(X, y)
        new_list.append(D)
    return new_list


def make_X_to_D(sampled_batches):
    new_list = []
    for batch in sampled_batches:
        _X = torch.flatten(batch["X"], start_dim=2)
        new_list.append(_X)
    return new_list


def remove_batched_dimension_in_D(sampled_batches):
    new_list = []
    for batch in sampled_batches:
        D = batch
        new_list.append(D.reshape(-1, D.shape[-1]))
    return new_list


def convert_batches_to_np(sampled_batches):
    return [batch.cpu().detach().numpy() for batch in sampled_batches]


def convert_batches_to_fw_form(sampled_batches):
    with torch.no_grad():
        sampled_batches = coalesce_train_and_test_in_dicts(sampled_batches)
        # sampled_batches = coalesce_X_and_y_in_dicts(sampled_batches)
        sampled_batches = make_X_to_D(sampled_batches)
        sampled_batches = remove_batched_dimension_in_D(sampled_batches)
        sampled_batches = convert_batches_to_np(sampled_batches)
        return np.stack(sampled_batches)


########
# Main #
########


def get_active_learning_batches(sampled_batches_train, kernel_mat_func, fw_class):
    fw_batches_train = convert_batches_to_fw_form(sampled_batches_train)
    K = kernel_mat_func(fw_batches_train)
    fw = fw_class(K)
    fw.run()
    sampled_batches_train_fw = [sampled_batches_train[i] for i in fw.sampled_order]
    return sampled_batches_train_fw


def generate_dataloaders(args):
    # Create dataset
    if args.dataset == "omniglot":
        dataset_train = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_train,
            seed=args.seed,
            download=True,  # Only downloads if not in data_path
        )
        dataset_val = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_val,
            seed=args.seed,
            download=True,
        )
        dataset_test = omniglot(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_test,
            download=True,
        )
    elif args.dataset == "miniimagenet":
        dataset_train = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_train,
            seed=args.seed,
            download=True,  # Only downloads if not in data_path
        )
        dataset_val = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_val,
            seed=args.seed,
            download=True,
        )
        dataset_test = miniimagenet(
            folder=args.data_path,
            shots=args.k_shot,
            ways=args.n_way,
            shuffle=True,
            test_shots=args.k_query,
            meta_split=args.base_dataset_test,
            download=True,
        )
    else:
        raise Exception("Dataset {} not implemented".format(args.dataset))

    dataloader_train = BatchMetaDataLoader(
        dataset_train,
        batch_size=args.tasks_per_metaupdate,
        shuffle=True,
        num_workers=args.n_workers,
    )
    dataloader_val = BatchMetaDataLoader(
        dataset_val,
        batch_size=args.tasks_per_metaupdate,
        shuffle=True,
        num_workers=args.n_workers,
    )
    dataloader_test = BatchMetaDataLoader(
        dataset_test,
        batch_size=args.tasks_per_metaupdate,
        shuffle=True,
        num_workers=args.n_workers,
    )

    return dataloader_train, dataloader_val, dataloader_test


###########################
# Create config from args #
###########################


def dump_args_to_json(args, save_path):
    d = dict(args._get_kwargs())
    d.pop("device", None)
    d.pop("num_workers", None)
    with open(save_path / "config.json", "w+") as f:
        json.dump(d, f, sort_keys=True, indent=4)


def dump_runs_to_npy(
    test_loss_uniform, test_acc_uniform, test_loss_fw, test_acc_fw, save_path,
):
    np.save(save_path / "test_loss_uniform.npy", test_loss_uniform)
    np.save(save_path / "test_acc_uniform.npy", test_acc_uniform)
    np.save(save_path / "test_loss_fw.npy", test_loss_fw)
    np.save(save_path / "test_acc_fw.npy", test_acc_fw)


def dump_runs_to_json(runs_dict, save_path):
    with open(save_path / "runs.json", "w+") as f:
        json.dump(runs_dict, f)
