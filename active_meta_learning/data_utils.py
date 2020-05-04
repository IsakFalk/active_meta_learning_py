import json

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from active_meta_learning.data import EnvironmentDataSet


def set_random_seeds(seed):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)


################
# Sphere data  #
################


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


def get_task_parameters(sampled_batches):
    w_list = []
    for batch in sampled_batches:
        w_list.append(batch["w"].squeeze())
    return np.stack(w_list)


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
    Concatete X and y
    """
    batch_dim = X.shape[0]
    examples_dim = X.shape[1]
    feature_dim = X.shape[2] + y.shape[2]
    new_shape = (batch_dim, examples_dim, feature_dim)
    D = torch.zeros(new_shape)
    D[:, :, : X.shape[2]] = X
    D[:, :, X.shape[2] :] = y
    return D


def coalesce_X_and_y_to_D(sampled_batches):
    """
    Coalesce X and y into one dataset D from dicts
    """
    new_list = []
    for batch in sampled_batches:
        X, y = batch["X"], batch["y"]
        D = concatenate_X_and_y(X, y)
        new_list.append(D)
    return new_list


def remove_batched_dimension_in_D(sampled_batches):
    new_list = []
    for batch in sampled_batches:
        D = batch
        new_list.append(D.reshape(-1, *tuple(D.shape[2:])))
    return new_list


def convert_batches_to_np(sampled_batches):
    return [batch.cpu().detach().numpy() for batch in sampled_batches]


def convert_batches_to_fw_form(sampled_batches):
    with torch.no_grad():
        sampled_batches = coalesce_train_and_test_in_dicts(sampled_batches)
        sampled_batches = coalesce_X_and_y_to_D(sampled_batches)
        sampled_batches = remove_batched_dimension_in_D(sampled_batches)
        sampled_batches = convert_batches_to_np(sampled_batches)
        return np.stack(sampled_batches)


###########################
# Create config from args #
###########################


def dump_args_to_json(args, save_path):
    d = dict(args._get_kwargs())
    d.pop("device", None)
    d.pop("num_workers", None)
    with open(save_path / "config.json", "w+") as f:
        json.dump(d, f, sort_keys=True, indent=4)


def dump_runs_to_npy(test_loss_uniform, test_loss_fw, save_path):
    np.save(save_path / "test_loss_uniform.npy", test_loss_uniform)
    np.save(save_path / "test_loss_fw.npy", test_loss_fw)


def dump_runs_to_json(runs_dict, save_path):
    with open(save_path / "runs.json", "w+") as f:
        json.dump(runs_dict, f)


########
# Misc #
########


def form_datasets_from_tasks(tasks, use_only_train=False):
    datasets = []
    for task in tasks:
        X_tr, y_tr = task["train"]
        X_te, y_te = task["test"]
        if use_only_train:
            X = X_tr
            y = y_tr
        else:
            X = np.concatenate((X_tr, X_te), axis=0)
            y = np.concatenate((y_tr, y_te), axis=0)
        D = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        datasets.append(D)
    # adds new axis
    datasets = np.stack(datasets, axis=0)
    return datasets


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


def reorder_list(l, new_order):
    return [l[i] for i in new_order]


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
