
import numpy as np
import torch

####################################
# Get dataloader data into fw form #
####################################


def aggregate_sampled_task_batches(dataloader):
    """
    Get num_batches batch of tasks from dataloader.
    :param dataloader: torchmeta dataloader
    :param num_batches: number of batches to sample
    :return sampled_batches: list of samples (dicts)
    """
    sampled_batches = []
    for batch in dataloader:
        sampled_batches.append(batch)
    return sampled_batches


def make_toy_into_dict_list(sampled_batches):
    new_list = []
    for batch in sampled_batches:
        X = batch[0]
        y = batch[1]
        new_list.append(dict(X=X, y=y))
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
        sampled_batches = make_toy_into_dict_list(sampled_batches)
        sampled_batches = coalesce_X_and_y_to_D(sampled_batches)
        sampled_batches = remove_batched_dimension_in_D(sampled_batches)
        sampled_batches = convert_batches_to_np(sampled_batches)
        return np.stack(sampled_batches)


## Simply for training


def convert_batches_to_nn_dict_train_test_form(sampled_batches, k_train):
    new_list = []
    sampled_batches = make_toy_into_dict_list(sampled_batches)
    for batch in sampled_batches:
        X = batch["X"]
        y = batch["y"]
        X_train = X[:, :k_train]
        X_test = X[:, k_train:]
        y_train = y[:, :k_train]
        y_test = y[:, k_train:]
        new_list.append(dict(train=(X_train, y_train), test=(X_test, y_test)))
    return new_list


########
# Main #
########


def get_active_learning_batches(
    sampled_batches_train, k_train, kernel_mat_func, fw_class
):
    nn_sampled_batches_train = convert_batches_to_nn_dict_train_test_form(
        sampled_batches_train, k_train
    )
    fw_batches_train = convert_batches_to_fw_form(sampled_batches_train)
    K = kernel_mat_func(fw_batches_train)
    fw = fw_class(K)
    fw.run()
    nn_sampled_batches_train_fw = [
        nn_sampled_batches_train[i] for i in fw.sampled_order
    ]
    return nn_sampled_batches_train_fw
