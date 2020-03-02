# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# above licence recreated from LEO code ^^^
# Copied from LEO code, changed to fit with pytorch
"""Create problem instance for embedeed miniimagenet and tieredimagenet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import random

import enum
import numpy as np
import six
import torch
from torch.utils.data import IterableDataset
import tensorflow as tf


NDIM = 640

ProblemInstance = collections.namedtuple(
    "ProblemInstance",
    ["tr_input", "tr_output", "tr_info", "val_input", "val_output", "val_info"],
)


class StrEnum(enum.Enum):
    """An Enum represented by a string."""

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()


class MetaDataset(StrEnum):
    """Datasets supported by the DataProvider class."""

    MINI = "miniImageNet"
    TIERED = "tieredImageNet"


class EmbeddingCrop(StrEnum):
    """Embedding types supported by the DataProvider class."""

    CENTER = "center"
    MULTIVIEW = "multiview"


class MetaSplit(StrEnum):
    """Meta-datasets split supported by the DataProvider class."""

    TRAIN = "train"
    VALID = "val"
    TEST = "test"


class DataProvider:
    """Creates problem instances from a specific split and dataset."""


"""DataLoader for embedded miniimagenet / tieredimagenet"""


class LeoEmbeddingDataset(IterableDataset):
    def __init__(
        self,
        data_path,
        dataset_name="miniImageNet",
        dataset_split="train",
        embedding_crop="center",
        train_on_val=False,  # use train+val into bigger train set
        n_way=5,
        k_shot=1,
        k_query=15,
        total_examples_per_class=600,
    ):
        super(LeoEmbeddingDataset).__init__()
        self._data_path = data_path
        self._dataset_name = MetaDataset(dataset_name)
        self._dataset_split = MetaSplit(dataset_split)
        self._embedding_crop = EmbeddingCrop(embedding_crop)
        self._train_on_val = train_on_val
        self._n_way = n_way
        self._k_shot = k_shot
        self._k_query = k_query
        self._total_examples_per_class = total_examples_per_class

        self._index_data(self._load_data())

    def _load_data(self):
        """Loads data into memory and caches ."""
        raw_data = self._load(
            tf.io.gfile.GFile(self._get_full_pickle_path(self._dataset_split), "rb")
        )
        if self._dataset_split == MetaSplit.TRAIN and self._train_on_val:
            valid_data = self._load(
                tf.io.gfile.GFile(self._get_full_pickle_path(MetaSplit.VALID), "rb")
            )
            for key in valid_data:
                raw_data[key] = np.concatenate([raw_data[key], valid_data[key]], axis=0)

        return raw_data

    def _load(self, opened_file):
        if six.PY2:
            result = pickle.load(opened_file)
        else:
            result = pickle.load(
                opened_file, encoding="latin1"
            )  # pylint: disable=unexpected-keyword-arg
        return result

    def _index_data(self, raw_data):
        """Builds an index of images embeddings by class."""
        self._all_class_images = collections.OrderedDict()
        self._image_embedding = collections.OrderedDict()
        for i, k in enumerate(raw_data["keys"]):
            _, class_label, image_file = k.split("-")
            image_file_class_label = image_file.split("_")[0]
            assert class_label == image_file_class_label
            self._image_embedding[image_file] = raw_data["embeddings"][i]
            if class_label not in self._all_class_images:
                self._all_class_images[class_label] = []
            self._all_class_images[class_label].append(image_file)

        self._check_data_index(raw_data)

        self._all_class_images = collections.OrderedDict(
            [(k, np.array(v)) for k, v in six.iteritems(self._all_class_images)]
        )

    def _check_data_index(self, raw_data):
        """Performs checks of the data index and image counts per class."""
        n = raw_data["keys"].shape[0]
        error_message = "{} != {}".format(len(self._image_embedding), n)
        assert len(self._image_embedding) == n, error_message
        error_message = "{} != {}".format(raw_data["embeddings"].shape[0], n)
        assert raw_data["embeddings"].shape[0] == n, error_message

        all_class_folders = list(self._all_class_images.keys())
        error_message = "no duplicate class names"
        assert len(set(all_class_folders)) == len(all_class_folders), error_message
        image_counts = set(
            [len(class_images) for class_images in self._all_class_images.values()]
        )
        error_message = (
            "len(image_counts) should have at least one element but " "is: {}"
        ).format(image_counts)
        assert len(image_counts) >= 1, error_message
        assert min(image_counts) > 0

    def _get_full_pickle_path(self, split_name):
        full_pickle_path = os.path.join(
            self._data_path,
            str(self._dataset_name),
            str(self._embedding_crop),
            "{}_embeddings.pkl".format(split_name),
        )
        return full_pickle_path

    def _generate_one_instance_py(self, n_way, k_shot, k_query):
        """Iterator yielding a dataset of n_way classes, k_show and k_query samples per class"""

        # Should wrap this in while true and yield a sample in the end
        class_list = list(self._all_class_images.keys())
        sample_count = k_shot + k_query
        embeddings = self._image_embedding

        while True:
            shuffled_folders = random.sample(class_list, n_way)
            random.shuffle(shuffled_folders)
            error_message = "len(shuffled_folders) {} is not n_way: {}".format(
                len(shuffled_folders), n_way
            )
            image_paths = []
            class_ids = []
            for class_id, class_name in enumerate(shuffled_folders):
                all_images = self._all_class_images[class_name]
                all_images = np.random.choice(all_images, sample_count, replace=False)
                error_message = "{} == {} failed".format(len(all_images), sample_count)
                assert len(all_images) == sample_count, error_message
                image_paths.append(all_images)
                class_ids.append([[class_id]] * sample_count)

            label_array = np.array(class_ids, dtype=np.int32)
            path_array = np.array(image_paths)
            embedding_array = np.array(
                [
                    [embeddings[image_path] for image_path in class_paths]
                    for class_paths in path_array
                ]
            )
            # Current form [n_way, k_shot + k_query, last_dim]
            yield embedding_array, label_array

    def __iter__(self):
        return self._generate_one_instance_py(self._n_way, self._k_query, self._k_shot)

    def _embedding_collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples (D_x, D_y).

        data is a task sampled from embedding dataset and contains D_x
        which correspond to the inputs and D_y to the labels. Dimensions are
        D_x: (n_way, k_shot+k_query, emb_d)
        D_y: (n_way, k_shot+k_query, 1)

        We batch this into a dict batch, of batch size B, which looks like
        keys(batch) = ["train", "test"]
        batch[key] = ({key}_xs, {key}_ys)
        such that
        {key}_xs: (B, n_way*k_{shot, query}, emb_d)
        {key}_ys: (B, n_way*k_{shot, query}, 1)
        """
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        for D_x, D_y in data:
            k_x = D_x.shape[1]
            k_y = D_y.shape[1]
            assert (
                k_x == self._k_shot + self._k_query
            ), "k_shot ({}) + k_query ({}) !== {}".format(
                self._k_shot, self._k_query, k_x
            )
            assert (
                k_y == self._k_shot + self._k_query
            ), "k_shot ({}) + k_query ({}) !== {}".format(
                self._k_shot, self._k_query, k_y
            )
            train_xs.append(
                torch.tensor(
                    D_x[:, : self._k_shot, :].reshape(self._n_way * self._k_shot, -1)
                )
            )
            train_ys.append(
                torch.tensor(
                    D_y[:, : self._k_shot, :].reshape(self._n_way * self._k_shot, -1)
                )
            )
            test_xs.append(
                torch.tensor(
                    D_x[:, self._k_shot :, :].reshape(self._n_way * self._k_query, -1)
                )
            )
            test_ys.append(
                torch.tensor(
                    D_y[:, self._k_shot :, :].reshape(self._n_way * self._k_query, -1)
                )
            )

        # stack the datasets along dimension 0
        train_x = torch.stack(train_xs, dim=0)
        train_y = torch.stack(train_ys, dim=0)
        test_x = torch.stack(test_xs, dim=0)
        test_y = torch.stack(test_ys, dim=0)
        collated_batch = {"train": (train_x, train_y), "test": (test_x, test_y)}
        return collated_batch
