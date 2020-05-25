r"""Partition PyTorch datasets."""
# -*- coding: utf-8 -*-
import logging
import random

import numpy as np
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch.distributed import get_backend_tensor

_logger = logging.getLogger("mlbench")


class Partition(object):
    """Dataset-like object, but only access a subset of it.

    Wraps a dataset, only exposing the entries selected by the `indices`
    parameter.

    Args:
        data (:obj:`list` of data entries): The data to partition over
        indices (:obj:`list` of :obj:`int`): indices of entries to use
    """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]

    def __getattr__(self, item):
        return self.data.__getattribute__(item)


class Partitioner(object):
    """Use a partition of dataset."""

    def consistent_indices(self, rank, indices, shuffle):
        r""" synchronize indices among workers. """
        if rank == 0 and shuffle:
            random.shuffle(indices)

        # broadcast.
        indices = get_backend_tensor(torch.IntTensor(indices))

        dist.broadcast(indices, src=0)

        return indices.tolist()


class DataPartitioner(Partitioner):
    """ Partitions a dataset into different sized chunks.

    Used for train:test:validation split.

    Args:
        data (:obj:`list` of data entries): The data to partition over
        rank (int): The rank of the current node
        shuffle (bool): Whether to shuffle entries or not
        sizes (:obj:`list` of :obj:`float`): The relative sizes of the
            splits. Should sum up to 1.0. (Default = [0.7, 0.2, 0.1])
    """

    def __init__(self, data, rank, shuffle, sizes=[0.7, 0.2, 0.1]):
        # prepare info.
        self.data = data
        self.data_size = len(self.data)
        self.partitions = []

        # get shuffled/unshuffled data.
        indices = [x for x in range(0, self.data_size)]
        indices = self.consistent_indices(rank, indices, shuffle)

        # partition indices.
        sizes = np.cumsum(sizes)
        from_index = 0
        for ind, _ in enumerate(sizes):
            to_index = int(sizes[ind] * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

    def use(self, partition_ind):
        """Return a partition of data.

        Args:
            partition_ind (int): The index of the partition to get
        """
        return Partition(self.data, self.partitions[partition_ind])


def partition_dataset_by_rank(
    dataset, rank, world_size, distribution="uniform", shuffle=True
):
    r"""Given a dataset, partition it by a distribution and each rank takes part of data.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`): The dataset
        rank (int): The rank of the current worker
        world_size (int): The total number of workers
        distribution (str): The sampling distribution to use. Default: `uniform`
        shuffle (bool): Whether to shuffle the dataset before partitioning. Default: `True`
    """
    if distribution != "uniform":
        raise NotImplementedError(
            "Distribution {} not implemented.".format(distribution)
        )

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, shuffle, partition_sizes)
    partitioned_data = partition.use(rank)
    _logger.debug("Partition dataset use {}-th.".format(rank))
    return partitioned_data
