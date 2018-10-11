import torch
import os
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ..partition import DataPartitioner

logger = logging.getLogger('mlbench')


class CIFAR10V1(datasets.CIFAR10):
    """
    * https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """

    def __init__(self, root, train=True, download=False):
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }

        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
            ])
        super(CIFAR10V1, self).__init__(root=root, train=train, transform=transform, download=download)


def _create_dataset(train, options):
    train_or_val = 'train' if train else 'val'
    logger.debug("Using {} dataset version {}.".format(train_or_val, options.dataset_version))

    dataset_path = os.path.join(options.dataset_root, options.dataset)
    logger.debug("Loading/Downloading dataset from {}.".format(dataset_path))

    if options.dataset_version == 'v1':
        return CIFAR10V1(dataset_path, train, download=True)
    else:
        raise NotImplementedError


def partition_dataset(dataset, rank, world_size, distribution='uniform', shuffle_before_partition=True):
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, rank, shuffle_before_partition, partition_sizes)
    partitioned_data = partition.use(rank)
    logger.debug("Partition dataset use {}-th.".format(rank))
    return partitioned_data


def create_partition_transform_dataset(train, options):
    """Return a torchvision.datasets."""
    full_dataset = _create_dataset(train, options)
    partitioned_dataset = partition_dataset(full_dataset, options.rank, options.world_size,
                                            shuffle_before_partition=options.shuffle_before_partition)

    return partitioned_dataset
