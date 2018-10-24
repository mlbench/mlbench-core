r""" Create dataset and dataloader in PyTorch. """
import os
import logging
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .partition import DataPartitioner

_logger = logging.getLogger('mlbench')


class CIFAR10V1(datasets.CIFAR10):
    """CIFAR10V1 Dataset.

    Loads CIFAR10V1 images with mean and std-dev normalisation.
    Performs random crop and random horizontal flip on train and
    only normalisation on val.
    Based on `torchvision.datasets.CIFAR10` and `Pytorch CIFAR 10 Example`_.

    Args:
        root (str): Root folder for the dataset
        train (bool): Whether to get the train or validation set (default=True)
        download (bool): Whether to download the dataset if it's not present

    .. _Pytorch CIFAR 10 Example:
       https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """

    def __init__(self, root, train=True, download=False):
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }

        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    cifar10_stats['mean'], cifar10_stats['std']),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    cifar10_stats['mean'], cifar10_stats['std']),
            ])
        super(CIFAR10V1, self).__init__(root=root, train=train,
                                        transform=transform, download=download)


class Imagenet(datasets.ImageFolder):
    """Imagenet (ILSVRC2017) Dataset.

    Loads Imagenet images with mean and std-dev normalisation.
    Performs random crop and random horizontal flip on train and
    resize + center crop on val.
    Based on `torchvision.datasets.ImageFolder`

    Args:
        root (str): Root folder of Imagenet dataset (without `train/` or `val/`)
        train (bool): Whether to get the train or validation set (default=True)
    """
    def __init__(self, root, train=True):
        self.train = train

        imagenet_stats = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

        if train:
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        imagenet_stats['mean'], imagenet_stats['std']),
                ])
            self.root = os.path.join(self.root, 'train')
        else:
            transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        imagenet_stats['mean'], imagenet_stats['std'])
                ])
            self.root = os.path.join(self.root, 'val')

        super().__init__(self.root, transform)


# Map dataset name and version to class
_VERSIONED_DATASET_MAP = {
    ('cifar10', 1): CIFAR10V1,
}


def _create_dataset(train, config):
    train_or_val = 'train' if train else 'val'
    _logger.debug("Using {} dataset version {}.".format(
        train_or_val, config.dataset_version))

    dataset_path = os.path.join(config.dataset_root, config.dataset)
    _logger.debug("Loading/Downloading dataset from {}.".format(dataset_path))

    versioned_dataset = (config.dataset, config.dataset_version)
    if versioned_dataset not in _VERSIONED_DATASET_MAP:
        raise ValueError("Versioned dataset {} not found in {}.".format(
            versioned_dataset, _VERSIONED_DATASET_MAP))

    return _VERSIONED_DATASET_MAP[versioned_dataset](dataset_path, train, download=True)


def partition_dataset_by_rank(dataset, rank, world_size, distribution='uniform', shuffle=True):
    r"""Given a dataset, partition it by a distribution and each rank takes part of data. """
    if distribution != 'uniform':
        raise NotImplementedError(
            "Distribution {} not implemented.".format(distribution))

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(
        dataset, rank, shuffle, partition_sizes)
    partitioned_data = partition.use(rank)
    _logger.debug("Partition dataset use {}-th.".format(rank))
    return partitioned_data


def create_partition_transform_dataset(train, config):
    r"""Return a transformed and partitioned dataset.

    The full dataset is partitioned into `world_size` partitions. Each process uses the partition
    corresponding to its rank.
    :param train: Use training dataset or validation/test dataset
    :type train: boolean
    :param config: configurations
    :type config: Namespace
    :returns: A partial dataset.
    :rtype: {}
    """
    full_dataset = _create_dataset(train, config)
    partitioned_dataset = partition_dataset_by_rank(
        full_dataset, config.rank, config.world_size,
        shuffle=config.shuffle_before_partition)

    num_samples_per_device = len(partitioned_dataset)
    num_batches_per_device = math.ceil(
        1.0 * num_samples_per_device / config.batch_size)

    if train:
        config.num_samples_per_device_train = num_samples_per_device
        config.num_batches_per_device_train = num_batches_per_device
    else:
        config.num_samples_per_device_val = num_samples_per_device
        config.num_batches_per_device_val = num_batches_per_device
    return partitioned_dataset
