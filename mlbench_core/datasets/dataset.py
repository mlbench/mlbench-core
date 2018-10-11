import logging
import torch

from .cifar10.pytorch import create_partition_transform_dataset

logger = logging.getLogger('mlbench')

_DATASET_NAMES = ['cifar10', 'imagenet']


class Dataset(object):
    """A wrapper class."""

    def __init__(self, backend):
        self.backend = backend

    def __iter__(self):
        # TODO:
        raise NotImplementedError

    def __str__(self):
        return 'Dataset(backend={backend})'.format(backend=self.backend)


class PyTorchDataset(Dataset):
    def __init__(self, dataloader):
        super(PyTorchDataset, self).__init__(backend='pytorch')

        self.dataloader = dataloader

    def __repr__(self):
        return 'PyTorchDataset({})'.format(self.dataloader.__repr__())


def make_pytorch_dataset(train, options):
    """Create a pytorch dataloader and wrap it."""
    logger.debug("Creating PyTorch dataset.")
    dataset = create_partition_transform_dataset(train, options)

    # Loaded dataset with dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=options.batch_size, shuffle=train,
        num_workers=options.num_parallel_workers, pin_memory=options.use_cuda, drop_last=False)

    return PyTorchDataset(data_loader)


def make_tensorflow_dataset(train, options):
    raise NotImplementedError


def create_dataset(train, options):
    if options.dataset not in _DATASET_NAMES:
        raise ValueError("Supported datasets {}. Got {}.".format(_DATASET_NAMES, options.dataset_name))

    if options.pytorch:
        return make_pytorch_dataset(train, options)

    if options.tensorflow:
        return make_tensorflow_dataset(train, options)

    raise NotImplementedError("DataLoader should be tf or torch.")
