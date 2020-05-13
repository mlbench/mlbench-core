import logging
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

_logger = logging.getLogger("mlbench")


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
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
                ]
            )
        super(CIFAR10V1, self).__init__(
            root=root, train=train, transform=transform, download=download
        )


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

        imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_stats["mean"], imagenet_stats["std"]),
                ]
            )
            self.root = os.path.join(self.root, "train")
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_stats["mean"], imagenet_stats["std"]),
                ]
            )
            self.root = os.path.join(self.root, "val")

        super().__init__(self.root, transform)
