import logging
import os
import pickle

import cv2
import lmdb
import numpy as np
import torch.utils.data
from PIL import Image
from tensorpack.utils.compatible_serialize import loads

from mlbench_core.dataset.util.tools import progress_download

_logger = logging.getLogger("mlbench")

# All available datasets
_LIBSVM_DATASETS = [
    {"name": "australian_train", "n_samples": 690, "n_features": 14, "sparse": False},
    {"name": "duke_train", "n_samples": 38, "n_features": 7129, "sparse": True},
    {"name": "duke_test", "n_samples": 4, "n_features": 7129, "sparse": True},
    {
        "name": "epsilon_train",
        "n_samples": 400000,
        "n_features": 2000,
        "sparse": False,
        "url": "https://storage.googleapis.com/mlbench-datasets/libsvm"
        "/epsilon_train.lmdb",
    },
    {
        "name": "epsilon_test",
        "n_samples": 100000,
        "n_features": 2000,
        "sparse": False,
        "url": "https://storage.googleapis.com/mlbench-datasets/libsvm"
        "/epsilon_test.lmdb",
    },
    {"name": "rcv1_train", "n_samples": 677399, "n_features": 47236, "sparse": True},
    {"name": "synthetic_dense", "n_samples": 10000, "n_features": 100, "sparse": False},
    {
        "name": "webspam_train",
        "n_samples": 350000,
        "n_features": 16609143,
        "sparse": True,
    },
]


class LMDBDataset(torch.utils.data.Dataset):
    """
    LMDB Dataset

    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        name,
        data_type,
        root,
        is_image=False,
        target_transform=None,
        download=True,
    ):

        root, self.transform = maybe_download_lmdb(name, data_type, root)
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(
                LMDBPTClass(
                    root=lmdb_file,
                    transform=self.transform,
                    target_transform=target_transform,
                    is_image=is_image,
                )
            )

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]

        self._get_index_zones = self._build_indices()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if not self.root.endswith(".lmdb"):
            for l in os.listdir(self.root):
                if "_" in l and "-lock" not in l:
                    yield os.path.join(self.root, l)
        else:
            yield self.root

    def _build_indices(self):
        indices = self.indices
        from_to_indices = enumerate(zip(indices[:-1], indices[1:]))

        def f(x):
            if len(list(from_to_indices)) == 0:
                return 0, x

            for ind, (from_index, to_index) in from_to_indices:
                if from_index <= x < to_index:
                    return ind, x - from_index

        return f

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target)
        """
        block_index, item_index = self._get_index_zones(index)
        image, target = self.dbs[block_index][item_index]
        return image, np.array([target])

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = "Dataset {}\n".format(self.__class__.__name__)
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class LMDBPTClass(torch.utils.data.Dataset):
    """
    LMDB Dataset loader Class

    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
        is_image (bool): Whether the dataset file is an image or not
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_image = is_image

        # open lmdb env.
        self.env = self._open_lmdb()

        # get file stats.
        self._get_length()

        # prepare cache_file
        self._prepare_cache()

    def _open_lmdb(self):
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True,
            lock=False,
            readahead=False,
            map_size=1099511627776 * 2,
            max_readers=1,
            meminit=False,
        )

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

            if txn.get(b"__keys__") is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + "_cache_"
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor() if key != b"__keys__"]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _image_decode(self, x):
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype("uint8")
        return Image.fromarray(image, "RGB")

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = loads(bin_file)
        if self.is_image:
            image = self._image_decode(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.root + ")"


def construct_sparse_matrix(triplet, n_features):
    from scipy.sparse import coo_matrix, csr_matrix

    data, row, col = triplet
    mat = coo_matrix((data, (row, col)), shape=(len(set(row)), n_features))
    return csr_matrix(mat)[list(set(row))]


def maybe_transform_sparse(stats):
    return (
        (lambda x: construct_sparse_matrix(x, stats["n_features"]))
        if stats["sparse"]
        else None
    )


def get_dataset_info(name):
    stats = list(filter(lambda x: x["name"] == name, _LIBSVM_DATASETS))
    assert len(stats) == 1, "{} not found.".format(name)
    return stats[0]


def maybe_download_lmdb(name, data_type, dataset_dir):
    """Downloads the given dataset

    Args:
        name (str): Name of the dataset, one of
            `[australian, duke, epsilon, rcv1, synthetic, webspam]`
        data_type (str): One of `test` and `train`
        dataset_dir (str): Directory where to store the dataset

    Returns:
        (str, bool): path of lmdb file and flag for sparse transform
    """

    full_name = "{}_{}".format(name, data_type)
    stats = get_dataset_info(full_name)
    lmdb_path = os.path.join(dataset_dir, "{}_{}.lmdb".format(name, data_type))

    if not (os.path.exists(lmdb_path) and os.path.isfile(lmdb_path)):
        if "url" not in stats:
            raise FileNotFoundError(
                "Could not download LIBSVM dataset {}".format(full_name)
            )
        _logger.info("Downloading dataset {}".format(full_name))

        progress_download(stats["url"], dest=lmdb_path)

    return lmdb_path, maybe_transform_sparse(stats)
