import lmdb
import os
import torch.utils.data
import pickle
import numpy as np
import math
import logging

from tensorpack.utils.compatible_serialize import loads
from tensorpack.dataflow.serialize import LMDBSerializer

from .partition import DataPartitioner

_logger = logging.getLogger('mlbench')

_LIBSVM_DATASETS = [
    {'name': 'webspam', 'n_samples': 350000, 'n_features': 16609143, 'sparse': True},
    {'name': 'epsilon-train', 'n_samples': 400000, 'n_features': 2000, 'sparse': False},
    {'name': 'duke-train', 'n_samples': 44, 'n_features': 7129, 'sparse': True},
    {'name': 'australian-train', 'n_samples': 690, 'n_features': 14, 'sparse': False},
    {'name': 'rcv1-train', 'n_samples': 677399, 'n_features': 47236, 'sparse': True},
    {'name': 'synthetic-dense', 'n_samples': 10000, 'n_features': 100, 'sparse': False},
]


class IMDBPT(torch.utils.data.Dataset):
    """
    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True, n_features=None):
        self.n_features = n_features
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(LMDBPTClass(
                root=lmdb_file, transform=transform,
                target_transform=target_transform, is_image=is_image))

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._get_index_zones = self._build_indices()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if not self.root.endswith('.lmdb'):
            for l in os.listdir(self.root):
                if '_' in l and '-lock' not in l:
                    yield os.path.join(self.root, l)
        else:
            yield self.root

    def _build_indices(self):
        indices = self.indices
        from_to_indices = enumerate(zip(indices[: -1], indices[1:]))

        def f(x):
            if len(list(from_to_indices)) == 0:
                return 0, x

            for ind, (from_index, to_index) in from_to_indices:
                if from_index <= x and x < to_index:
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
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class LMDBPTClass(torch.utils.data.Dataset):
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
            readonly=True, lock=False, readahead=False,
            map_size=1099511627776 * 2,
            max_readers=1, meminit=False)

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

            if txn.get(b'__keys__') is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + '_cache_'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key
                             for key, _ in txn.cursor() if key != b'__keys__']
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _image_decode(self, x):
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype('uint8')
        return Image.fromarray(image, 'RGB')

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
        return self.__class__.__name__ + ' (' + self.root + ')'


def construct_sparse_matrix(triplet, n_features):
    from scipy.sparse import coo_matrix, csr_matrix
    data, row, col = triplet
    mat = coo_matrix((data, (row, col)), shape=(len(set(row)), n_features))
    return csr_matrix(mat)[list(set(row))]


def maybe_transform_sparse(stats):
    return (lambda x: construct_sparse_matrix(x, stats['n_features'])) \
        if stats['sparse'] else None


def get_dataset_info(name):
    stats = list(filter(lambda x: x['name'] == name, _LIBSVM_DATASETS))
    assert len(stats) == 1, '{} not found.'.format(name)
    return stats[0]


def load_libsvm_lmdb(name, lmdb_path):
    stats = get_dataset_info(name)
    dataset = IMDBPT(lmdb_path, transform=maybe_transform_sparse(stats),
                     target_transform=None, is_image=False)
    return dataset


def partition_dataset_by_rank(dataset, rank, world_size, distribution='uniform', shuffle=True):
    r"""Given a dataset, partition it by a distribution and each rank takes part of data.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`): The dataset
        rank (int): The rank of the current worker
        world_size (int): The total number of workers
        distribution (str): The sampling distribution to use. Default: `uniform`
        shuffle (bool): Whether to shuffle the dataset before partitioning. Default: `True`
    """
    if distribution != 'uniform':
        raise NotImplementedError(
            "Distribution {} not implemented.".format(distribution))

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(
        dataset, rank, shuffle, partition_sizes)
    partitioned_data = partition.use(rank)
    _logger.debug("Partition dataset use {}-th.".format(rank))
    return partitioned_data

