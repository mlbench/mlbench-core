import logging
import os
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from mlbench_core.dataset.util.tools import maybe_download_and_extract_tar_gz

from .wmt17 import Dictionary, collate_batch

logger = logging.getLogger("mlbench")

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def _read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def _index_file_path(prefix_path):
    return prefix_path + ".idx"


def _data_file_path(prefix_path):
    return prefix_path + ".bin"


class WMT17Dataset(Dataset):
    """Dataset for WMT17 EN to DE translation, for transformer model.

    The dataset needs to be pre-processed before training by using the script
    `mlbench_core/dataset/nlp/pytorch/wmt17/preprocess/preprocess.py`.

    This class uses the `.bin` and `.idx` files, outputted by the pre-processing script

    Args:
        root (str): Root folder where to download files
        lang (tuple): Language pair
        download (bool): Download dataset from gcloud S3
        train (bool): Load train set
        validation (bool): Load validation set
        test (bool): Load test set
        left_pad (tuple[bool]): left- or right-padding (true: left, false: right) for (source, target)
        max_positions (tuple[int]): Maximum number of tokens in (source, target)
        seq_len_multiple (int): Pad sources to multiples of this
        shuffle (bool): Shuffle dataset
    """

    urls = [
        (
            "https://storage.googleapis.com/mlbench-datasets/translation/wmt17_en_de.tar.gz",
            "wmt17_en_de.tar.gz",
        )
    ]
    name = "wmt17"
    dirname = ""

    prefixes = {"train": "train", "validation": "dev", "test": "test"}

    def __init__(
        self,
        root,
        lang=("en", "de"),
        download=False,
        train=False,
        validation=False,
        test=False,
        left_pad=(True, False),
        max_positions=(256, 256),
        seq_len_multiple=1,
        shuffle=True,
    ):
        src_lang, trg_lang = lang
        self.left_pad_source, self.left_pad_target = left_pad
        self.max_source_positions, self.max_target_positions = max_positions

        self.seq_len_multiple = seq_len_multiple
        self.shuffle = shuffle
        if download:
            url, file_name = self.urls[0]
            maybe_download_and_extract_tar_gz(root, file_name, url)

        self.src_dict = Dictionary.load(
            os.path.join(root, "dict.{}.txt".format(src_lang))
        )
        self.trg_dict = Dictionary.load(
            os.path.join(root, "dict.{}.txt".format(trg_lang))
        )

        if train:
            self.prefix = self.prefixes["train"]
        elif validation:
            self.prefix = self.prefixes["validation"]
        elif test:
            self.prefix = self.prefixes["test"]
        else:
            raise NotImplementedError()

        assert self.src_dict.pad() == self.trg_dict.pad()
        assert self.src_dict.eos() == self.trg_dict.eos()

        self.src_path = os.path.join(
            root, "{}.{}-{}.{}".format(self.prefix, src_lang, trg_lang, src_lang)
        )
        self.trg_path = os.path.join(
            root, "{}.{}-{}.{}".format(self.prefix, src_lang, trg_lang, trg_lang)
        )

        assert self._exists()

        self.src_data = IndexedDataset(self.src_path)
        self.trg_data = IndexedDataset(self.trg_path)

        self.src_sizes = np.array(self.src_data.sizes)
        self.trg_sizes = np.array(self.trg_data.sizes)

        print(
            "| Sentences are being padded to multiples of: {}".format(
                self.seq_len_multiple
            )
        )

    def __len__(self):
        return len(self.src_data)

    def _exists(self):
        """Checks if all the files needed exist

        Returns:
            (bool): True of all the files exist
        """
        return (
            os.path.exists(_index_file_path(self.src_path))
            and os.path.exists(_data_file_path(self.src_path))
            and os.path.exists(_index_file_path(self.trg_path))
            and os.path.join(_data_file_path(self.trg_path))
        )

    def __getitem__(self, index):
        return {
            "id": index,
            "source": self.src_data[index],
            "target": self.trg_data[index],
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate_batch(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            bsz_mult=8,
            seq_len_multiple=self.seq_len_multiple,
        )

    def ordered_indices(self, partition_indices=None, seed=None):
        """Returns a list of indices, ordered by source and target lengths

        If `partition_indices` is not `None`, only those indices will be returned,
        but sorted.

        The indices will be shuffled before sorting, using the given seed

        Args:
            partition_indices (Optional[list[int]]): The list of indices to sort
                Default: `None`
            seed (Optional[int]): Seed to use for shuffling

        Returns:
            (`obj`:np.array): Array of sorted indices
        """
        if self.shuffle:
            indices = np.random.RandomState(seed).permutation(
                len(self) if partition_indices is None else partition_indices
            )
        else:
            indices = (
                np.arange(len(self)) if partition_indices is None else partition_indices
            )

        if self.trg_sizes is not None:
            indices = indices[np.argsort(self.trg_sizes[indices], kind="mergesort")]

        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]


class IndexedDataset(Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path):
        super().__init__()
        # Read Index file
        with open(_index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == b"TNTIDX\x00\x00"
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack("<QQ", f.read(16))
            self.dim_offsets = _read_longs(f, self.size + 1)
            self.data_offsets = _read_longs(f, self.size + 1)
            self.sizes = _read_longs(f, self.s)
        with open(_data_file_path(path), "rb") as f:
            self.buffer = np.empty(self.data_offsets[-1], dtype=self.dtype)
            f.readinto(self.buffer)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        np.copyto(a, self.buffer[self.data_offsets[i] : self.data_offsets[i + 1]])
        return torch.from_numpy(a).long()

    def __len__(self):
        return self.size
