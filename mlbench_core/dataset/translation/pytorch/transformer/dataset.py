import os
import struct

import torch
from torch.utils.data import Dataset

from mlbench_core.dataset.translation.pytorch.transformer import Dictionary
from mlbench_core.dataset.translation.pytorch.transformer.utils import *


def _read_index_file(path):
    with open(index_file_path(path), "rb") as f:
        magic = f.read(8)
        assert magic == b"TNTIDX\x00\x00"
        version = f.read(8)
        assert struct.unpack("<Q", version) == (1,)
        code, element_size = struct.unpack("<QQ", f.read(16))
        dtype = dtypes[code]
        size, s = struct.unpack("<QQ", f.read(16))
        dim_offsets = read_longs(f, size + 1)
        data_offsets = read_longs(f, size + 1)
        sizes = read_longs(f, s)

    return dtype, dim_offsets, data_offsets, sizes


class IndexedDataset(Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path):
        super().__init__()
        # Read Index file
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == b"TNTIDX\x00\x00"
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack("<QQ", f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)
        with open(data_file_path(path), "rb") as f:
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


class WMT17Dataset(Dataset):
    urls = [
        (
            "https://storage.googleapis.com/mlbench-datasets/translation/wmt16_en_de.tar.gz",
            "wmt16_en_de.tar.gz",
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
        train=True,
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
            pass  # Download tar.gz file

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

        assert self.exists()

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

    def exists(self):

        return (
            os.path.exists(index_file_path(self.src_path))
            and os.path.exists(data_file_path(self.src_path))
            and os.path.exists(index_file_path(self.trg_path))
            and os.path.join(data_file_path(self.trg_path))
        )

    def __getitem__(self, index):
        return {
            "id": index,
            "source": self.src_data[index],
            "target": self.trg_data[index],
        }
