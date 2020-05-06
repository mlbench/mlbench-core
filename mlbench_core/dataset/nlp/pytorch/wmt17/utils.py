import struct

import numpy as np
import torch
from torch.utils.data import Dataset

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


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


def collate_tokens(
    values,
    pad_idx,
    eos_idx,
    left_pad,
    move_eos_to_beginning=False,
    n_seq_per_batch_multiple=8,
    seq_len_multiple=1,
):
    """ Convert a list of 1d tensors into a padded 2d tensor.

    Args:
        values: Python list where each element is a PyT 1d tensor
        pad_idx: The index into the translation dictionary for the pad token (typically refer to 'dict.pad()')
        eos_idx: The index into the translation dictionary for the eos token (typically refer to 'dict.eos()')
        left_pad: Bool, left- or right-padding (true: left, false: right)
        move_eos_to_beginning: Reverse order of sequence of tokens (true: reverse, false:leave in original order)
        n_seq_per_batch_multiple: The number of sequences per batch to round down to
        seq_len_multiple: The number of tokens per sequence to round up to
    """
    size_of_seq_dim = max(v.size(0) for v in values)  # Unpadded size
    n_seq_in_batch = len(values)

    if n_seq_per_batch_multiple % seq_len_multiple == 0:
        n_seq_multiple = n_seq_per_batch_multiple / seq_len_multiple
    else:
        n_seq_multiple = n_seq_per_batch_multiple

    if n_seq_in_batch < n_seq_multiple or n_seq_in_batch % n_seq_multiple > 0:
        seq_len_multiple = n_seq_per_batch_multiple

    size_of_seq_dim = (
        (size_of_seq_dim + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple
    )  # Padded seq len, rounded up to next multiple

    padded_2d_tensor = values[0].new(len(values), size_of_seq_dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()

        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if left_pad:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][size_of_seq_dim - len(val) :])
    else:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][: len(val)])

    return padded_2d_tensor
