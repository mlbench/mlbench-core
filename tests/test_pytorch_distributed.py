"""Tests for `mlbench_core.utils.pytorch.distributed`"""
import torch

from mlbench_core.utils.pytorch.distributed import pack_tensors, unpack_tensors


def test_pack_tensors():
    tensors = [torch.rand(2, 2), torch.rand(2, 2)]

    flattened = [y for x in tensors for y in x.view(-1)]

    vec, indices, sizes = pack_tensors(tensors)

    assert vec.tolist() == flattened
    assert indices == [0, 4, 8]
    assert sizes == [(2, 2), (2, 2)]


def test_unpack_tensors():
    tensors = [torch.rand(2, 2), torch.rand(2, 2)]
    vec, indices, sizes = pack_tensors(tensors)

    unpacked = unpack_tensors(vec, indices, sizes)

    assert all((x == y).all() for x, y in zip(tensors, unpacked))
