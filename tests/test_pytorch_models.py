import pytest

import torch

from mlbench_core.models.pytorch.resnet import *


def test_resnet18():
    resnet = resnet18_bkj(1000)

    inp = torch.rand(2, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 2
    assert outp.shape[1] == 1000

    resnet = resnet18_bkj(500)

    inp = torch.rand(3, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 3
    assert outp.shape[1] == 500


def test_resnet20():
    resnet = get_resnet_model('resnet20', 1, 'fp32')

    inp = torch.rand(2, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 2
    assert outp.shape[1] == 10

    resnet = get_resnet_model('resnet20', 1, 'fp32')

    inp = torch.rand(3, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 3
    assert outp.shape[1] == 10


def test_resnet20v2():
    resnet = get_resnet_model('resnet20', 2, 'fp32')

    inp = torch.rand(2, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 2
    assert outp.shape[1] == 10

    resnet = get_resnet_model('resnet20', 2, 'fp32')

    inp = torch.rand(3, 3, 32, 32)

    outp = resnet(inp)

    assert outp is not None
    assert outp.shape[0] == 3
    assert outp.shape[1] == 10