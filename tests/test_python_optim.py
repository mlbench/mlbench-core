"""Tests for `mlbench_core.optim.pytorch` package."""
import pytest
import torch
import torch.distributed as dist
from torch.nn.modules import Linear, MSELoss
from torch.optim import SGD

from mlbench_core.optim.pytorch.centralized import (
    CentralizedAdam,
    CentralizedSGD,
    CentralizedSparsifiedSGD,
    CustomCentralizedOptimizer,
    GenericCentralizedOptimizer,
    PowerSGD,
)
from mlbench_core.optim.pytorch.optim import SignSGD, SparsifiedSGD


def test_SparsifiedSGD():
    model = Linear(2, 1)
    opt = SparsifiedSGD(model.parameters(), lr=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()


def test_SignSGD():
    model = Linear(2, 1)
    opt = SignSGD(model.parameters(), lr=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()


def test_GenericCentralizedOptimizer():
    model = Linear(2, 1)
    opt = SGD(model.parameters(), lr=1)
    c_opt = GenericCentralizedOptimizer(world_size=1, model=model)
    c_opt.optimizer = opt

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    c_opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()


def test_CentralizedSparsifiedSGD(mocker):
    dist.init_process_group(
        "gloo", world_size=1, init_method="file:///tmp/somefile", rank=0
    )
    model = Linear(2, 1, bias=False)
    opt = CentralizedSparsifiedSGD(model.parameters(), lr=10, sparse_grad_size=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1, 2], [2, 3]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()
    dist.destroy_process_group()


def test_CentralizedSGD():
    model = Linear(2, 1)
    opt = CentralizedSGD(world_size=1, model=model, lr=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()


def test_CentralizedAdam():
    model = Linear(2, 1)
    opt = CentralizedAdam(world_size=1, model=model, lr=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()


def test_PowerSGD():
    dist.init_process_group(
        "gloo", world_size=1, init_method="file:///tmp/somefile", rank=0
    )
    model = Linear(2, 1)
    opt = PowerSGD(world_size=1, model=model, lr=1)

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    opt.step()
    dist.destroy_process_group()


def test_CustomCentralizedOptimizer():

    model = Linear(2, 1)
    opt = SGD(params=model.parameters(), lr=1)
    c_opt = CustomCentralizedOptimizer(
        world_size=1, model=model, optimizer=opt, average_world=True
    )

    input_data = torch.Tensor([[1, 2], [3, 4]])
    target = torch.Tensor([[1], [2]])

    c_opt.zero_grad()
    output = model(input_data)
    loss = MSELoss()(output, target)
    loss.backward()
    c_opt.step()
