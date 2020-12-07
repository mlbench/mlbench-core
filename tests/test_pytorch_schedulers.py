import math

import pytest
import torch

from mlbench_core.lr_scheduler.pytorch.lr import (
    LRLinearWarmUp,
    MultiStepLRLinearWarmUp,
    SQRTTimeDecayLR,
    TimeDecayLR,
)


def test_linear_warmup_1():
    """Tests Linear Warmup LR"""
    init_lr = 0
    scaled_lr = 10
    warmup_duration = 5
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=scaled_lr)

    scheduler = LRLinearWarmUp(
        optimizer=opt,
        init_lr=init_lr,
        scaled_lr=scaled_lr,
        warmup_duration=warmup_duration,
    )

    lrs = [0, 2, 4, 6, 8, 10, 10]
    for i in range(7):
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == lrs[i]
        scheduler.step()


def test_linear_warmup_2():
    """Tests Linear Warmup LR"""
    init_lr = 10
    scaled_lr = 10
    warmup_duration = 5
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=scaled_lr)

    scheduler = LRLinearWarmUp(
        optimizer=opt,
        init_lr=init_lr,
        scaled_lr=scaled_lr,
        warmup_duration=warmup_duration,
    )

    for i in range(7):
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == scaled_lr
        scheduler.step()


def test_multi_step_lr():
    """Tests Multi step LR without warmup"""
    scaled_lr = 10
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=scaled_lr)

    scheduler = MultiStepLRLinearWarmUp(
        optimizer=opt, scaled_lr=scaled_lr, gamma=0.5, milestones=[2, 3]
    )

    lrs = [10, 10, 5, 2.5]
    for i in range(4):
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == lrs[i]
        scheduler.step()


def test_multi_step_lin_warmup():
    """Tests Multistep LR with linear warmup"""
    init_lr = 0
    scaled_lr = 10
    warmup_duration = 5
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=scaled_lr)

    scheduler = MultiStepLRLinearWarmUp(
        optimizer=opt,
        warmup_init_lr=init_lr,
        scaled_lr=scaled_lr,
        warmup_duration=warmup_duration,
        gamma=0.5,
        milestones=[7, 8],
    )

    lrs = [0, 2, 4, 6, 8, 10, 10, 5, 2.5]
    for i in range(9):
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == lrs[i]
        scheduler.step()


def test_time_decay_lr():
    """Tests Time Decay LR"""
    lr = 10
    beta = 1
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=lr)

    scheduler = TimeDecayLR(optimizer=opt, beta=beta)

    for i in range(10):
        true_lr = lr / (i + beta)
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == pytest.approx(true_lr)
        scheduler.step()


def test_sqrt_time_decay_lr():
    """Tests SQRT Time Decay LR"""
    lr = 10
    params = torch.nn.Parameter(torch.Tensor([1, 2, 3]))
    opt = torch.optim.SGD([params], lr=lr)

    scheduler = SQRTTimeDecayLR(optimizer=opt)

    for i in range(10):
        true_lr = lr / math.sqrt(max(1, i))
        last_lr = scheduler.get_last_lr()[0]
        assert last_lr == pytest.approx(true_lr)
        scheduler.step()
