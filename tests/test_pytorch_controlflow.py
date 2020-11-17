#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.controlflow.pytorch` package."""
import itertools
import random

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mlbench_core.controlflow.pytorch.controlflow import (
    compute_train_batch_metrics,
    record_train_batch_stats,
    validation_round,
)
from mlbench_core.controlflow.pytorch.helpers import (
    convert_dtype,
    iterate_dataloader,
    maybe_range,
)
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.lr_scheduler.pytorch import multistep_learning_rates_with_warmup


@pytest.fixture
def model():
    return nn.Linear(1, 2)


@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def loss_function():
    return nn.CrossEntropyLoss()


@pytest.fixture
def metrics():
    return [TopKAccuracy(topk=1)]


@pytest.fixture
def scheduler(optimizer):
    return multistep_learning_rates_with_warmup(
        optimizer,
        1,
        0.1,
        0.1,
        [5, 10],
        warmup_duration=2,
        warmup_linear_scaling=False,
        warmup_lr=0.2,
    )


def _create_random_sets():
    train_set = [random.random() for _ in range(100)]
    train_set = [
        (
            torch.FloatTensor([n * 50 - 25]),
            1 if (n > 0.5) != (random.random() < 0.1) else 0,
        )
        for n in train_set
    ]

    test_set = [random.random() for _ in range(10)]
    test_set = [
        (
            torch.FloatTensor([n * 50 - 25]),
            1 if (n > 0.5) != (random.random() < 0.1) else 0,
        )
        for n in test_set
    ]

    return train_set, test_set


def test_compute_train_metrics(
    mocker, model, optimizer, loss_function, metrics, scheduler
):
    mocker.patch("mlbench_core.utils.pytorch.distributed.dist")
    mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    batch_size = 2

    train_set, test_set = _create_random_sets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        metric_values = compute_train_batch_metrics(loss, output, target, metrics)
        metric_values = [(k, v) for k, v in metric_values.items() if k.name == "Prec@1"]
        assert len(metric_values) == 1

        metric, value = metric_values[0]

        assert value == metrics[0](loss, output, target)


def test_validation_round(mocker, model, optimizer, loss_function, metrics, scheduler):
    mocker.patch("mlbench_core.utils.pytorch.distributed.dist")
    mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    batch_size = 2

    train_set, test_set = _create_random_sets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

    metric_values, loss_values = validation_round(
        test_loader,
        model=model,
        loss_function=loss_function,
        metrics=metrics,
        dtype="fp32",
    )

    assert "Prec@1" in [m.name for m in metric_values]


def test_maybe_range():
    r = maybe_range(10)

    assert len(r) == 10
    assert r == range(10)

    r = maybe_range(None)

    assert isinstance(r, itertools.count)
    assert next(r) == 0
    assert next(r) == 1


def test_convert_dtype():
    t = torch.IntTensor([0])

    tt = convert_dtype("fp32", t)

    assert tt.dtype == torch.float32

    tt2 = convert_dtype("fp64", t)

    assert tt2.dtype == torch.float64

    with pytest.raises(NotImplementedError):
        tt3 = convert_dtype("int", t)


def test_iterate_dataloader(mocker):
    dataloader = [
        (torch.IntTensor([0]), torch.IntTensor([1])),
        (torch.IntTensor([2]), torch.IntTensor([3])),
    ]

    it = iterate_dataloader(
        dataloader, "fp32", max_batch_per_epoch=2, transform_target_type=True
    )

    first = next(it)

    assert first[0].dtype == torch.float32
    assert first[1].dtype == torch.float32
    assert first[0].data.item() == 0.0
    assert first[1].item() == 1.0

    second = next(it)

    assert second[0].dtype == torch.float32
    assert second[1].dtype == torch.float32
    assert second[0].data.item() == 2.0
    assert second[1].item() == 3.0
