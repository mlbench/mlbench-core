#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

import pytest
import time
import itertools
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch.helpers import *
from mlbench_core.utils import Tracker


def test_timeit():
    timeit = Timeit()

    cumu = timeit.cumu

    time.sleep(0.1)
    timeit.pause()
    new_cumu = timeit.cumu

    assert new_cumu - cumu - 0.1 < 0.01

    time.sleep(0.1)
    newer_cumu = timeit.cumu

    assert new_cumu == newer_cumu

    timeit.resume()
    time.sleep(0.1)
    timeit.pause()

    last_cumu = timeit.cumu

    assert last_cumu - newer_cumu - 0.1 < 0.01
    assert last_cumu - cumu - 0.3 < 0.01


def test_maybe_range():
    r = maybe_range(10)

    assert len(r) == 10
    assert r == range(10)

    r = maybe_range(None)

    assert isinstance(r, itertools.count)
    assert next(r) == 0
    assert next(r) == 1


def test_update_best_runtime_metric(mocker):
    tracker = Tracker()
    tracker.records = {}
    tracker.current_epoch = 1
    #tracker = mocker.patch('mlbench_core.utils.pytorch.helpers.Tracker')

    is_best, best_metric_name = update_best_runtime_metric(tracker, 10.0, 'prec')

    assert is_best == True
    assert best_metric_name == "best_prec"

    is_best, best_metric_name = update_best_runtime_metric(tracker, 11.0, 'prec')

    assert is_best == True
    assert best_metric_name == "best_prec"

    is_best, best_metric_name = update_best_runtime_metric(tracker, 9.0, 'prec')

    assert is_best == False
    assert best_metric_name == "best_prec"


def test_convert_dtype():
    t = torch.IntTensor([0])

    tt = convert_dtype('fp32', t)

    assert tt.dtype == torch.float32

    tt2 = convert_dtype('fp64', t)

    assert tt2.dtype == torch.float64

    with pytest.raises(NotImplementedError):
        tt3 = convert_dtype('int', t)


def test_config_pytorch(mocker):
    mocker.patch('torch.distributed.get_rank', return_value=1)
    mocker.patch('torch.distributed.get_world_size', return_value=1)
    mocker.patch('mlbench_core.utils.pytorch.helpers.FCGraph')

    rank, world_size, graph = config_pytorch(use_cuda=False, seed=42, cudnn_deterministic=True)

    assert rank == 1
    assert world_size == 1
    assert graph is not None


def test_log_metrics(mocker):
    mocker.patch('mlbench_core.utils.pytorch.helpers.ApiClient')

    log_metrics("1", 1, 1, "loss", 123)

    mocker.patch.dict('os.environ', {'MLBENCH_IN_DOCKER': 'True'})

    log_metrics("1", 1, 1, "loss", 123)


def test_config_path(mocker):
    sh = mocker.patch('shutil.rmtree')
    osmk = mocker.patch('os.makedirs')

    config_path('/tmp/checkpoints', resume=False)

    osmk.assert_called_once_with('/tmp/checkpoints', exist_ok=True)
    assert sh.call_count == 0

    config_path('/tmp/checkpoints', resume=True)

    assert sh.call_count == 1
    assert osmk.call_count == 2


def test_iterate_dataloader(mocker):
    dataloader = [
        (torch.IntTensor([0]), torch.IntTensor([1])),
        (torch.IntTensor([2]), torch.IntTensor([3]))]

    it = iterate_dataloader(dataloader, 'fp32', max_batch_per_epoch=2, transform_target_type=True)

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