#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

from mlbench_core.utils import LogMetrics
from mlbench_core.utils.pytorch.helpers import config_path, config_pytorch


def test_config_pytorch(mocker):
    mocker.patch("torch.distributed.get_rank", return_value=1)
    mocker.patch("torch.distributed.get_world_size", return_value=1)
    mocker.patch("mlbench_core.utils.pytorch.helpers.FCGraph")

    rank, world_size, graph = config_pytorch(
        use_cuda=False, seed=42, cudnn_deterministic=True
    )

    assert rank == 1
    assert world_size == 1
    assert graph is not None


def test_LogMetrics(mocker):
    mocker.patch("mlbench_core.api.ApiClient")

    LogMetrics.log("1", 1, 1, "loss", 123)

    mocker.patch.dict("os.environ", {"MLBENCH_IN_DOCKER": "True"})

    LogMetrics.log("1", 1, 1, "loss", 123)


def test_config_path(mocker):
    sh = mocker.patch("shutil.rmtree")
    osmk = mocker.patch("os.makedirs")

    config_path("/tmp/checkpoints", delete_existing_ckpts=False)

    osmk.assert_called_once_with("/tmp/checkpoints", exist_ok=True)
    assert sh.call_count == 0

    config_path("/tmp/checkpoints", delete_existing_ckpts=True)

    assert sh.call_count == 1
    assert osmk.call_count == 2
