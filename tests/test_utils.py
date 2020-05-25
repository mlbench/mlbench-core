#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.utils.pytorch.helpers` package."""

import datetime

import torch
from freezegun import freeze_time

from mlbench_core.evaluation.goals import task1_time_to_accuracy_light_goal
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch.utils import orthogonalize


def test_tracker():
    tracker = Tracker([TopKAccuracy(5)], 1, 0)

    assert tracker is not None


def test_tracker_goal(mocker):
    patched = mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    metric = TopKAccuracy(1)
    tracker = Tracker([metric], 1, 0, task1_time_to_accuracy_light_goal())

    tracker.start()

    assert tracker.start_time is not None

    tracker.train()

    tracker.record_stat("global_Prec@1", 69, log_to_api=True)
    tracker.batch_end()

    assert not tracker.goal_reached

    tracker.record_stat("global_Prec@1", 70, log_to_api=True)
    tracker.batch_end()

    assert not tracker.goal_reached

    tracker.validation()

    tracker.record_stat("global_Prec@1", 69, log_to_api=True)
    tracker.batch_end()

    assert not tracker.goal_reached

    tracker.record_stat("global_Prec@1", 70, log_to_api=True)

    assert tracker.goal_reached


def test_tracker_goal_times(mocker):
    patched = mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    metric = TopKAccuracy(1)
    tracker = Tracker([metric], 1, 0, task1_time_to_accuracy_light_goal())

    tracker.start()

    assert tracker.start_time is not None

    tracker.train()

    with freeze_time(datetime.datetime.now()) as frozen:
        tracker.batch_start()
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("init")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("fwd_pass")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("comp_loss")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("backprop")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("opt_step")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.batch_end()

        assert abs(tracker.get_total_communication_time() - 0.5) < 0.01
        assert abs(tracker.get_total_compute_time() - 1.5) < 0.01

        tracker.batch_start()
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("init")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("fwd_pass")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("comp_loss")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("backprop")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.record_batch_step("opt_step")
        frozen.tick(delta=datetime.timedelta(seconds=0.5))
        tracker.batch_end()

        assert abs(tracker.get_total_communication_time() - 1.0) < 0.01
        assert abs(tracker.get_total_compute_time() - 3.0) < 0.01

        tracker.validation()
        tracker.record_stat("global_Prec@1", 70, log_to_api=True)

        assert tracker.goal_reached
        assert any(filter(lambda c: c[1][3] == "TaskResult", patched.method_calls))


def test_orthogonalize():
    m = torch.rand(2, 2)
    identity = torch.eye(2)

    orthogonalize(m)

    # check if m'*m = I
    assert torch.allclose(torch.matmul(m.t(), m), identity, atol=1e-04)
