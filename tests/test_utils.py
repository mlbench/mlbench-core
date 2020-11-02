"""Tests for `mlbench_core.utils` package."""

import datetime

from freezegun import freeze_time

from mlbench_core.evaluation.goals import task1_time_to_accuracy_light_goal
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.utils import LogMetrics, Tracker


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


def _do_batch(tracker, frozen):
    tracker.batch_start()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_load()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_init()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_fwd_pass()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_comp_loss()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_backprop()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_agg()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_opt_step()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.record_batch_comp_metrics()
    frozen.tick(delta=datetime.timedelta(seconds=0.5))
    tracker.batch_end()


def test_tracker_goal_times(mocker):
    patched = mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    metric = TopKAccuracy(1)
    tracker = Tracker([metric], 1, 0, task1_time_to_accuracy_light_goal())

    tracker.start()

    assert tracker.start_time is not None

    tracker.train()

    with freeze_time(datetime.datetime.now()) as frozen:
        _do_batch(tracker, frozen)

        assert abs(tracker.get_total_preprocess_time() - 0.5) < 0.01
        assert abs(tracker.get_total_communication_time() - 0.5) < 0.01
        assert abs(tracker.get_total_compute_time() - 2.0) < 0.01
        assert abs(tracker.get_total_metrics_time() - 0.5) < 0.01

        _do_batch(tracker, frozen)

        assert abs(tracker.get_total_preprocess_time() - 1.0) < 0.01
        assert abs(tracker.get_total_communication_time() - 1.0) < 0.01
        assert abs(tracker.get_total_compute_time() - 4.0) < 0.01
        assert abs(tracker.get_total_metrics_time() - 1.0) < 0.01

        tracker.validation()
        tracker.record_stat("global_Prec@1", 70, log_to_api=True)

        assert tracker.goal_reached
        assert any(filter(lambda c: c[1][3] == "TaskResult", patched.method_calls))


def test_LogMetrics(mocker):
    mocker.patch("mlbench_core.api.ApiClient")

    LogMetrics.log("1", 1, 1, "loss", 123)

    mocker.patch.dict("os.environ", {"MLBENCH_IN_DOCKER": "True"})

    LogMetrics.log("1", 1, 1, "loss", 123)
