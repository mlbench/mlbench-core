#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from click.testing import CliRunner

from mlbench_core.cli import cli_group

########################## GLOBALS #################################

runner = CliRunner()

########################## HELPERS #################################


def status_mock_helper(mocker, run_name):
    """Generates patched objects for "mlbench status" command.

    This is a helper function used in fixtures that test "mlbench status".

    Args:
        mocker (pytest.fixture)
        run_name (string)

    Returns:
        tuple: patched ApiClient, run ID and run name
    """
    setup = mocker.patch("mlbench_core.cli.cli.setup_client_from_config")
    client = mocker.patch("mlbench_core.cli.cli.ApiClient")
    setup.return_value = True

    rundict = {
        "name": "none" if run_name is None else run_name,
        "job_id": "",
        "job_metadata": "",
        "id": "my-id",
    }

    client.return_value.get_runs.return_value.result.return_value.json.return_value = [
        rundict
    ]

    return client, rundict["id"], rundict["name"]


########################## FIXTURES #################################


@pytest.fixture
def status_mock(mocker):
    """Patches status function when a run is found"""
    return status_mock_helper(mocker, "test-run-name")


@pytest.fixture
def status_mock_no_run(mocker):
    """Patches status function when no runs are found"""
    return status_mock_helper(mocker, None)


########################## TESTS #################################


def test_status(status_mock):
    """Tests "mlbench status" command when a run is found"""

    client, rid, name = status_mock
    url = "my/test/url"

    # skip reporting
    client.return_value.get_run_metrics.return_value.result.return_value.status_code = (
        301
    )

    get_run_metrics = client.return_value.get_run_metrics

    cmd = ["status", name, "-u", url]

    result = runner.invoke(cli_group, cmd)
    assert result.exit_code == 0

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    get_run_metrics.assert_called()
    assert get_run_metrics.mock_calls[0][1][0] == rid
    assert get_run_metrics.mock_calls[1][1][0] == rid


def test_status_no_run(status_mock_no_run):
    """Tests "mlbench status" command when no run is found."""

    client, rid, _ = status_mock_no_run
    get_run_metrics = client.return_value.get_run_metrics

    url = "my/test/url"
    name = "my-test-name"
    cmd = ["status", name, "-u", url]

    result = runner.invoke(cli_group, cmd)
    assert result.exit_code == 0

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not get_run_metrics.called


def test_charts(status_mock, tmpdir):
    """Tests "mlbench charts" command on 3 finished runs"""
    folder = tmpdir.mkdir("charts")

    client, rid, name = status_mock

    # skip reporting
    client.return_value.get_run_metrics.return_value.result.return_value.status_code = (
        301
    )

    client.return_value.get_runs.return_value.result.return_value.json.return_value = [
        {"id": 1, "name": "test-0", "num_workers": 1, "state": "finished"},
        {"id": 2, "name": "test-1", "num_workers": 2, "state": "finished"},
        {"id": 3, "name": "test-2", "num_workers": 4, "state": "finished"},
    ]

    client.return_value.get_run_metrics.return_value.result.return_value.json.return_value = {
        "global_cum_agg @ 0": [{"value": "1.0"}],
        "global_cum_backprop @ 0": [{"value": "1.0"}],
        "global_cum_batch_load @ 0": [{"value": "1.0"}],
        "global_cum_comp_loss @ 0": [{"value": "1.0"}],
        "global_cum_comp_metrics @ 0": [{"value": "1.0"}],
        "global_cum_fwd_pass @ 0": [{"value": "1.0"}],
        "global_cum_opt_step @ 0": [{"value": "1.0"}],
    }

    cmd = ["charts", str(folder)]

    runner.invoke(cli_group, cmd, input="0 1 2")

    assert folder.join("total_time.png").exists()
    assert folder.join("speedup.png").exists()
    assert folder.join("time_for_all_phases.png").exists()


def test_delete_no_run(status_mock_no_run):
    """Tests mlbench delete command when no run is found"""

    client, rid, _ = status_mock_no_run
    delete_run_client = client.return_value.delete_run

    url = "my/test/url"
    name = "my-test-name"
    cmd = ["delete", name, "-u", url]

    result = runner.invoke(cli_group, cmd)
    assert result.exit_code == 0

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not delete_run_client.called


def test_delete_run(status_mock):
    """Tests mlbench delete command when a run is found"""

    client, rid, name = status_mock
    url = "my/test/url"

    delete_run_client = client.return_value.delete_run

    cmd = ["delete", name, "-u", url]

    result = runner.invoke(cli_group, cmd)
    assert result.exit_code == 0

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    delete_run_client.assert_called_once_with(rid)
