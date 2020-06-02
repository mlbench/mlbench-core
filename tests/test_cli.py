#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from click.testing import CliRunner
from google.auth.exceptions import DefaultCredentialsError

from mlbench_core.cli import cli_group

########################## GLOBALS #################################

runner = CliRunner()

CREATE_GCLOUD_DEFAULTS = {
    "machine_type": "n1-standard-4",
    "disk_size": 50,
    "num_cpus": 4,
    "num_gpus": 0,
    "gpu_type": "nvidia-tesla-k80",
    "zone": "europe-west1-b",
    "preemptible": False,
}


########################## HELPERS #################################


def get_gcloud_cmd_line(args, option_dict=None):
    """Constructs command line for gcloud commands

    Returns command line as a list, e.g. ["3", "myrelease", "-g", "1", "--gpu-type", "nvidia-tesla-k80"].

    Args: 
        args (list): command-line arguments (e.g. [3, "myrelease"])
        option_dict (dict): command-line options, as in CREATE_GCLOUD_DEFAULTS

    Returns:
        list: command line constructed from inputs
    """

    flags = {
        "machine_type": "-t",
        "disk_size": "-d",
        "num_cpus": "-c",
        "num_gpus": "-g",
        "gpu_type": "--gpu-type",
        "zone": "-z",
        "preemptible": "-e",
        "project": "-p",
        "custom_value": "-v",
    }

    args = list(map(str, args))

    def get_array(option):
        # e.g. option='num_gpus' --> ['-g', 0]
        if options[option] == True:  # check if flag is boolean
            return [flags[option]]
        elif options[option] == False:
            return []
        else:
            return [flags[option], options[option]]

    if option_dict is not None:
        options = option_dict

        cmd_line_options = [
            str(elem) for option in options for elem in get_array(option)
        ]

        return cmd_line_options + args

    return args


def status_mock_helper(mocker, gcloud_mock, run_name):
    """Generates patched objects for "mlbench status" command.

    This is a helper function used in fixtures that test "mlbench status".

    Args:
        mocker (pytest.fixture)
        gcloud_mock (pytest.fixture)
        run_name (string)

    Returns:
        tuple: patched ApiClient, run ID and run name
    """
    setup = mocker.patch("mlbench_core.cli.cli.setup_client_from_config")
    setup.return_value = True

    client = gcloud_mock["apiclient"]
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


def create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict=None):
    """Tests creation of gcloud cluster

    Tests that the correct arguments are passed to Gcloud, Tiller, etc... based 
    on the input arguments and options. 

    Args:
        args (list): command-line arguments
        option_dict (dict): command-line options, as in CREATE_GCLOUD_DEFAULTS. Any 
                            missing options are substituted with the defaults.
    """

    container_v1 = gcloud_mock["containerv1"]
    tiller = gcloud_mock["tiller"]
    auth = gcloud_mock["auth"]

    cluster = container_v1.types.Cluster
    get_operation = container_v1.ClusterManagerClient.return_value.get_operation
    nodeconfig = container_v1.types.NodeConfig
    accelerator = container_v1.types.AcceleratorConfig

    install_release = tiller.return_value.install_release

    if option_dict is not None and "project" in option_dict:
        project = option_dict["project"]
    else:
        _, project = auth()

    cmd_line = get_gcloud_cmd_line(args, option_dict)

    if option_dict is None:
        option_dict = CREATE_GCLOUD_DEFAULTS
    else:
        for k in CREATE_GCLOUD_DEFAULTS:  # add missing defaults
            if k not in option_dict:
                option_dict[k] = CREATE_GCLOUD_DEFAULTS[k]

    runner.invoke(cli_group, ["create-cluster", "gcloud"] + cmd_line)

    cluster.assert_called_once()
    assert cluster.call_args[1]["initial_node_count"] == args[0]  # num_workers

    nodeconfig.assert_called_once()
    assert nodeconfig.call_args[1]["machine_type"] == option_dict["machine_type"]
    assert nodeconfig.call_args[1]["disk_size_gb"] == option_dict["disk_size"]
    assert nodeconfig.call_args[1]["preemptible"] == option_dict["preemptible"]

    tiller.assert_called_once()
    install_release.assert_called_once()

    assert install_release.call_args[1]["values"]["limits"]["workers"] == args[0] - 1
    assert (
        install_release.call_args[1]["values"]["limits"]["gpu"]
        == option_dict["num_gpus"]
    )
    assert (
        install_release.call_args[1]["values"]["limits"]["cpu"]
        == option_dict["num_cpus"] - 1
    )

    get_operation.assert_called()
    assert option_dict["zone"] in get_operation.call_args[1]["name"]
    assert project in get_operation.call_args[1]["name"]

    if option_dict["num_gpus"] > 0:
        num_gpus = option_dict["num_gpus"]
        gpu_type = option_dict["gpu_type"]
        accelerator.assert_called_once_with(
            accelerator_count=num_gpus, accelerator_type=gpu_type
        )


########################## FIXTURES #################################


@pytest.fixture
def gcloud_auth(mocker):
    """Patches google.auth.default to bypass authentication"""

    m = mocker.patch("google.auth.default")
    m.return_value = (m.credentials, "test_project")

    return m


@pytest.fixture
def gcloud_mock(mocker, gcloud_auth):
    """Patches all gcloud objects and makes loops run once.

    Loops like 

        while response.status < response.DONE:
            #do something 
    
    are executed once.

    Returns:
        dict: dictionary of patched objects
    """

    # One response loop
    container_v1 = mocker.patch("google.cloud.container_v1")
    gclient = container_v1.ClusterManagerClient.return_value
    response_create = gclient.create_cluster.return_value
    response_delete = gclient.delete_cluster.return_value
    response_get_op = gclient.get_operation.return_value

    response_create.status = 0
    response_create.DONE = 1
    response_create.name = "test-create-name"  # for concatenation

    response_delete.status = 0
    response_delete.DONE = 1
    response_delete.name = "test-delete-name"

    response_get_op.status = 1
    response_get_op.DONE = 1
    response_get_op.name = "test-get-op-name"

    return {
        "auth": gcloud_auth,
        "gclient": gclient,
        "containerv1": container_v1,
        "discovery": mocker.patch("googleapiclient.discovery"),
        "http": mocker.patch("googleapiclient.http"),
        "tiller": mocker.patch("mlbench_core.cli.cli.Tiller"),
        "chartbuilder": mocker.patch("mlbench_core.cli.cli.ChartBuilder"),
        "apiclient": mocker.patch("mlbench_core.cli.cli.ApiClient"),
        "k8sclient": mocker.patch("mlbench_core.cli.cli.client"),
        "sleep": mocker.patch("time.sleep"),
        "popen": mocker.patch("subprocess.Popen"),
        "usrdatadir": mocker.patch("appdirs.user_data_dir"),
        "configparser": mocker.patch("configparser.ConfigParser"),
        "makedirs": mocker.patch("os.makedirs"),
        "pathexists": mocker.patch("os.path.exists"),
        "pathjoin": mocker.patch("os.path.join"),
    }


@pytest.fixture
def status_mock(mocker, gcloud_mock):
    """Patches status function when a run is found"""
    return status_mock_helper(mocker, gcloud_mock, "test-run-name")


@pytest.fixture
def status_mock_no_run(mocker, gcloud_mock):
    """Patches status function when no runs are found"""
    return status_mock_helper(mocker, gcloud_mock, None)


########################## TESTS #################################


def test_invalid_num_workers(mocker, gcloud_auth):
    """Tests create-cluster with n_workers < 2"""

    res = runner.invoke(cli_group, ["create-cluster", "gcloud", "1", "test"])
    assert type(res.exception) == AssertionError


@pytest.mark.parametrize(
    "args,option_dict",
    [
        ([3, "test"], None),  # default
        (  # non-default
            [3, "test"],
            {
                "machine_type": "my-custom-type",
                "disk_size": 20,
                "num_cpus": 5,
                "num_gpus": 0,
                "gpu_type": "my-nvidia-tesla-p100",
                "zone": "my-europe-west1-b",
                "preemptible": True,
                "project": "myproj",
            },
        ),
        ([10, "test"], {"num_gpus": 3}),  # with gpu
    ],
)
def test_create_gcloud_cluster(mocker, gcloud_mock, args, option_dict):
    """Tests creating a gcloud cluster.

    Uses default values, non-default values and GPU (see decorator).
    """
    create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict)


def test_delete_gcloud_cluster(gcloud_mock):
    """Tests deleting a gcloud cluster"""

    gclient = gcloud_mock["gclient"]
    delete_cluster = gclient.delete_cluster
    get_operation = gclient.get_operation

    args = ["test-name"]
    opts = {"project": "test-proj", "zone": "test-zone"}

    cmd = get_gcloud_cmd_line(args, opts)

    runner.invoke(cli_group, ["delete-cluster", "gcloud"] + cmd)

    delete_cluster.assert_called_once()
    get_operation.assert_called_once()

    for arg in list(opts.values()):
        assert arg in delete_cluster.call_args[1]["name"]
        assert arg in get_operation.call_args[1]["name"]

    assert args[0] in delete_cluster.call_args[1]["name"]
    assert delete_cluster.return_value.name in get_operation.call_args[1]["name"]


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

    runner.invoke(cli_group, cmd)

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

    runner.invoke(cli_group, cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not get_run_metrics.called


def test_delete_no_run(status_mock_no_run):
    """Tests mlbench delete command when no run is found"""

    client, rid, _ = status_mock_no_run
    delete_run_client = client.return_value.delete_run

    url = "my/test/url"
    name = "my-test-name"
    cmd = ["delete", name, "-u", url]

    runner.invoke(cli_group, cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not delete_run_client.called


def test_delete_run(status_mock):
    """Tests mlbench delete command when a run is found"""

    client, rid, name = status_mock
    url = "my/test/url"

    delete_run_client = client.return_value.delete_run

    cmd = ["delete", name, "-u", url]

    runner.invoke(cli_group, cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    delete_run_client.assert_called_once_with(rid)
