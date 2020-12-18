#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from click.testing import CliRunner

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
    deploy_chart = gcloud_mock["deploy_chart"]
    auth = gcloud_mock["auth"]

    cluster = container_v1.types.Cluster
    get_operation = container_v1.ClusterManagerClient.return_value.get_operation
    get_operation_request = container_v1.types.GetOperationRequest
    nodeconfig = container_v1.types.NodeConfig
    accelerator = container_v1.types.AcceleratorConfig

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

    result = runner.invoke(
        cli_group, ["create-cluster", "gcloud"] + cmd_line, catch_exceptions=False
    )

    assert result.exit_code == 0

    cluster.assert_called_once()
    assert cluster.call_args[1]["initial_node_count"] == args[0]  # num_workers

    nodeconfig.assert_called_once()
    assert nodeconfig.call_args[1]["machine_type"] == option_dict["machine_type"]
    assert nodeconfig.call_args[1]["disk_size_gb"] == option_dict["disk_size"]
    assert nodeconfig.call_args[1]["preemptible"] == option_dict["preemptible"]

    deploy_chart.assert_called_once()
    assert deploy_chart.call_args[1]["num_workers"] == args[0] - 1
    assert deploy_chart.call_args[1]["num_gpus"] == option_dict["num_gpus"]
    assert deploy_chart.call_args[1]["num_cpus"] == option_dict["num_cpus"] - 1

    get_operation.assert_called()
    assert option_dict["zone"] in get_operation_request.call_args[0][0]["name"]
    assert project in get_operation_request.call_args[0][0]["name"]

    if option_dict["num_gpus"] > 0:
        num_gpus = option_dict["num_gpus"]
        gpu_type = option_dict["gpu_type"]
        accelerator.assert_called_once_with(
            {"accelerator_count": num_gpus, "accelerator_type": gpu_type}
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
    container_v1 = mocker.patch("mlbench_core.cli.gcloud_utils.container_v1")
    gclient = container_v1.ClusterManagerClient.return_value
    response_create = gclient.create_cluster.return_value
    response_delete = gclient.delete_cluster.return_value
    response_get_op = gclient.get_operation.return_value

    response_create.status = 0
    response_create.Status.DONE = 1
    response_create.name = "test-create-name"  # for concatenation

    response_delete.status = 0
    response_delete.Status.DONE = 1
    response_delete.name = "test-delete-name"

    response_get_op.status = 1
    response_get_op.Status.DONE = 1
    response_get_op.name = "test-get-op-name"

    return {
        "auth": gcloud_auth,
        "gclient": gclient,
        "containerv1": container_v1,
        "discovery": mocker.patch("mlbench_core.cli.gcloud_utils.discovery"),
        "http": mocker.patch("mlbench_core.cli.gcloud_utils.http"),
        "apiclient": mocker.patch("mlbench_core.cli.cli.ApiClient"),
        "k8sclient_gcloud": mocker.patch("mlbench_core.cli.gcloud_utils.kube_client"),
        "deploy_chart": mocker.patch("mlbench_core.cli.cli.deploy_chart"),
        "create_kubeconfig": mocker.patch(
            "mlbench_core.cli.cli.setup_gcloud_kube_client",
        ),
        "configparser": mocker.patch("configparser.ConfigParser"),
    }


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
    container_v1 = gcloud_mock["containerv1"]
    delete_cluster = gclient.delete_cluster
    get_operation_request = container_v1.types.GetOperationRequest
    delete_cluster_request = container_v1.types.DeleteClusterRequest
    get_operation = gclient.get_operation

    args = ["test-name"]
    opts = {"project": "test-proj", "zone": "test-zone"}

    cmd = get_gcloud_cmd_line(args, opts)

    result = runner.invoke(cli_group, ["delete-cluster", "gcloud"] + cmd)
    assert result.exit_code == 0

    delete_cluster.assert_called_once()
    get_operation.assert_called_once()

    for arg in list(opts.values()):
        assert arg in delete_cluster_request.call_args[0][0]["name"]
        assert arg in get_operation_request.call_args[0][0]["name"]

    assert args[0] in delete_cluster_request.call_args[0][0]["name"]
    assert (
        delete_cluster.return_value.name
        in get_operation_request.call_args[0][0]["name"]
    )
