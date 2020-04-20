#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from mlbench_core.cli import cli_group

create_gcloud = cli_group.commands['create-cluster'].commands['gcloud'].main


CREATE_GCLOUD_DEFAULTS = {
    'machine_type'  : 'n1-standard-4',
    'disk_size'     : 50,
    'num_cpus'      : 1,
    'num_gpus'      : 0,
    'gpu_type'      : "nvidia-tesla-p100",
    'zone'          : "europe-west1-b",
    'preemptible'   : False,
}

@pytest.fixture
def gcloud_auth(mocker):
    m = mocker.patch("google.auth.default")
    m.return_value = (m.credentials, "test_project")

    return m

@pytest.fixture
def sys_mock(mocker):
    #Avoid error on SystemExit
    mocker.patch("mlbench_core.cli.cli.sys.exit")

@pytest.fixture
def gcloud_mock(mocker, gcloud_auth, sys_mock):

    # One response loop
    container_v1 = mocker.patch("google.cloud.container_v1")
    gclient = container_v1.ClusterManagerClient.return_value
    response_1 = gclient.create_cluster.return_value
    response_2 = gclient.get_operation.return_value

    response_1.status = 0
    response_1.DONE = 1
    response_1.name = 'test' # for concatenation

    response_2.status = 1
    response_2.DONE = 1
    response_2.name = 'test'


    return {
        'auth'              : gcloud_auth,
        'containerv1'       : container_v1,
        'discovery'         : mocker.patch("googleapiclient.discovery"),
        'http'              : mocker.patch("googleapiclient.http"),
        'tiller'            : mocker.patch("mlbench_core.cli.cli.Tiller"),
        'chartbuilder'      : mocker.patch("mlbench_core.cli.cli.ChartBuilder"),
        'apiclient'         : mocker.patch("mlbench_core.cli.cli.ApiClient"),
        'k8sclient'         : mocker.patch("mlbench_core.cli.cli.client"),
        'sleep'             : mocker.patch("time.sleep"),
        'popen'             : mocker.patch("subprocess.Popen"),
        'usrdatadir'        : mocker.patch("appdirs.user_data_dir"),
        'configparser'      : mocker.patch("configparser.ConfigParser"),
        'makedirs'          : mocker.patch("os.makedirs"),
        'pathexists'        : mocker.patch("os.path.exists"),
        'pathjoin'          : mocker.patch("os.path.join"),
    }
    

def test_invalid_num_workers(mocker, gcloud_auth):
    
    with pytest.raises(AssertionError):
        create_gcloud(['1','test'])

#def test_no_credentials(mocker):


def test_create_cluster_default(mocker, gcloud_mock):
    
    container_v1 = gcloud_mock["containerv1"]
    tiller = gcloud_mock["tiller"]
    auth = gcloud_mock["auth"]

    cluster = container_v1.types.Cluster
    get_operation = container_v1.ClusterManagerClient.return_value.get_operation
    nodeconfig = container_v1.types.NodeConfig
    install_release = tiller.return_value.install_release
    _, project = auth()


    num_workers = 3
    release = 'test'
    args = [str(num_workers), release]

    create_gcloud(args)

    cluster.assert_called_once()
    assert cluster.call_args[1]["initial_node_count"] == num_workers
    
    nodeconfig.assert_called_once()
    assert nodeconfig.call_args[1]["machine_type"] == CREATE_GCLOUD_DEFAULTS['machine_type']
    assert nodeconfig.call_args[1]["disk_size_gb"] == CREATE_GCLOUD_DEFAULTS['disk_size']
    assert nodeconfig.call_args[1]["preemptible"] == CREATE_GCLOUD_DEFAULTS['preemptible']

    tiller.assert_called_once()
    install_release.assert_called_once()

    assert install_release.call_args[1]["values"]["limits"]["workers"] == num_workers - 1
    assert install_release.call_args[1]["values"]["limits"]["gpu"] == CREATE_GCLOUD_DEFAULTS["num_gpus"]
    assert install_release.call_args[1]["values"]["limits"]["cpu"] == CREATE_GCLOUD_DEFAULTS["num_cpus"]

    get_operation.assert_called()
    assert CREATE_GCLOUD_DEFAULTS["zone"] in get_operation.call_args[1]["name"]
    assert project in get_operation.call_args[1]["name"]