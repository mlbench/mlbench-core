#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from mlbench_core.cli import cli_group
from google.auth.exceptions import DefaultCredentialsError
#import click

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
    

def get_gcloud_cmd_line(args, option_dict=None):
    '''
    args: (list) command-line arguments
    option_dict: (dict) key-value for options line in CREATE_GCLOUD_DEFAULTS
    '''

    flags ={
        'machine_type'  : '-t',
        'disk_size'     : '-d',
        'num_cpus'      : '-c',
        'num_gpus'      : '-g',
        'gpu_type'      : '--gpu-type',
        'zone'          : '-z',
        'preemptible'   : '-e',
        'custom_value'  : '-v'
    }

    args = list(map(str, args))
    def get_array(option):
        #e.g. option='num_gpus' --> ['-g', 0]
        if options[option] == True: # check if flag is boolean
            return [flags[option]]
        elif options[option] == False:
            return []
        else:
            return [flags[option], options[option]] 

    if option_dict is not None:
        options = option_dict
        cmd_line_options = [str(elem) for option in options for elem in get_array(option)]

        return args + cmd_line_options
    
    return args



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


def create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict=None):
    container_v1 = gcloud_mock["containerv1"]
    tiller = gcloud_mock["tiller"]
    auth = gcloud_mock["auth"]

    cluster = container_v1.types.Cluster
    get_operation = container_v1.ClusterManagerClient.return_value.get_operation
    nodeconfig = container_v1.types.NodeConfig
    install_release = tiller.return_value.install_release
    _, project = auth()

    cmd_line = get_gcloud_cmd_line(args, option_dict)

    if option_dict is None:
        option_dict = CREATE_GCLOUD_DEFAULTS

    create_gcloud(cmd_line)

    cluster.assert_called_once()
    assert cluster.call_args[1]["initial_node_count"] == args[0] # num_workers
    
    nodeconfig.assert_called_once()
    assert nodeconfig.call_args[1]["machine_type"] == option_dict['machine_type']
    assert nodeconfig.call_args[1]["disk_size_gb"] == option_dict['disk_size']
    assert nodeconfig.call_args[1]["preemptible"] == option_dict['preemptible']

    tiller.assert_called_once()
    install_release.assert_called_once()

    assert install_release.call_args[1]["values"]["limits"]["workers"] == args[0] - 1
    assert install_release.call_args[1]["values"]["limits"]["gpu"] == option_dict["num_gpus"]
    assert install_release.call_args[1]["values"]["limits"]["cpu"] == option_dict["num_cpus"]

    get_operation.assert_called()
    assert option_dict["zone"] in get_operation.call_args[1]["name"]
    assert project in get_operation.call_args[1]["name"]


def test_create_cluster_default(mocker, gcloud_mock):
    args = [3, 'test']
    create_gcloud_test_helper(mocker, gcloud_mock, args)

def test_create_cluster_non_default(mocker, gcloud_mock):

    args = [3, 'test']
    option_dict = {
        'machine_type'  : 'my-custom-type',
        'disk_size'     : 20,
        'num_cpus'      : 5,
        'num_gpus'      : 0,
        'gpu_type'      : "my-nvidia-tesla-p100",
        'zone'          : "my-europe-west1-b",
        'preemptible'   : True,
    }

    create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict)

#def test_create_cluster_gpu(mocker, gcloud_mock):