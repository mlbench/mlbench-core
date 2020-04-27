#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.cli` package."""

import pytest
from mlbench_core.cli import cli_group
from google.auth.exceptions import DefaultCredentialsError
#import click

create_gcloud = cli_group.commands['create-cluster'].commands['gcloud'].main
delete_gcloud = cli_group.commands['delete-cluster'].commands['gcloud'].main
get_status = cli_group.commands['status']
delete_run = cli_group.commands['delete']

CREATE_GCLOUD_DEFAULTS = {
    'machine_type'  : 'n1-standard-4',
    'disk_size'     : 50,
    'num_cpus'      : 1,
    'num_gpus'      : 0,
    'gpu_type'      : "nvidia-tesla-k80",
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
        'project'       : '-p',
        'custom_value'  : '-v',
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
    response_create = gclient.create_cluster.return_value
    response_delete = gclient.delete_cluster.return_value
    response_get_op = gclient.get_operation.return_value

    response_create.status = 0
    response_create.DONE = 1
    response_create.name = 'test-create-name' # for concatenation

    response_delete.status = 0
    response_delete.DONE = 1
    response_delete.name = 'test-delete-name'

    response_get_op.status = 1
    response_get_op.DONE = 1
    response_get_op.name = 'test-get-op-name'


    return {
        'auth'              : gcloud_auth,
        'gclient'           : gclient,
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




@pytest.fixture
def status_mock(mocker, gcloud_mock):
    return status_mock_helper(mocker, gcloud_mock, 'test-run-name')


@pytest.fixture
def status_mock_no_run(mocker, gcloud_mock):
    return status_mock_helper(mocker, gcloud_mock, None)


def status_mock_helper(mocker, gcloud_mock, run_name):
    setup = mocker.patch('mlbench_core.cli.cli.setup_client_from_config')
    setup.return_value = True

    client = gcloud_mock['apiclient']
    rundict = {
        'name' : 'none' if run_name is None else run_name,
        'job_id': '',
        'job_metadata' : '',
        'id': 'my-id',
    }

    client.return_value.get_runs.return_value.result.return_value.json.return_value = [rundict]

    return client, rundict['id'], rundict['name']

def create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict=None):
    container_v1 = gcloud_mock["containerv1"]
    tiller = gcloud_mock["tiller"]
    auth = gcloud_mock["auth"]

    cluster = container_v1.types.Cluster
    get_operation = container_v1.ClusterManagerClient.return_value.get_operation
    nodeconfig = container_v1.types.NodeConfig
    accelerator = container_v1.types.AcceleratorConfig

    install_release = tiller.return_value.install_release
    
    if option_dict is not None and 'project' in option_dict: 
        project = option_dict['project']
    else:
        _, project = auth()

    cmd_line = get_gcloud_cmd_line(args, option_dict)

    if option_dict is None:
        option_dict = CREATE_GCLOUD_DEFAULTS
    else:
        for k in CREATE_GCLOUD_DEFAULTS: # add missing defaults
            if k not in option_dict:
                option_dict[k] = CREATE_GCLOUD_DEFAULTS[k]      

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

    
    if option_dict["num_gpus"] > 0:
        num_gpus = option_dict["num_gpus"]
        gpu_type = option_dict["gpu_type"]
        accelerator.assert_called_once_with(accelerator_count=num_gpus, accelerator_type=gpu_type)


def test_invalid_num_workers(mocker, gcloud_auth):
    
    with pytest.raises(AssertionError):
        create_gcloud(['1','test'])


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
        'project'       : 'myproj'
    }

    create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict)

def test_create_cluster_gpu(mocker, gcloud_mock):
    args = [10, 'test']
    option_dict = {
        'num_gpus'  : 3
    }

    create_gcloud_test_helper(mocker, gcloud_mock, args, option_dict)


def test_delete_cluster(gcloud_mock):
    gclient = gcloud_mock['gclient']
    delete_cluster = gclient.delete_cluster
    get_operation = gclient.get_operation

    args = ['test-name']
    opts = {'project' : 'test-proj', 'zone' : 'test-zone'}

    cmd = get_gcloud_cmd_line(args, opts)

    delete_gcloud(cmd)

    delete_cluster.assert_called_once()
    get_operation.assert_called_once()

    for arg in list(opts.values()):
        assert arg in delete_cluster.call_args[1]["name"]    
        assert arg in get_operation.call_args[1]["name"]

    assert args[0] in delete_cluster.call_args[1]["name"]
    assert delete_cluster.return_value.name in get_operation.call_args[1]["name"]



def test_status(status_mock):
    
    client, rid, name = status_mock
    url = 'my/test/url'

    # skip reporting
    client.return_value.get_run_metrics.return_value.result.return_value.status_code = 301

    get_run_metrics = client.return_value.get_run_metrics

    cmd = [name, '-u', url]

    get_status(cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    get_run_metrics.assert_called()
    assert get_run_metrics.mock_calls[0][1][0] == rid
    assert get_run_metrics.mock_calls[1][1][0] == rid

def test_status_no_run(status_mock_no_run):
    
    client, rid, _ = status_mock_no_run
    get_run_metrics = client.return_value.get_run_metrics

    url = 'my/test/url'
    name = 'my-test-name'
    cmd = [name, '-u', url]

    get_status(cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not get_run_metrics.called
    


def test_delete_no_run(status_mock_no_run):
    # Tests deleting a non-existent run
    client, rid, _ = status_mock_no_run
    delete_run_client = client.return_value.delete_run

    url = 'my/test/url'
    name = 'my-test-name'
    cmd = [name, '-u', url]

    delete_run(cmd)
    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    assert not delete_run_client.called


def test_delete_run(status_mock):
    
    client, rid, name = status_mock
    url = 'my/test/url'

    delete_run_client = client.return_value.delete_run

    cmd = [name, '-u', url]

    delete_run(cmd)

    client.assert_called_once_with(in_cluster=False, url=url, load_config=False)
    delete_run_client.assert_called_once_with(rid)