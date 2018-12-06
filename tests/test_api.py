#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.api` package."""

import pytest
import datetime

from mlbench_core.api import ApiClient


@pytest.fixture
def kubernetes_api_client_node_port(mocker):
    mock_client = mocker.patch('kubernetes.client.CoreV1Api')
    mock_client.return_value.read_namespaced_service.return_value.spec.type = "NodePort"
    mock_client.return_value.read_namespaced_service.return_value.spec.ports.__getitem__.return_value.node_port = 12345
    mock_client.return_value.list_namespaced_pod.return_value.items.__len__.return_value = 1
    mock_client.return_value.read_node.return_value.status.addresses.__len__.return_value = 1
    mock_client.return_value.read_node.return_value.status.addresses.__iter__.return_value = [mocker.MagicMock(type="ExternalIP", address="1.1.1.1")]

    return mock_client


@pytest.fixture
def kubernetes_api_client_node_port_internal(mocker):
    mock_client = mocker.patch('kubernetes.client.CoreV1Api')
    mock_client.return_value.read_namespaced_service.return_value.spec.type = "NodePort"
    mock_client.return_value.read_namespaced_service.return_value.spec.ports.__getitem__.return_value.node_port = 12345
    mock_client.return_value.list_namespaced_pod.return_value.items.__len__.return_value = 1
    mock_client.return_value.read_node.return_value.status.addresses.__len__.return_value = 1
    mock_client.return_value.read_node.return_value.status.addresses.__iter__.return_value = [mocker.MagicMock(type="InternalIP", address="1.1.1.1")]

    return mock_client


@pytest.fixture
def kubernetes_api_client_clusterip(mocker):
    mock_client = mocker.patch('kubernetes.client.CoreV1Api')
    mock_client.return_value.read_namespaced_service.return_value.spec.type = "ClusterIP"
    mock_client.return_value.read_namespaced_service.return_value.spec.cluster_ip = "1.1.1.1"
    mock_client.return_value.read_namespaced_service.return_value.spec.ports.__getitem__.return_value.port = 12345

    return mock_client


@pytest.fixture
def kubernetes_api_client_loadbalancer(mocker):
    mock_client = mocker.patch('kubernetes.client.CoreV1Api')
    mock_client.return_value.read_namespaced_service.return_value.spec.type = "LoadBalancer"
    mock_client.return_value.read_namespaced_service.return_value.spec.ports.__getitem__.return_value.port = 12345
    mock_client.return_value.read_namespaced_service.return_value.status.load_balancer.ingress.ip = "1.1.1.1"

    return mock_client


@pytest.fixture
def kubernetes_api_client_incluster(mocker):
    mock_client = mocker.patch('kubernetes.client.CoreV1Api')
    mock_client.return_value.list_namespaced_pod.return_value.items.__len__.return_value = 1
    mock_client.return_value.list_namespaced_pod.return_value.items.__getitem__.return_value.status.pod_ip = "1.1.1.1"

    return mock_client


def test_instantiation(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    with ApiClient(in_cluster=False, service_name="rel-mlbench-master") as client:
        assert client is not None
        assert client.endpoint == "http://1.1.1.1:12345/api/"


def test_instantiation_nodeport_internal(mocker, kubernetes_api_client_node_port_internal):
    mocker.patch('kubernetes.config.load_kube_config')
    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    assert client is not None
    assert client.endpoint == "http://1.1.1.1:12345/api/"


def test_instantiation_url():
    client = ApiClient(url="1.1.1.1:12345")

    assert client is not None
    assert client.endpoint == "http://1.1.1.1:12345/api/"


def test_instantiation_incluster(mocker, kubernetes_api_client_incluster):
    mocker.patch('kubernetes.config.load_incluster_config')

    client = ApiClient(in_cluster=True)

    assert client is not None
    assert client.endpoint == "http://1.1.1.1:80/api/"

def test_instantiation_clusterip(mocker, kubernetes_api_client_clusterip):
    mocker.patch('kubernetes.config.load_kube_config')
    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    assert client is not None
    assert client.endpoint == "http://1.1.1.1:12345/api/"


def test_instantiation_loadbalancer(mocker, kubernetes_api_client_loadbalancer):
    mocker.patch('kubernetes.config.load_kube_config')
    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    assert client is not None
    assert client.endpoint == "http://1.1.1.1:12345/api/"


def test_get_all_metrics(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_all_metrics()

    assert result is not None
    assert result.result().json() == "a"


def test_get_run_metrics(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_run_metrics("1", since=datetime.datetime.now(), summarize=100)

    assert result is not None
    assert result.result().json() == "a"


def test_get_pod_metrics(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_pod_metrics("rel-mlbench-worker-0", since=datetime.datetime.now(), summarize=100)

    assert result is not None
    assert result.result().json() == "a"


def test_post_metrics(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.post_metric("1", "loss", 10.0, cumulative=False)

    assert result is not None
    assert result.result().json() == "a"


def test_get_runs(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_runs()

    assert result is not None
    assert result.result().json() == "a"


def test_get_run(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_run("1")

    assert result is not None
    assert result.result().json() == "a"


def test_create_run_official(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.create_run(
        "test_run",
        5,
        num_cpus=4.1,
        max_bandwidth=10000,
        image='Test Image')

    assert result is not None
    assert result.result().json() == "a"


def test_create_run_custom(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.create_run(
        "test_run",
        5,
        num_cpus=4.1,
        max_bandwidth=10000,
        custom_image_name="localhost:5000/mlbench_worker:latest",
        custom_image_command="/.openmpi/bin/mpirun /app/main.py",
        custom_image_all_nodes=False)

    assert result is not None
    assert result.result().json() == "a"


def test_get_worker_pods(mocker, kubernetes_api_client_node_port):
    mocker.patch('kubernetes.config.load_kube_config')
    rg = mocker.patch('concurrent.futures.ProcessPoolExecutor')
    rg.return_value.submit.return_value.result.return_value.json.return_value = "a"

    client = ApiClient(in_cluster=False, service_name="rel-mlbench-master")

    result = client.get_worker_pods()

    assert result is not None
    assert result.result().json() == "a"