import subprocess
from time import sleep

from kubernetes import client

from mlbench_core.cli.chartbuilder import ChartBuilder


def _get_master_pod(release_name, pods):
    """Given a release name and a list of pods, returns the master pod of the release

    Args:
        release_name (str): Release name
        pods (:obj:`V1PodList`): List of pods

    Returns:
        (:obj:`Pod`, optional): The master pod
    """

    master_pod_name = "{}-mlbench-master-".format(release_name)
    for pod in pods.items:
        if master_pod_name in pod.metadata.name:
            return pod

    return None


def _wait_for_deployment(release_name):
    """Given a release name, waits for the master pod to be running

    Args:
        release_name (str): Release name

    Raises:
        ValueError: If the master pod is not running
    """
    kube_api = client.CoreV1Api()
    pods = kube_api.list_namespaced_pod(namespace="default")
    master_pod = _get_master_pod(release_name, pods)
    while master_pod is None or master_pod.status.phase == "Pending":
        pods = kube_api.list_namespaced_pod(namespace="default")
        master_pod = _get_master_pod(release_name, pods)
        sleep(1)
    if master_pod is None or master_pod.status.phase in ["Failed", "Unknown"]:
        raise ValueError("Could not deploy chart")


def deploy_chart(
    num_workers,
    num_gpus,
    num_cpus,
    release_name,
    custom_value,
    kube_context,
    custom_chart=None,
):
    """Deploys the mlbench-helm chart given its values

    Args:
        num_workers (int): Number of worker nodes (excluding master)
        num_gpus (int): Number of GPUs per node
        num_cpus (int): Number of CPUs per node
        release_name (str): Release name
        custom_value (str): Custom values for chart
        kube_context (str): Current kube-context (must be saved in kubeconfig)
        custom_chart (dict, optional): Custom chart to use (e.g. local chart)
    """
    sleep(5)

    # install chart
    chart = ChartBuilder(
        {
            "name": "mlbench-helm",
            "source": {
                "type": "git",
                "location": "https://github.com/mlbench/mlbench-helm",
            },
        }
        if custom_chart is None
        else custom_chart
    )

    values = {"limits": {"workers": num_workers, "gpu": num_gpus, "cpu": num_cpus}}
    if custom_value:
        # merge custom values with values
        for cv in custom_value:
            key, v = cv.split("=", 1)

            current = values
            key_path = key.split(".")

            for k in key_path[:-1]:
                if k not in current:
                    current[k] = {}

                current = current[k]

            current[key_path[-1]] = v

    chart_path = chart.get_chart(release_name, values)

    output = subprocess.check_output(
        [
            "kubectl",
            "apply",
            "--validate=false",
            "--context={}".format(kube_context),
            "-f",
            chart_path,
        ]
    )
    sleep(1)

    _wait_for_deployment(release_name)
