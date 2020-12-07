import os
import subprocess
import sys
from time import sleep
from urllib import request

import click
import google.auth
import yaml
from google.api_core.exceptions import AlreadyExists, NotFound
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import container_v1
from googleapiclient import discovery, http
from kubernetes import client as kube_client

GCLOUD_NVIDIA_DAEMONSET = (
    "https://raw.githubusercontent.com/"
    "GoogleCloudPlatform/container-engine-accelerators/"
    "stable/nvidia-driver-installer/cos/"
    "daemonset-preloaded.yaml"
)


def _create_kube_config_gcloud_entry(cluster_name, cluster_zone, project):
    """Uses GCloud CLI to create an entry for Kubectl.

    This is needed as we install the charts using kubectl, and it needs the correct config

    Args:
        cluster_name (str): Name of cluster
        cluster_zone (str): Zone of cluster
        project (str): Current used project

    Returns:
        (str): Kube context for the cluster
    """
    p = subprocess.Popen(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            cluster_name,
            "--zone",
            cluster_zone,
            "--project",
            project,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output, error = p.communicate()

    if p.returncode != 0:
        raise click.UsageError(
            "Failed to add kube config entry:\n {}".format(error.decode())
        )

    context_name = "gke_{}_{}_{}".format(project, cluster_zone, cluster_name)
    return context_name


def setup_gcloud_kube_client(cluster_endpoint, cluster_name, cluster_zone, project):
    """Sets up the kube configuration on the given cluster

    Args:
        cluster_endpoint (str): Endpoint of cluster
        cluster_name (str): Cluster name
        cluster_zone (str): Cluster zone
        project (str): Cluster project

    Returns:
        (str): Kube context for this cluster
    """
    try:
        credentials, _ = google.auth.default()
    except DefaultCredentialsError:
        raise click.UsageError(
            "Couldn't find gcloud credentials. Install the gcloud"
            " sdk ( https://cloud.google.com/sdk/docs/quickstart-linux ) and "
            "run 'gcloud auth application-default login' to login and create "
            "your credentials."
        )

    # Setup kube client
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    configuration = kube_client.Configuration()
    configuration.host = f"https://{cluster_endpoint}:443"
    configuration.verify_ssl = False
    configuration.api_key = {"authorization": "Bearer " + credentials.token}
    kube_client.Configuration.set_default(configuration)

    kube_context = _create_kube_config_gcloud_entry(cluster_name, cluster_zone, project)

    return kube_context


def gcloud_create_cluster(
    name,
    name_path,
    num_workers,
    num_gpus,
    gpu_type,
    machine_type,
    disk_size,
    preemptible,
    kubernetes_version,
    project,
):
    """Creates a GCloud cluster based on the given arguments

    Args:
        name (str): Cluster name
        name_path (str): Cluster name path (` "projects/{}/locations/{}/clusters/"`)
        num_workers (int): Total number of nodes (including master node)
        num_gpus (int): Total number of gpus per node
        gpu_type (str): GPU type
        machine_type (str): Machine type
        disk_size (int): Disk size in GB
        preemptible (bool): Use preemptible machines
        kubernetes_version (str): Kubernetes version
        project (str): GCLoud project

    Returns:
        (:obj:`google.container_v1.ClusterManagementClient`, str, :obj:): The client for cluster communication,
            firewall name, and cluster firewalls
    """
    assert num_workers >= 2, "Number of workers should be at least 2"

    # create cluster
    gclient = container_v1.ClusterManagerClient()

    extraargs = {}

    if num_gpus > 0:
        extraargs["accelerators"] = [
            container_v1.types.AcceleratorConfig(
                accelerator_count=num_gpus, accelerator_type=gpu_type
            )
        ]

    # delete existing firewall, if any
    firewalls = discovery.build("compute", "v1", cache_discovery=False).firewalls()
    existing_firewalls = firewalls.list(project=project).execute()
    fw_name = "{}-firewall".format(name)

    if any(f["name"] == fw_name for f in existing_firewalls["items"]):
        response = {}
        while not hasattr(response, "status"):
            try:
                response = firewalls.delete(project=project, firewall=fw_name).execute()
            except http.HttpError as e:
                if e.resp.status == 404:
                    response = {}
                    break
                click.echo("Wait for firewall to be available for deletion")
                sleep(5)
                response = {}
        while hasattr(response, "status") and response.status < response.DONE:
            response = gclient.get_operation(None, None, None, name=response.selfLink)
            sleep(1)

    # create cluster
    cluster = container_v1.types.Cluster(
        name=name,
        initial_node_count=num_workers,
        node_config=container_v1.types.NodeConfig(
            machine_type=machine_type,
            disk_size_gb=disk_size,
            preemptible=preemptible,
            oauth_scopes=[
                "https://www.googleapis.com/auth/devstorage.full_control",
            ],
            **extraargs,
        ),
        addons_config=container_v1.types.AddonsConfig(
            http_load_balancing=container_v1.types.HttpLoadBalancing(
                disabled=True,
            ),
            horizontal_pod_autoscaling=container_v1.types.HorizontalPodAutoscaling(
                disabled=True,
            ),
            kubernetes_dashboard=container_v1.types.KubernetesDashboard(
                disabled=True,
            ),
            network_policy_config=container_v1.types.NetworkPolicyConfig(
                disabled=False,
            ),
        ),
        logging_service=None,
        monitoring_service=None,
        initial_cluster_version=kubernetes_version,
    )
    try:
        response = gclient.create_cluster(cluster, parent=name_path)
    except AlreadyExists as e:
        click.echo("Exception from Google: " + str(e))
        click.echo(
            "A cluster with this name already exists in the specified project and zone"
        )
        click.echo(
            "Try running 'gcloud container clusters list' to list all active clusters"
        )
        sys.exit(1)

    # wait for cluster to load
    while response.status < response.DONE:
        response = gclient.get_operation(
            None, None, None, name=os.path.join(name_path, response.name)
        )
        sleep(1)

    if response.status != response.DONE:
        raise ValueError("Cluster creation failed!")

    return gclient, fw_name, firewalls


def deploy_nvidia_daemonset():
    """ Deploys the NVIDIA daemon set to the cluster"""
    with request.urlopen(GCLOUD_NVIDIA_DAEMONSET) as r:
        dep = yaml.safe_load(r)
        dep["spec"]["selector"] = {
            "matchLabels": dep["spec"]["template"]["metadata"]["labels"]
        }
        dep = kube_client.ApiClient()._ApiClient__deserialize(dep, "V1DaemonSet")
        k8s_client = kube_client.AppsV1Api()
        k8s_client.create_namespaced_daemon_set("kube-system", body=dep)


def gcloud_delete_cluster(name, zone, project):
    """Deletes the cluster

    Args:
        name (str): Cluster name
        zone (str): Cluster zone
        project (str): Cluster project
    """
    try:
        credentials, default_project = google.auth.default()
    except DefaultCredentialsError:
        raise click.UsageError(
            "Couldn't find gcloud credentials. Install the gcloud"
            " sdk ( https://cloud.google.com/sdk/docs/quickstart-linux ) and "
            "run 'gcloud auth application-default login' to login and create "
            "your credentials."
        )

    if not project:
        project = default_project

    # delete cluster
    gclient = container_v1.ClusterManagerClient()

    name_path = "projects/{}/locations/{}/clusters/".format(project, zone)
    cluster_path = os.path.join(name_path, name)

    try:
        response = gclient.delete_cluster(None, None, None, name=cluster_path)
    except NotFound as e:
        click.echo("Exception from Google: " + str(e))
        click.echo("Double-check your project, zone and cluster name")
        click.echo(
            "Try running 'gcloud container clusters list' to list all active clusters"
        )
        sys.exit(1)

    # wait for operation to complete
    while response.status < response.DONE:
        response = gclient.get_operation(
            None, None, None, name=os.path.join(name_path, response.name)
        )
        sleep(1)

    if response.status != response.DONE:
        raise ValueError("Cluster deletion failed!")
