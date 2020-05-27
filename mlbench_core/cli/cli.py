# -*- coding: utf-8 -*-

"""Console script for mlbench_cli."""
import configparser
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from time import sleep
from urllib import request

import click
import urllib3
import yaml
from appdirs import user_data_dir
from kubernetes import client
from pyhelm.chartbuilder import ChartBuilder
from pyhelm.tiller import Tiller
from tabulate import tabulate

import mlbench_core
from mlbench_core.api import MLBENCH_BACKENDS, MLBENCH_IMAGES, ApiClient

GCLOUD_NVIDIA_DAEMONSET = (
    "https://raw.githubusercontent.com/"
    "GoogleCloudPlatform/container-engine-accelerators/"
    "stable/nvidia-driver-installer/cos/"
    "daemonset-preloaded.yaml"
)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
TILLER_MANIFEST_DEPLOYMENT = """apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  creationTimestamp: null
  labels:
    app: helm
    name: tiller
  name: tiller-deploy
  namespace: kube-system
spec:
  replicas: 1
  strategy: {}
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: helm
        name: tiller
    spec:
      serviceAccount: tiller
      containers:
      - env:
        - name: TILLER_NAMESPACE
          value: kube-system
        - name: TILLER_HISTORY_MAX
          value: '0'
        image: gcr.io/kubernetes-helm/tiller:v2.14.3
        imagePullPolicy: IfNotPresent
        livenessProbe:
          httpGet:
            path: /liveness
            port: 44135
          initialDelaySeconds: 1
          timeoutSeconds: 1
        name: tiller
        ports:
        - containerPort: 44134
          name: tiller
        - containerPort: 44135
          name: http
        readinessProbe:
          httpGet:
            path: /readiness
            port: 44135
          initialDelaySeconds: 1
          timeoutSeconds: 1
        resources: {}
status: {}"""

TILLER_MANIFEST_SERVICE = """apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    app: helm
    name: tiller
  name: tiller-deploy
  namespace: kube-system
spec:
  ports:
  - name: tiller
    port: 44134
    targetPort: tiller
  selector:
    app: helm
    name: tiller
  type: ClusterIP
status:
  loadBalancer: {}"""


@click.group()
@click.version_option(mlbench_core.__version__, help="Print mlbench version")
def cli_group(args=None):
    """Console script for mlbench_cli."""
    return 0


@cli_group.command()
@click.argument("name", type=str)
@click.argument("num_workers", nargs=-1, type=int, metavar="num-workers")
@click.option("--gpu", "-g", default=False, type=bool, is_flag=True)
@click.option("--light", "-l", default=False, type=bool, is_flag=True)
@click.option("--dashboard-url", "-u", default=None, type=str)
def run(name, num_workers, gpu, light, dashboard_url):
    """Start a new run for a benchmark image"""
    current_run_inputs = {}

    last_run_inputs_dir_location = os.path.join(
        os.environ["HOME"], ".local", "share", "mlbench"
    )
    Path(last_run_inputs_dir_location).mkdir(parents=True, exist_ok=True)

    last_run_inputs_file_location = os.path.join(
        last_run_inputs_dir_location, "last_run_inputs.pkl"
    )

    try:
        last_run_inputs = pickle.load(open(last_run_inputs_file_location, "rb"))
    except FileNotFoundError as e:
        last_run_inputs = {}

    images = list(MLBENCH_IMAGES.keys())

    text_prompt = "Benchmark: \n\n"

    text_prompt += "\n".join("[{}]\t{}".format(i, t) for i, t in enumerate(images))
    text_prompt += "\n[{}]\tCustom Image".format(len(images))

    text_prompt += "\n\nSelection"

    selection = click.prompt(
        text_prompt,
        type=click.IntRange(0, len(images)),
        default=last_run_inputs.get("benchmark", 0),
    )
    current_run_inputs["benchmark"] = selection

    if selection == len(images):
        # run custom image
        image = click.prompt(
            "Image", type=str, default=last_run_inputs.get("image", None)
        )
        current_run_inputs["image"] = image
        image_command = click.prompt(
            "Command", type=str, default=last_run_inputs.get("image_command", None)
        )
        current_run_inputs["image_command"] = image_command
        benchmark = {
            "custom_image_name": image,
            "custom_image_command": image_command,
        }
    else:
        benchmark = {"image": images[selection]}

    # Backend Prompt
    text_prompt = "Backend: \n\n"
    text_prompt += "\n".join(
        "[{}]\t{}".format(i, t) for i, t in enumerate(MLBENCH_BACKENDS)
    )
    text_prompt += "\n[{}]\tCustom Backend".format(len(MLBENCH_BACKENDS))
    text_prompt += "\n\nSelection"

    selection = click.prompt(
        text_prompt,
        type=click.IntRange(0, len(MLBENCH_BACKENDS)),
        default=last_run_inputs.get("backend", 0),
    )
    current_run_inputs["backend"] = selection

    if selection == len(MLBENCH_BACKENDS):
        backend = click.prompt(
            "Backend", type=str, default=last_run_inputs.get("custom_backend", None)
        )
        current_run_inputs["custom_backend"] = backend
        run_on_all = click.confirm(
            "Run command on all nodes (otherwise just first node)",
            default=last_run_inputs.get("run_on_all", None),
        )
        current_run_inputs["run_on_all"] = run_on_all
        benchmark["custom_backend"] = backend
        benchmark["run_all_nodes"] = run_on_all
    else:
        benchmark["backend"] = MLBENCH_BACKENDS[selection]

    pickle.dump(current_run_inputs, open(last_run_inputs_file_location, "wb"))

    benchmark["gpu_enabled"] = gpu
    benchmark["light_target"] = light

    loaded = setup_client_from_config()

    client = ApiClient(in_cluster=False, url=dashboard_url, load_config=not loaded)

    results = []

    for num_w in num_workers:
        current_name = "{}-{}".format(name, num_w)

        res = client.create_run(current_name, num_w, **benchmark)
        results.append(res)

    for res in results:
        act_result = res.result()
        if act_result.status_code > 201:
            try:
                click.echo(
                    "Couldn't start run: {}".format(act_result.json()["message"])
                )
            except json.JSONDecodeError:
                print(str(act_result.text))
                click.echo(
                    "Couldn't start run: Status {} for request".format(
                        act_result.status_code
                    )
                )
            return

        click.echo("Run started with name {}".format(act_result.json()["name"]))


@cli_group.command()
@click.argument("name", type=str, required=False)
@click.option("--dashboard-url", "-u", default=None, type=str)
def status(name, dashboard_url):
    """Get the status of a benchmark run, or all runs if no name is given"""
    loaded = setup_client_from_config()

    client = ApiClient(in_cluster=False, url=dashboard_url, load_config=not loaded)

    ret = client.get_runs()
    runs = ret.result().json()

    if name is None:  # List all runs
        for run in runs:
            del run["job_id"]
            del run["job_metadata"]

        click.echo(tabulate(runs, headers="keys"))
        return

    try:
        run = next(r for r in runs if r["name"] == name)
    except StopIteration:
        click.echo("Run not found")
        return

    del run["job_id"]
    del run["job_metadata"]

    click.echo(tabulate([run], headers="keys"))

    loss = client.get_run_metrics(
        run["id"], metric_filter="val_global_loss @ 0", last_n=1
    )
    prec = client.get_run_metrics(
        run["id"], metric_filter="val_global_Prec@1 @ 0", last_n=1
    )

    loss = loss.result()
    prec = prec.result()

    if loss.status_code < 300 and "val_global_loss @ 0" in loss.json():
        val = loss.json()["val_global_loss @ 0"][0]
        click.echo(
            "Current Global Loss: {0:.2f} ({1})".format(
                float(val["value"]), val["date"]
            )
        )
    else:
        click.echo("No Validation Loss Data yet")
    if prec.status_code < 300 and "val_global_Prec@1 @ 0" in prec.json():
        val = prec.json()["val_global_Prec@1 @ 0"][0]
        click.echo(
            "Current Global Precision: {0:.2f} ({1})".format(
                float(val["value"]), val["date"]
            )
        )
    else:
        click.echo("No Validation Precision Data yet")


@cli_group.command()
def get_dashboard_url():
    """Returns the dashboard URL of the current cluster"""
    loaded = setup_client_from_config()

    if not loaded:
        click.echo("No Cluster config found")
        return

    client = ApiClient(in_cluster=False, load_config=False)

    click.echo(client.endpoint.replace("api/", ""))


@cli_group.command()
@click.argument("name", type=str)
@click.option("--dashboard-url", "-u", default=None, type=str)
def delete(name, dashboard_url):
    """Delete a benchmark run"""
    loaded = setup_client_from_config()

    client = ApiClient(in_cluster=False, url=dashboard_url, load_config=not loaded)

    ret = client.get_runs()
    runs = ret.result().json()

    try:
        run = next(r for r in runs if r["name"] == name)
    except StopIteration:
        click.echo("Run not found")
        return

    del run["job_id"]
    del run["job_metadata"]

    client.delete_run(run["id"])


@cli_group.command()
@click.argument("name", type=str)
@click.option("--output", "-o", default="results.zip", type=str)
@click.option("--dashboard-url", "-u", default=None, type=str)
def download(name, output, dashboard_url):
    """Download the results of a benchmark run"""
    loaded = setup_client_from_config()

    client = ApiClient(in_cluster=False, url=dashboard_url, load_config=not loaded)

    ret = client.get_runs()
    runs = ret.result().json()

    run = next(r for r in runs if r["name"] == name)

    ret = client.download_run_metrics(run["id"])

    with open(output, "wb") as f:
        f.write(ret.result().content)


@cli_group.group("delete-cluster")
def delete_cluster():
    pass


@delete_cluster.command("gcloud")
@click.argument("name", type=str)
@click.option("--project", "-p", default=None, type=str)
@click.option("--zone", "-z", default="europe-west1-b", type=str)
def delete_gcloud(name, zone, project):
    from google.cloud import container_v1
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

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

    name_path = "projects/{}/locations/{}/".format(project, zone)

    cluster_path = "{}clusters/{}".format(name_path, name)

    response = gclient.delete_cluster(None, None, None, name=cluster_path)

    # wait for operation to complete
    while response.status < response.DONE:
        response = gclient.get_operation(
            None, None, None, name=name_path + "/" + response.name
        )
        sleep(1)

    if response.status != response.DONE:
        raise ValueError("Cluster deletion failed!")

    click.echo("Cluster deleted.")


@cli_group.group("create-cluster")
def create_cluster():
    pass


@create_cluster.command("gcloud")
@click.argument("num_workers", type=int, metavar="num-workers")
@click.argument("release", type=str)
@click.option("--kubernetes-version", "-k", type=str, default="1.13")
@click.option("--machine-type", "-t", default="n1-standard-4", type=str)
@click.option("--disk-size", "-d", default=50, type=int)
@click.option("--num-cpus", "-c", default=4, type=int)
@click.option("--num-gpus", "-g", default=0, type=int)
@click.option("--gpu-type", default="nvidia-tesla-k80", type=str)
@click.option("--zone", "-z", default="europe-west1-b", type=str)
@click.option("--project", "-p", default=None, type=str)
@click.option("--preemptible", "-e", is_flag=True)
@click.option("--custom-value", "-v", multiple=True)
def create_gcloud(
    num_workers,
    release,
    kubernetes_version,
    machine_type,
    disk_size,
    num_cpus,
    num_gpus,
    gpu_type,
    zone,
    project,
    preemptible,
    custom_value,
):
    from google.cloud import container_v1
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    from googleapiclient import discovery, http

    try:
        credentials, default_project = google.auth.default()
    except DefaultCredentialsError:
        raise click.UsageError(
            "Couldn't find gcloud credentials. Install the gcloud"
            " sdk ( https://cloud.google.com/sdk/docs/quickstart-linux ) and "
            "run 'gcloud auth application-default login' to login and create "
            "your credentials."
        )

    assert num_workers >= 2, "Number of workers should be at least 2"

    if not project:
        project = default_project

    # create cluster
    gclient = container_v1.ClusterManagerClient()

    name = "{}-{}".format(release, num_workers)
    name_path = "projects/{}/locations/{}/".format(project, zone)

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
            oauth_scopes=["https://www.googleapis.com/auth/devstorage.full_control",],
            **extraargs,
        ),
        addons_config=container_v1.types.AddonsConfig(
            http_load_balancing=container_v1.types.HttpLoadBalancing(disabled=True,),
            horizontal_pod_autoscaling=container_v1.types.HorizontalPodAutoscaling(
                disabled=True,
            ),
            kubernetes_dashboard=container_v1.types.KubernetesDashboard(disabled=True,),
            network_policy_config=container_v1.types.NetworkPolicyConfig(
                disabled=False,
            ),
        ),
        logging_service=None,
        monitoring_service=None,
    )
    response = gclient.create_cluster(None, None, cluster, parent=name_path)

    # wait for cluster to load
    while response.status < response.DONE:
        response = gclient.get_operation(
            None, None, None, name=name_path + "/" + response.name
        )
        sleep(1)

    if response.status != response.DONE:
        raise ValueError("Cluster creation failed!")

    cluster = gclient.get_cluster(None, None, None, name=name_path + "/" + name)

    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    configuration = client.Configuration()
    configuration.host = f"https://{cluster.endpoint}:443"
    configuration.verify_ssl = False
    configuration.api_key = {"authorization": "Bearer " + credentials.token}
    client.Configuration.set_default(configuration)

    if num_gpus > 0:
        with request.urlopen(GCLOUD_NVIDIA_DAEMONSET) as r:
            dep = yaml.safe_load(r)
            dep["spec"]["selector"] = {
                "matchLabels": dep["spec"]["template"]["metadata"]["labels"]
            }
            dep = client.ApiClient()._ApiClient__deserialize(dep, "V1DaemonSet")
            k8s_client = client.AppsV1Api()
            k8s_client.create_namespaced_daemon_set("kube-system", body=dep)

    # create tiller service account
    client.CoreV1Api().create_namespaced_service_account(
        "kube-system",
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "tiller",
                "generateName": "tiller",
                "namespace": "kube-system",
            },
        },
    )

    client.RbacAuthorizationV1beta1Api().create_cluster_role_binding(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1beta1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "tiller"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "cluster-admin",
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": "tiller", "namespace": "kube-system"}
            ],
        }
    )

    # deploy tiller
    tiller_service = yaml.safe_load(TILLER_MANIFEST_SERVICE)
    tiller_dep = yaml.safe_load(TILLER_MANIFEST_DEPLOYMENT)
    client.CoreV1Api().create_namespaced_service("kube-system", tiller_service)
    client.ExtensionsV1beta1Api().create_namespaced_deployment(
        "kube-system", tiller_dep
    )

    sleep(1)

    pods = client.CoreV1Api().list_namespaced_pod(
        namespace="kube-system", label_selector="app=helm"
    )

    tiller_pod = pods.items[0]

    while True:
        # Wait for tiller
        resp = client.CoreV1Api().read_namespaced_pod(
            namespace="kube-system", name=tiller_pod.metadata.name
        )
        if resp.status.phase != "Pending":
            break
        sleep(5)

    # kubernetes python doesn't currently support port forward
    # https://github.com/kubernetes-client/python/issues/166
    ports = 44134

    # resp = stream(
    #     client.CoreV1Api().connect_get_namespaced_pod_portforward,
    #     name=tiller_pod.metadata.name,
    #     namespace=tiller_pod.metadata.namespace,
    #     ports=ports
    #     )

    with subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "--namespace={}".format(tiller_pod.metadata.namespace),
            tiller_pod.metadata.name,
            "{0}:{0}".format(ports),
            "--server={}".format(configuration.host),
            "--token={}".format(credentials.token),
            "--insecure-skip-tls-verify=true",
        ]
    ) as portforward:

        sleep(5)
        # install chart
        tiller = Tiller("localhost")
        chart = ChartBuilder(
            {
                "name": "mlbench-helm",
                "source": {
                    "type": "git",
                    "location": "https://github.com/mlbench/mlbench-helm",
                },
            }
        )

        values = {
            "limits": {"workers": num_workers - 1, "gpu": num_gpus, "cpu": num_cpus - 1}
        }

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

        tiller.install_release(
            chart.get_helm_chart(),
            name=name,
            wait=True,
            dry_run=False,
            namespace="default",
            values=values,
        )

        portforward.terminate()

    # open port in firewall
    mlbench_client = ApiClient(in_cluster=False, load_config=False)
    firewall_body = {
        "name": fw_name,
        "direction": "INGRESS",
        "sourceRanges": "0.0.0.0/0",
        "allowed": [{"IPProtocol": "tcp", "ports": [mlbench_client.port]}],
    }

    firewalls.insert(project=project, body=firewall_body).execute()

    config = get_config()

    config.set("general", "provider", "gke")

    config.set("gke", "cluster", cluster.endpoint)

    write_config(config)

    click.echo("MLBench successfully deployed")


def get_config_path():
    user_dir = user_data_dir("mlbench", "mlbench")
    return os.path.join(user_dir, "mlbench.ini")


def get_config():
    path = get_config_path()

    config = configparser.ConfigParser()

    if os.path.exists(path):
        config.read(path)

    if not config.has_section("general"):
        config.add_section("general")

    if not config.has_section("gke"):
        config.add_section("gke")

    return config


def write_config(config):
    path = get_config_path()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as configfile:
        config.write(configfile)


def setup_client_from_config():
    config = get_config()

    provider = config.get("general", "provider", fallback=None)

    if not provider:
        return False

    if provider == "gke":
        return setup_gke_client_from_config(config)
    else:
        raise NotImplementedError()


def setup_gke_client_from_config(config):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    cluster = config.get("gke", "cluster")
    if not cluster:
        return False

    try:
        credentials, default_project = google.auth.default()
    except DefaultCredentialsError:
        raise click.UsageError(
            "Couldn't find gcloud credentials. Install the gcloud"
            " sdk ( https://cloud.google.com/sdk/docs/quickstart-linux ) and "
            "run 'gcloud auth application-default login' to login and create "
            "your credentials."
        )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    configuration = client.Configuration()
    configuration.host = f"https://{cluster}:443"
    configuration.verify_ssl = False
    configuration.api_key = {"authorization": "Bearer " + credentials.token}
    client.Configuration.set_default(configuration)

    return True


if __name__ == "__main__":
    sys.exit(cli_group())  # pragma: no cover
