# -*- coding: utf-8 -*-

"""Console script for mlbench_cli."""
from mlbench_core.api import ApiClient, MLBENCH_IMAGES

import click
from kubernetes import client, config
from kubernetes.stream import stream
from pyhelm.chartbuilder import ChartBuilder
from pyhelm.tiller import Tiller
from tabulate import tabulate
from time import sleep
from urllib import request
import yaml

import sys


GCLOUD_NVIDIA_DAEMONSET = ('https://raw.githubusercontent.com/'
                           'GoogleCloudPlatform/container-engine-accelerators/'
                           'stable/nvidia-driver-installer/cos/'
                           'daemonset-preloaded.yaml')

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
      automountServiceAccountToken: true
      containers:
      - env:
        - name: TILLER_NAMESPACE
          value: kube-system
        - name: TILLER_HISTORY_MAX
          value: "0"
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
def cli(args=None):
    """Console script for mlbench_cli."""
    return 0


@cli.command()
@click.argument('name', type=str)
@click.argument('num_workers', nargs=-1, type=int, metavar='num-workers')
@click.option('--dashboard-url', '--u', default=None, type=str)
def run(name, num_workers, dashboard_url):
    """Start a new run for a benchmark image"""
    images = list(MLBENCH_IMAGES.keys())

    text_prompt = 'Benchmark: \n\n'

    text_prompt += '\n'.join(
        '[{}]\t{}'.format(i, t) for i, t in enumerate(images)
    )
    text_prompt += '\n[{}]\tCustom Image'.format(len(images))

    text_prompt += '\n\nSelection'

    selection = click.prompt(
        text_prompt,
        type=click.IntRange(0, len(images)),
        default=0
    )

    if selection == len(images):
        # run custom image
        image = click.prompt('Image:', type=str)
        image_command = click.prompt('Command:', type=str)
        run_on_all = click.confirm(
            'Run command on all nodes (otherwise just first node):', type=bool)
        benchmark = {
            "custom_image_name": image,
            "custom_image_command": image_command,
            "custom_image_all_nodes": run_on_all
        }
    else:
        benchmark = {"image": images[selection]}

    client = ApiClient(in_cluster=False, url=dashboard_url)

    results = []

    for num_w in num_workers:
        current_name = '{}-{}'.format(name, num_w)

        res = client.create_run(current_name, num_w, **benchmark)
        results.append(res)

    for res in results:
        act_result = res.result()
        if act_result.status > 201:
            click.Abort("Couldn't start run: {}".format(act_result.json()["message"]))
        click.echo("Run started with name {}".format(act_result.json()["name"]))


@cli.command()
@click.argument('name', type=str)
@click.option('--dashboard-url', '--u', default=None, type=str)
def status(name, dashboard_url):
    """Get the status of a benchmark run"""
    client = ApiClient(in_cluster=False, url=dashboard_url)

    ret = client.get_runs()
    runs = ret.result().json()

    run = next(r for r in runs if r['name'] == name)

    del run["job_id"]
    del run["job_metadata"]

    click.echo(tabulate([run], headers="keys"))


@cli.command()
@click.argument('name', type=str)
@click.option('--output', '-o', type=str)
@click.option('--dashboard-url', '-u', default=None, type=str)
def download(name, output, dashboard_url):
    """Download the results of a benchmark run"""
    client = ApiClient(in_cluster=False, url=dashboard_url)

    ret = client.get_runs()
    runs = ret.result().json()

    run = next(r for r in runs if r['name'] == name)

    ret = client.download_run_metrics(run['id'])

    with open(output, 'wb') as f:
        f.write(ret.result().content)


@cli.group('create-cluster')
def create_cluster():
    pass


@create_cluster.command('gcloud')
@click.argument('num_workers', type=int, metavar='num-workers')
@click.argument('release', type=str)
@click.option('--kubernetes-version', '-k', type=str, default='1.13')
@click.option('--machine-type', '-t', default='n1-standard-2', type=str)
@click.option('--disk-size', '-d', default=50, type=int)
@click.option('--num-cpus', '-c', default=1, type=int)
@click.option('--num-gpus', '-g', default=0, type=int)
@click.option('--gpu-type', default='nvidia-tesla-p100', type=str)
@click.option('--zone', '-z', default='europe-west1-b', type=str)
@click.option('--project', '-p', default=None, type=str)
@click.option('--preemptible', '-e', is_flag=True)
def create_gcloud(num_workers, release, kubernetes_version, machine_type, disk_size, num_cpus,
                  num_gpus, gpu_type, zone, project, preemptible):
    from google.cloud import container_v1
    import google.auth
    from google.auth import compute_engine

    credentials, default_project = google.auth.default()

    if not project:
        project = default_project

    # create cluster
    gclient = container_v1.ClusterManagerClient()

    name = '{}-{}'.format(release, num_workers)
    name_path = 'projects/{}/locations/{}/'.format(project, zone)

    extraargs = {}

    if num_gpus > 0:
        extraargs['accelerator'] = 'type={},count={}'.format(gpu_type, num_gpus)

    cluster = container_v1.types.Cluster(
            name=name,
            initial_node_count=num_workers,
            node_config=container_v1.types.NodeConfig(
                machine_type=machine_type,
                disk_size_gb=disk_size,
                preemptible=preemptible,
                oauth_scopes=[
                    'https://www.googleapis.com/auth/devstorage.full_control',
                ],
                **extraargs
            ),
            addons_config=google.cloud.container_v1.types.AddonsConfig(
                http_load_balancing=google.cloud.container_v1.types.HttpLoadBalancing(
                    disabled=True,
                ),
                horizontal_pod_autoscaling=google.cloud.container_v1.types.HorizontalPodAutoscaling(
                    disabled=True,
                ),
                kubernetes_dashboard=google.cloud.container_v1.types.KubernetesDashboard(
                    disabled=True,
                ),
                network_policy_config=container_v1.types.NetworkPolicyConfig(
                    disabled=False,
                ),
            ),
            logging_service=None,
            monitoring_service=None
        )
    response = gclient.create_cluster(None, None, cluster, parent=name_path)

    # wait for cluster to load
    while response.status < response.DONE:
        response = gclient.get_operation(None,None,None,name=name_path + '/' + response.name)
        sleep(1)

    if response.status != response.DONE:
        raise ValueError("Cluster creation failed!")

    cluster = gclient.get_cluster(None, None, None, name=name_path + '/' + name)

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
            k8s_client = client.ExtensionsV1beta1Api()
            k8s_client.create_namespaced_deployment(
                body=dep, namespace="default")

    # create tiller service account
    client.CoreV1Api().create_namespaced_service_account(
        "kube-system",
        {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'generateName': "tiller",
                'namespace': "kube-system",
            },
        })

    client.RbacAuthorizationV1beta1Api().create_cluster_role_binding(
        {
            'apiVersion': 'rbac.authorization.k8s.io/v1beta1',
            'kind': 'ClusterRoleBinding',
            'metadata': {
                'name': 'tiller'
            },
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'ClusterRole',
                'name': 'cluster-admin'
            },
            'subjects': [
                {
                    'kind': 'ServiceAccount',
                    'name': 'tiller',
                    'namespace': 'kube-system'
                }
            ]
        })

    # deploy tiller
    tiller_service = yaml.safe_load(TILLER_MANIFEST_SERVICE)
    tiller_dep = yaml.safe_load(TILLER_MANIFEST_DEPLOYMENT)
    client.CoreV1Api().create_namespaced_service(
        "kube-system",
        tiller_service)
    client.ExtensionsV1beta1Api().create_namespaced_deployment(
        "kube-system",
        tiller_dep)

    sleep(1)

    pods = client.CoreV1Api().list_namespaced_pod(
        namespace="kube-system",
        label_selector='app=helm'
    )

    tiller_pod = pods.items[0]

    while True:
        # Wait for tiller
        resp = client.CoreV1Api().read_namespaced_pod(
            namespace="kube-system",
            name=tiller_pod.metadata.name
        )
        if resp.status.phase != 'Pending':
            break
        sleep(1)

    ports = 44134

    resp = stream(
        client.CoreV1Api().connect_get_namespaced_pod_portforward,
        name=tiller_pod.metadata.name,
        namespace=tiller_pod.metadata.namespace,
        ports=ports
        )

    # install chart
    tiller = Tiller('127.0.0.1')
    chart = ChartBuilder(
        {
            "name": "mlbench-helm",
            "source": {
                "type": "repo",
                "location": "https://github.com/mlbench/mlbench-helm"
            }})
    tiller.install_release(
        chart.get_helm_chart(),
        name=name,
        wait=True,
        dry_run=False,
        namespace='default')

    resp.close()


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
