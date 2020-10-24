# -*- coding: utf-8 -*-

"""Console script for mlbench_cli."""
import configparser
import json
import os
import pickle
import subprocess
import sys
from os.path import expanduser
from pathlib import Path

import boto3
import botocore.exceptions
import click
import matplotlib.pyplot as plt
import urllib3
import yaml
from appdirs import user_data_dir
from kubernetes import config as kube_config
from tabulate import tabulate

import mlbench_core
from mlbench_core.api import MLBENCH_BACKENDS, MLBENCH_IMAGES, ApiClient
from mlbench_core.cli.aws_utils import (
    aws_create_cluster,
    deploy_nvidia_daemonset_aws,
    setup_aws_kube_client,
)
from mlbench_core.cli.gcloud_utils import (
    deploy_nvidia_daemonset,
    gcloud_create_cluster,
    gcloud_delete_cluster,
    setup_gcloud_kube_client,
)
from mlbench_core.cli.kind_utils import (
    create_local_registry,
    kind_create_cluster,
    kind_delete_cluster,
)
from mlbench_core.cli.utils import deploy_chart

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@click.group()
@click.version_option(mlbench_core.__version__, help="Print mlbench version")
def cli_group(args=None):
    """Console script for mlbench_cli."""
    return 0


@cli_group.command()
@click.argument("name", type=str)
@click.argument("num_workers", nargs=-1, type=int, metavar="num-workers")
@click.option("--gpu", "-g", default=False, type=bool, is_flag=True)
@click.option("--num-cpus", "-c", default=4, type=int)
@click.option("--light", "-l", default=False, type=bool, is_flag=True)
@click.option("--dashboard-url", "-u", default=None, type=str)
def run(name, num_workers, gpu, num_cpus, light, dashboard_url):
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
    benchmark["num_cpus"] = num_cpus - 1

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
@click.argument("folder", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    "--filter", "-f", default=None, type=str, help="String filter to filter runs"
)
@click.option("--dashboard-url", "-u", default=None, type=str)
def charts(folder, filter, dashboard_url):
    """Chart the results of benchmark runs

    Save generated charts in FOLDER
    """
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
    loaded = setup_client_from_config()

    client = ApiClient(in_cluster=False, url=dashboard_url, load_config=not loaded)

    ret = client.get_runs()
    runs = ret.result().json()
    runs = [r for r in runs if r["state"] == "finished"]

    if filter:
        runs = [r for r in runs if filter in r["name"]]

    options = {i: r for i, r in enumerate(runs, start=0)}

    if len(options) < 2:
        click.echo("At least two finished runs are needed to create a summary")
        return

    options["all"] = {"name": "*all runs*"}

    prompt = 'Select the runs to generate a summary for (e.g. "0 1 2"): \n\t{}'.format(
        "\n\t".join("{} [{}]".format(r["name"], i) for i, r in options.items())
    )

    choice = click.prompt(
        prompt,
        default=0,
        type=click.Choice([options.keys()]),
        show_choices=False,
        value_proc=lambda x: runs
        if "all" in x
        else [options[int(i)] for i in x.split(" ")],
    )

    if len(choice) < 2:
        click.echo("At least two finished runs are needed to create a summary")
        return

    results = []

    def _get_metric(name, run):
        """Gets a metric from the dashboard."""
        name = "global_cum_{} @ 0".format(name)
        return float(
            client.get_run_metrics(run["id"], metric_filter=name, last_n=1)
            .result()
            .json()[name][0]["value"]
        )

    for run in choice:
        agg = _get_metric("agg", run)

        backprop = _get_metric("backprop", run)

        batch_load = _get_metric("batch_load", run)

        comp_loss = _get_metric("comp_loss", run)

        comp_metrics = _get_metric("comp_metrics", run)

        fwd_pass = _get_metric("fwd_pass", run)

        opt_step = _get_metric("opt_step", run)

        compute = (
            fwd_pass
            + comp_loss
            + backprop
            + opt_step
            + (agg if run["num_workers"] == 1 else 0)
        )
        communicate = agg if run["num_workers"] != 1 else 0

        results.append(
            (
                run["name"],
                compute,
                communicate,
                comp_metrics,
                batch_load,
                str(run["num_workers"]),
            )
        )

    results = sorted(results, key=lambda x: x[5])
    names, compute, communicate, metrics, batch_load, num_workers = zip(*results)

    width = 0.35
    fig, ax = plt.subplots()

    ax.bar(num_workers, compute, width, label="Compute")
    ax.bar(num_workers, communicate, width, label="Communication")

    ax.set_ylabel("Time (s)")
    ax.set_title("Total time by number of workers")
    ax.legend()
    plt.savefig(folder / "total_time.png", dpi=150)

    fig, ax = plt.subplots()

    combined = [c + r for _, c, r, _, _, _ in results]

    speedup = [combined[0] / c for c in combined]

    ax.bar(num_workers, speedup, width)

    ax.set_ylabel("Speedup factor")
    ax.set_title("Speedup")
    plt.savefig(folder / "speedup.png", dpi=150)

    fig, ax = plt.subplots()

    ax.bar(num_workers, compute, width, label="Compute")
    ax.bar(num_workers, communicate, width, label="Communication")
    ax.bar(num_workers, metrics, width, label="Metrics Computation")
    ax.bar(num_workers, batch_load, width, label="Batch Loading")

    ax.set_ylabel("Time (s)")
    ax.set_title("Total time by number of workers")
    ax.legend()
    plt.savefig(folder / "time_for_all_phases.png", dpi=150)

    click.echo("Summary created in {}".format(folder))


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


@cli_group.command("list-clusters")
def list_clusters():
    """List all currently configured clusters."""
    config = get_config()

    sections = [
        s.split(".", 1)
        for s in config.sections()
        if s.startswith("gke.") or s.startswith("aws.")
    ]

    if len(sections) == 0:
        click.echo("No clusters are currently configured")
        return

    current_provider = config.get("general", "provider", fallback=None)
    current_cluster = None

    if current_provider == "gke":
        current_cluster = config.get("gke", "current-cluster", fallback=None)
    elif current_provider == "aws":
        current_cluster = config.get("aws", "current-cluster", fallback=None)
    elif current_provider == "kind":
        current_cluster = config.get("kind", "current-cluster", fallback=None)

    gke = [
        name + " *" if current_provider == "gke" and current_cluster == name else name
        for t, name in sections
        if t == "gke"
    ]
    aws = [
        name + " *" if current_provider == "aws" and current_cluster == name else name
        for t, name in sections
        if t == "aws"
    ]
    kind = [
        name + " *" if current_provider == "kind" and current_cluster == name else name
        for t, name in sections
        if t == "kind"
    ]

    message = "Clusters:"

    if gke:
        message += "\n\tGoogle Cloud:\n\t\t{}".format("\n\t\t".join(gke))
    if aws:
        message += "\n\tAWS:\n\t\t{}".format("\n\t\t".join(aws))
    if kind:
        message += "\n\Kind:\n\t\t{}".format("\n\t\t".join(kind))

    click.echo(message)


@cli_group.group("set-cluster")
def set_cluster():
    """Set the current cluster to use."""
    pass


@set_cluster.command("gcloud")
@click.argument("name", type=str)
def set_gcloud_cluster(name):
    """Set current cluster to a gcloud cluster."""
    config = get_config()

    if not config.has_section("gke.{}".format(name)):
        click.echo("Cluster {} not found".format(name))

    config.set("general", "provider", "gke")
    config.set("gke", "current-cluster", name)
    write_config(config)

    click.echo("Ok")


@set_cluster.command("aws")
@click.argument("name", type=str)
def set_aws_cluster(name):
    """Set current cluster to an aws cluster."""
    config = get_config()

    if not config.has_section("aws.{}".format(name)):
        click.echo("Cluster {} not found".format(name))

    config.set("general", "provider", "aws")
    config.set("aws", "current-cluster", name)
    write_config(config)

    click.echo("Ok")


@set_cluster.command("kind")
@click.argument("name", type=str)
def set_kind_cluster(name):
    """Set current cluster to an aws cluster."""
    config = get_config()

    if not config.has_section("kind.{}".format(name)):
        click.echo("Cluster {} not found".format(name))

    config.set("general", "provider", "kind")
    config.set("kind", "current-cluster", name)
    write_config(config)

    click.echo("Ok")


@cli_group.group("delete-cluster")
def delete_cluster():
    """Delete a cluster."""
    pass


@delete_cluster.command("gcloud")
@click.argument("name", type=str)
@click.option("--project", "-p", default=None, type=str)
@click.option("--zone", "-z", default="europe-west1-b", type=str)
def delete_gcloud(name, zone, project):
    gcloud_delete_cluster(name, zone, project)
    delete_gcloud_cluster(name)


@delete_cluster.command("kind")
@click.argument("name", type=str)
def delete_kind(name):
    # Get registry name
    config = get_config()
    registry_name = config.get("kind.{}".format(name), "registry_name")

    # Delete cluster
    kind_delete_cluster(name, registry_name)

    # Delete cluster from config
    delete_kind_cluster(name)

    click.echo("Cluster deleted.")


@delete_cluster.command("aws")
@click.argument("name", type=str)
def delete_aws(name):
    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except botocore.exceptions.ClientError:
        raise click.UsageError(
            "Couldn't find aws credentials. Install the aws"
            " sdk ( https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html ) and "
            "run 'aws configure' to login and create "
            "your credentials."
        )

    # delete nodegroup
    stackName = "eks-auto-scaling-group-" + name

    cloudFormation = boto3.client("cloudformation")
    cloudFormation.delete_stack(StackName=stackName)

    waiter = cloudFormation.get_waiter("stack_delete_complete")
    click.echo("Waiting for nodegroup to be deleted.")
    waiter.wait(StackName=stackName)

    # delete EKS cluster
    eks = boto3.client("eks")
    eks.delete_cluster(name=name)
    waiter = eks.get_waiter("cluster_deleted")
    click.echo("Waiting for cluster to be deleted. This can take up to ten minutes.")
    waiter.wait(name=name)

    # delete VPC stack
    stack_name = name + "-stack"
    cf_client = boto3.client("cloudformation")
    cf_client.delete_stack(StackName=stack_name)
    waiter = cf_client.get_waiter("stack_delete_complete")
    click.echo("Waiting for the VPC stack to be deleted.")
    waiter.wait(StackName=stack_name)

    delete_aws_cluster(name)

    click.echo("Cluster deleted.")


@cli_group.group("create-cluster")
def create_cluster():
    """Create a new cluster."""
    pass


@create_cluster.command("gcloud")
@click.argument("num_workers", type=int, metavar="num-workers")
@click.argument("release", type=str)
@click.option("--kubernetes-version", "-k", type=str, default="1.15")
@click.option("--machine-type", "-t", default="n1-standard-4", type=str)
@click.option("--disk-size", "-d", default=50, type=int)
@click.option("--num-cpus", "-c", default=4, type=int)
@click.option("--num-gpus", "-g", default=0, type=int)
@click.option("--gpu-type", default="nvidia-tesla-k80", type=str)
@click.option("--zone", "-z", default="europe-west1-b", type=str)
@click.option("--project", "-p", default=None, type=str)
@click.option("--preemptible", "-e", is_flag=True)
@click.option("--custom-value", "-v", multiple=True)
@click.option("--chart-location", "-cl", default=None, type=str)
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
    chart_location,
):
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

    name = "{}-{}".format(release, num_workers)
    name_path = "projects/{}/locations/{}/clusters/".format(project, zone)

    click.echo("Creating Cluster")

    gclient, fw_name, firewalls = gcloud_create_cluster(
        name=name,
        name_path=name_path,
        num_workers=num_workers,
        num_gpus=num_gpus,
        gpu_type=gpu_type,
        machine_type=machine_type,
        disk_size=disk_size,
        preemptible=preemptible,
        kubernetes_version=kubernetes_version,
        project=project,
    )

    cluster = gclient.get_cluster(None, None, None, name=os.path.join(name_path, name))
    kube_context = setup_gcloud_kube_client(
        cluster.endpoint, cluster.name, cluster.zone, project
    )

    if num_gpus > 0:
        deploy_nvidia_daemonset()

    custom_chart = {
        "name": "mlbench-helm",
        "source": {
            "type": "git" if chart_location is None else "directory",
            "location": "https://github.com/mlbench/mlbench-helm"
            if chart_location is None
            else chart_location,
        },
    }

    click.echo("Deploying chart")
    deploy_chart(
        num_workers=num_workers - 1,
        num_gpus=num_gpus,
        num_cpus=num_cpus - 1,
        release_name=name,
        custom_value=custom_value,
        custom_chart=custom_chart,
        kube_context=kube_context,
    )

    # open port in firewall
    mlbench_client = ApiClient(in_cluster=False, load_config=False)
    firewall_body = {
        "name": fw_name,
        "direction": "INGRESS",
        "sourceRanges": "0.0.0.0/0",
        "allowed": [{"IPProtocol": "tcp", "ports": [mlbench_client.port]}],
    }

    firewalls.insert(project=project, body=firewall_body).execute()

    add_gcloud_cluster(name, cluster, project)

    click.echo("MLBench successfully deployed")


@create_cluster.command("aws")
@click.argument("num_workers", type=int, metavar="num-workers")
@click.argument("release", type=str)
@click.option("--kubernetes-version", "-k", type=str, default="1.15")
@click.option("--machine-type", "-t", default="t2.medium", type=str)
@click.option("--num-cpus", "-c", default=1, type=int)
@click.option("--num-gpus", "-g", default=0, type=int)
@click.option("--custom-value", "-v", multiple=True)
@click.option("--ami-id", "-a", default="ami-0ef76ba092ce4e253", type=str)
@click.option("--ssh-key", "-a", default="eksNodeKey", type=str)
def create_aws(
    num_workers,
    release,
    kubernetes_version,
    machine_type,
    num_cpus,
    num_gpus,
    custom_value,
    ami_id,
    ssh_key,
):
    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except botocore.exceptions.ClientError:
        raise click.UsageError(
            "Couldn't find aws credentials. Install the aws"
            " sdk ( https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html ) and "
            "run 'aws configure' to login and create "
            "your credentials."
        )
    name = "{}-{}".format(release, num_workers)
    nodeGroupName = name + "-node-group"

    kube_context, cf_client, stackName, cluster = aws_create_cluster(
        name,
        nodeGroupName,
        num_workers,
        machine_type,
        ssh_key,
        ami_id,
        kubernetes_version,
    )
    kube_config.load_kube_config(context=kube_context)

    if num_gpus > 0:
        deploy_nvidia_daemonset_aws()

    deploy_chart(
        num_workers=num_workers - 1,
        num_gpus=num_gpus,
        num_cpus=num_cpus - 1,
        release_name=name,
        custom_value=custom_value,
        kube_context=kube_context,
    )

    # open port in firewall
    mlbench_client = ApiClient(in_cluster=False, load_config=False)
    mlbench_port = mlbench_client.port
    r = cf_client.describe_stack_resources(
        StackName=stackName, LogicalResourceId="NodeSecurityGroup"
    )
    secGroupId = r["StackResources"][0]["PhysicalResourceId"]

    ec2 = boto3.client("ec2")
    ec2.authorize_security_group_ingress(
        GroupId=secGroupId,
        IpPermissions=[
            {
                "FromPort": mlbench_port,
                "IpProtocol": "tcp",
                "IpRanges": [
                    {
                        "CidrIp": "0.0.0.0/0",
                    },
                ],
                "ToPort": mlbench_port,
            },
        ],
    )
    add_aws_cluster(name, cluster)
    click.echo("MLBench successfully deployed")


@create_cluster.command("kind")
@click.argument("num_workers", type=int, metavar="num-workers")
@click.argument("release", type=str)
@click.option("--chart-location", "-cl", default=None, type=str)
@click.option("--registry_name", "-r", default="kind-registry", type=str)
@click.option("--registry_port", "-p", default="5000", type=str)
@click.option("--host_port", "-h", default="5000", type=str)
@click.option("--num-cpus", "-c", default=1, type=int)
@click.option("--num-gpus", "-g", default=0, type=int)
@click.option("--custom-value", "-v", multiple=True)
@click.option("--kubernetes-version", "-k", type=str, default="1.15")
def create_kind(
    num_workers,
    release,
    chart_location,
    registry_name,
    registry_port,
    host_port,
    num_cpus,
    num_gpus,
    custom_value,
    kubernetes_version,
):
    name = "{}-{}".format(release, num_workers)

    create_local_registry(registry_name, registry_port, host_port)

    kind_create_cluster(
        name=name,
        num_workers=num_workers,
        registry_name=registry_name,
        registry_port=registry_port,
        kubernetes_version=kubernetes_version,
    )

    kube_context = "kind-{}".format(name)
    kube_config.load_kube_config(context=kube_context)

    custom_chart = {
        "name": "mlbench-helm",
        "source": {
            "type": "git" if chart_location is None else "directory",
            "location": "https://github.com/mlbench/mlbench-helm"
            if chart_location is None
            else chart_location,
        },
    }
    click.echo("Deploying chart")
    deploy_chart(
        num_workers=num_workers - 1,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        release_name=name,
        custom_value=custom_value,
        custom_chart=custom_chart,
        kube_context=kube_context,
    )

    add_kind_cluster(name, registry_name)

    click.echo("MLBench successfully deployed")


def get_config_path():
    """Gets the path to the user config."""
    user_dir = user_data_dir("mlbench", "mlbench")
    return os.path.join(user_dir, "mlbench.ini")


def get_config():
    """Get the current users' config."""
    path = get_config_path()

    config = configparser.ConfigParser()

    if os.path.exists(path):
        config.read(path)

    if not config.has_section("general"):
        config.add_section("general")

    if not config.has_section("gke"):
        config.add_section("gke")

    if not config.has_section("aws"):
        config.add_section("aws")

    if not config.has_section("kind"):
        config.add_section("kind")

    return config


def write_config(config):
    """Save a config object."""
    path = get_config_path()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as configfile:
        config.write(configfile)


def add_gcloud_cluster(name, cluster, project):
    """Add a gcloud cluster to config."""
    config = get_config()

    config.set("general", "provider", "gke")
    config.set("gke", "current-cluster", name)

    section = "gke.{}".format(name)

    if not config.has_section(section):
        config.add_section(section)
    config.set(section, "endpoint", cluster.endpoint)
    config.set(section, "name", cluster.name)
    config.set(section, "zone", cluster.zone)
    config.set(section, "project", project)

    write_config(config)


def delete_gcloud_cluster(name):
    """Delete a gcloud cluster from config."""
    config = get_config()
    config.remove_section("gke.{}".format(name))

    if config.get("gke", "current-cluster", fallback=None):
        config.set("gke", "current-cluster", "")

    write_config(config)


def add_aws_cluster(name, cluster):
    """Add an aws cluster to config."""
    config = get_config()

    config.set("general", "provider", "aws")
    config.set("aws", "current-cluster", name)

    section = "aws.{}".format(name)

    if not config.has_section(section):
        config.add_section(section)

    config.set(section, "cluster", cluster["cluster"]["endpoint"])

    write_config(config)


def delete_aws_cluster(name):
    """Delete an aws cluster from config."""
    config = get_config()
    config.remove_section("aws.{}".format(name))

    if config.get("aws", "current-cluster", fallback=None):
        config.set("aws", "current-cluster", "")

    write_config(config)


def add_kind_cluster(name, registry_name):
    """Add an kind cluster to config."""
    config = get_config()

    config.set("general", "provider", "kind")
    config.set("kind", "current-cluster", name)

    section = "kind.{}".format(name)

    if not config.has_section(section):
        config.add_section(section)

    config.set(section, "cluster", name)
    config.set(section, "registry_name", registry_name)

    write_config(config)


def delete_kind_cluster(name):
    """Delete a kind cluster from config."""
    config = get_config()
    config.remove_section("kind.{}".format(name))

    if config.get("kind", "current-cluster", fallback=None):
        config.set("kind", "current-cluster", "")

    write_config(config)


def setup_client_from_config():
    """Setup the current kubernetes config."""
    config = get_config()

    provider = config.get("general", "provider", fallback=None)

    if not provider:
        return False

    if provider == "gke":
        return setup_gke_client_from_config(config)
    if provider == "kind":
        return setup_kind_client_from_config(config)
    if provider == "aws":
        return setup_aws_client_from_config(config)

    else:
        raise NotImplementedError()


def setup_kind_client_from_config(config):
    """Setup a kubernrtes cluster for kind from current config."""

    cluster = config.get("kind", "current-cluster", fallback=None)
    if not cluster:
        raise click.UsageError(
            "No kind cluster selected, create a new one with `mlbench create-cluster`"
            " or set one with `mlbench set-cluster`"
        )

    cluster = config.get("kind.{}".format(cluster), "cluster", fallback=None)
    if not cluster:
        return False

    kube_config.load_kube_config(context="kind-{}".format(cluster))
    return True


def setup_gke_client_from_config(config):
    """Setup a kubernetes cluster for gke from current config."""
    cluster = config.get("gke", "current-cluster", fallback=None)
    if not cluster:
        raise click.UsageError(
            "No gcloud cluster selected, create a new one with `mlbench create-cluster`"
            " or set one with `mlbench set-cluster`"
        )

    section = "gke.{}".format(cluster)
    cluster_endpoint = config.get(section, "endpoint", fallback=None)
    if not cluster:
        return False

    cluster_name = config.get(section, "name", fallback=None)
    if not cluster_name:
        return False

    cluster_zone = config.get(section, "zone", fallback=None)
    if not cluster_zone:
        return False

    project = config.get(section, "project", fallback=None)
    if not project:
        return False

    kube_context = setup_gcloud_kube_client(
        cluster_endpoint, cluster_name, cluster_zone, project
    )
    kube_config.load_kube_config(context=kube_context)
    return True


def setup_aws_client_from_config(config):
    """Setup a kubernetes cluster for aws from current config."""
    name = config.get("aws", "current-cluster", fallback=None)
    if not name:
        raise click.UsageError(
            "No gcloud cluster selected, create a new one with `mlbench create-cluster`"
            " or set one with `mlbench set-cluster`"
        )

    kube_context = setup_aws_kube_client(name)
    kube_config.load_kube_config(context=kube_context)
    return True


if __name__ == "__main__":
    sys.exit(cli_group())  # pragma: no cover
