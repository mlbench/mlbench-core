import os
import subprocess
import tempfile
from time import sleep

import click
import docker

KIND_NODES_IMAGES = {
    "1.15": "kindest/node:v1.15.12@sha256:d9b939055c1e852fe3d86955ee24976cab46cba518abcb8b13ba70917e6547a6",
    "1.16": "kindest/node:v1.16.15@sha256:a89c771f7de234e6547d43695c7ab047809ffc71a0c3b65aa54eda051c45ed20",
    "1.17": "kindest/node:v1.17.11@sha256:5240a7a2c34bf241afb54ac05669f8a46661912eab05705d660971eeb12f6555",
    "1.18": "kindest/node:v1.18.8@sha256:f4bcc97a0ad6e7abaf3f643d890add7efe6ee4ab90baeb374b4f41a4c95567eb",
    "1.19": "kindest/node:v1.19.1@sha256:98cf5288864662e37115e362b23e4369c8c4a408f99cbc06e58ac30ddc721600",
}
KIND_CONFIG = """
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:{reg_port}"]
    endpoint = ["http://{reg_name}:{reg_port}"]
nodes:
- role: control-plane
  image: {kind_node_image}
{workers_config}
"""


def create_local_registry(registry_name, registry_port, host_port):
    """Creates a local docker registry

    Args:
        registry_name (str): Name of registry
        registry_port (str): Port registry
        host_port (str): Host port to bind registry to
    """
    docker_client = docker.from_env()

    # check if local registry exists
    existing_containers = [
        container.name for container in docker_client.containers.list()
    ]
    registry_exists = registry_name in existing_containers

    if not registry_exists:
        # create local registry
        click.echo("Creating registry {}".format(registry_name))
        docker_client.containers.run(
            image="registry:2",
            name=registry_name,
            restart_policy={"Name": "always"},
            ports={registry_port: host_port},
            detach=True,
        )
        while docker_client.containers.get(registry_name).status != "running":
            if docker_client.containers.get(registry_name).status == "exited":
                raise click.UsageError("Failed to create local registry")
            sleep(1)


def kind_create_cluster(
    name, num_workers, registry_name, registry_port, kubernetes_version
):
    """Creates a local KIND cluster with the given name and connects it to the running local registry

    Args:
        name (str): Cluster name
        num_workers (int): Total number of nodes (including master node)
        registry_name (str): Local registry name
        registry_port (str): Local registry port
        kubernetes_version (str): Kubernetes version to use (available :1.15, 1.16, 1.17, 1.18, 1.19)
    """
    # create cluster
    with tempfile.TemporaryDirectory() as temp_directory:
        kind_config_file_location = os.path.join(temp_directory, "kind_config.yml")

        with open(kind_config_file_location, "w") as f:
            workers_config = "\n".join(
                ["- role: worker\n  image: {kind_node_image}"] * (num_workers - 1)
            )
            node_image = KIND_NODES_IMAGES[kubernetes_version]
            f.write(
                KIND_CONFIG.format(
                    reg_port=registry_port,
                    reg_name=registry_name,
                    workers_config=workers_config.format(kind_node_image=node_image),
                    kind_node_image=node_image,
                )
            )

        click.echo("Creating cluster {}".format(name))

        p = subprocess.Popen(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                name,
                "--config",
                kind_config_file_location,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output, error = p.communicate()
        if p.returncode != 0:
            raise click.UsageError(
                "Failed to create cluster with the following error:\n {}".format(
                    error.decode()
                )
            )

    _connect_to_local_registry(registry_name)


def kind_delete_cluster(name, registry_name):
    """Deletes the cluster using the kind command `kind delete cluster`
    Also disconnects from the local registry

    Args:
        name (str): Cluster name
        registry_name (str): Registry name
    """
    # Delete cluster using kind
    p = subprocess.Popen(
        ["kind", "delete", "cluster", "--name", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    _, error = p.communicate()
    if p.returncode != 0:
        raise click.UsageError(
            "Failed to delete cluster with the following error:\n {}".format(
                error.decode()
            )
        )

    _disconnect_from_local_registry(registry_name)


def _connect_to_local_registry(registry_name):
    """Connects the kind network in docker to the given registry name

    Args:
        registry_name (str): Local registry name
    """
    docker_client = docker.from_env()
    # Connect local registry to kind cluster
    kind_network = [x for x in docker_client.networks.list() if x.name == "kind"]
    if len(kind_network) == 0:
        raise ValueError("Kind network not found")
    kind_network = kind_network[0]
    kind_network.connect(registry_name)


def _disconnect_from_local_registry(registry_name):
    """Disconnects the kind network in docker to the given registry name

    Args:
        registry_name (str): Local registry name
    """
    # Disconnect from local registry
    docker_client = docker.from_env()

    kind_network = [x for x in docker_client.networks.list() if x.name == "kind"]
    if len(kind_network) == 0:
        raise ValueError("Kind network not found")
    kind_network = kind_network[0]

    kind_network.disconnect(registry_name)
