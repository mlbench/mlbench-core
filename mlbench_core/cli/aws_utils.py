import os
import subprocess
from os.path import expanduser
from urllib import request

import boto3
import botocore
import click
import yaml
from kubernetes import client as kube_client
from kubernetes import config as kube_config
from kubernetes.client.rest import ApiException
from kubernetes.config.kube_config import KUBE_CONFIG_DEFAULT_LOCATION

CF_TEMPLATE_URL = "https://amazon-eks.s3-us-west-2.amazonaws.com/cloudformation/2020-08-12/amazon-eks-vpc-sample.yaml"
AWS_NVIDIA_DAEMONSET = "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml"


def _create_ssh_key(ssh_key):

    # create ssh key

    ec2 = boto3.client("ec2")
    try:
        keypair = ec2.create_key_pair(KeyName=ssh_key)
        file_location = expanduser("~") + "/.ssh/{}.pem".format(ssh_key)
        click.echo("Writing ssh key to " + file_location)
        with open(file_location, "w") as file:
            file.write(keypair["KeyMaterial"])
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "InvalidKeyPair.Duplicate":

            # the key has already been created

            pass
        else:
            click.UsageError(error)


def _create_eks_nodegroup(
    name,
    num_workers,
    machine_type,
    nodeGroupName,
    ssh_key,
    ami_id,
    secGroupId,
    vpcId,
    subnets,
):

    params = [
        {"ParameterKey": "KeyName", "ParameterValue": ssh_key},
        {"ParameterKey": "NodeImageId", "ParameterValue": ami_id},
        {"ParameterKey": "NodeInstanceType", "ParameterValue": machine_type},
        {
            "ParameterKey": "NodeAutoScalingGroupMinSize",
            "ParameterValue": str(num_workers),
        },
        {
            "ParameterKey": "NodeAutoScalingGroupMaxSize",
            "ParameterValue": str(num_workers),
        },
        {
            "ParameterKey": "NodeAutoScalingGroupDesiredCapacity",
            "ParameterValue": str(num_workers),
        },
        {"ParameterKey": "ClusterName", "ParameterValue": name},
        {"ParameterKey": "NodeGroupName", "ParameterValue": nodeGroupName},
        {
            "ParameterKey": "ClusterControlPlaneSecurityGroup",
            "ParameterValue": secGroupId,
        },
        {"ParameterKey": "VpcId", "ParameterValue": vpcId},
        {"ParameterKey": "Subnets", "ParameterValue": ",".join(subnets)},
    ]

    stackName = "eks-auto-scaling-group-" + name
    templateURL = "https://amazon-eks.s3.us-west-2.amazonaws.com/cloudformation/2020-08-12/amazon-eks-nodegroup.yaml"

    cloudFormation = boto3.client("cloudformation")
    cloudFormation.create_stack(
        StackName=stackName,
        TemplateURL=templateURL,
        Parameters=params,
        Capabilities=["CAPABILITY_IAM"],
    )

    waiter = cloudFormation.get_waiter("stack_create_complete")
    click.echo("Waiting for nodegroup to be created.")
    waiter.wait(StackName=stackName)

    resource = boto3.resource("cloudformation")
    stack = resource.Stack(stackName)

    nodeInstanceRole = None
    for i in stack.outputs:
        if i["OutputKey"] == "NodeInstanceRole":
            nodeInstanceRole = i["OutputValue"]

    if nodeInstanceRole is None:
        raise ValueError("Could not get NodeInstanceRole")
    return (nodeInstanceRole, stackName)


def _create_config_map(nodeInstanceRole):

    # create config map to connect the nodes to the cluster

    v1 = kube_client.CoreV1Api()
    namespace = "kube-system"
    configMapName = "aws-auth"

    # Delete old config maps in case it exists

    body = kube_client.V1DeleteOptions()
    body.api_version = "v1"
    try:
        v1.delete_namespaced_config_map(
            name="aws-auth", namespace="kube-system", body=body
        )
    except ApiException as e:
        pass

    # Create new config map

    body = kube_client.V1ConfigMap()
    body.api_version = "v1"
    body.metadata = {}
    body.metadata["name"] = configMapName
    body.metadata["namespace"] = namespace

    body.data = {}
    body.data["mapRoles"] = (
        "- rolearn: "
        + nodeInstanceRole
        + """
  username: system:node:{{EC2PrivateDNSName}}
  groups:
    - system:bootstrappers
    - system:nodes
"""
    )

    response = v1.create_namespaced_config_map(namespace, body)


def aws_create_cluster(
    name,
    nodeGroupName,
    num_workers,
    machine_type,
    ssh_key,
    ami_id,
    kubernetes_version,
):

    assert num_workers >= 2, "Number of workers should be at least 2."

    # create VPC stack for cluster

    stack_name = name + "-stack"
    cf_client = boto3.client("cloudformation")
    cf_client.create_stack(StackName=stack_name, TemplateURL=CF_TEMPLATE_URL)
    waiter = cf_client.get_waiter("stack_create_complete")
    click.echo("Waiting for the VPC stack to be created.")

    # will throw an exception if the creation fails

    waiter.wait(StackName=stack_name)

    # obtain vpc id, security group id and subnet ids

    r = cf_client.describe_stack_resources(
        StackName=stack_name, LogicalResourceId="VPC"
    )
    vpcId = r["StackResources"][0]["PhysicalResourceId"]

    r = cf_client.describe_stack_resources(
        StackName=stack_name, LogicalResourceId="ControlPlaneSecurityGroup"
    )
    secGroupId = r["StackResources"][0]["PhysicalResourceId"]

    ec2 = boto3.resource("ec2")
    vpc = ec2.Vpc(vpcId)
    subnets = [subnet.id for subnet in vpc.subnets.all()]

    ec2 = boto3.client("ec2")

    # ensure that public IP addresses are assigned on launch in each subnet

    for subnet in subnets:
        ec2.modify_subnet_attribute(
            MapPublicIpOnLaunch={"Value": True}, SubnetId=subnet
        )

    # create a role and assign the policy needed for creating the EKS cluster

    iam = boto3.client("iam")
    try:
        assume_role_policy_document = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "eks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }"""
        iam.create_role(
            RoleName="EKSClusterRole",
            AssumeRolePolicyDocument=assume_role_policy_document,
        )
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "EntityAlreadyExists":

            # the role has already been created

            pass
        else:
            click.UsageError(error)

    iam.attach_role_policy(
        RoleName="EKSClusterRole",
        PolicyArn="arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
    )

    roleArn = iam.get_role(RoleName="EKSClusterRole")["Role"]["Arn"]

    # create the EKS cluster

    eks = boto3.client("eks")
    eks.create_cluster(
        name=name,
        version=kubernetes_version,
        roleArn=roleArn,
        resourcesVpcConfig={"subnetIds": subnets, "securityGroupIds": [secGroupId]},
    )
    waiter = eks.get_waiter("cluster_active")
    click.echo("Waiting for cluster to be created. This can take up to ten minutes.")
    waiter.wait(name=name)

    (kube_context, cluster) = _create_kube_config_aws_entry(name)
    kube_config.load_kube_config(context=kube_context)

    # Create ssh key

    _create_ssh_key(ssh_key)

    # Create eks nodegroup

    (nodeInstanceRole, stackName) = _create_eks_nodegroup(
        name,
        num_workers,
        machine_type,
        nodeGroupName,
        ssh_key,
        ami_id,
        secGroupId,
        vpcId,
        subnets,
    )

    _create_config_map(nodeInstanceRole)

    return (kube_context, cf_client, stackName, cluster)


def _create_kube_config_aws_entry(name):
    eks = boto3.client("eks")
    cluster = eks.describe_cluster(name=name)
    cluster_cert = cluster["cluster"]["certificateAuthority"]["data"]
    cluster_ep = cluster["cluster"]["endpoint"]
    cluster_arn = cluster["cluster"]["arn"]

    cluster_config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "cluster": {
                    "server": str(cluster_ep),
                    "certificate-authority-data": str(cluster_cert),
                },
                "name": cluster_arn,
            }
        ],
        "contexts": [
            {
                "context": {"cluster": cluster_arn, "user": cluster_arn},
                "name": cluster_arn,
            }
        ],
        "current-context": cluster_arn,
        "preferences": {},
        "users": [
            {
                "name": cluster_arn,
                "user": {
                    "exec": {
                        "apiVersion": "client.authentication.k8s.io/v1alpha1",
                        "command": "aws-iam-authenticator",
                        "args": ["token", "-i", name],
                    }
                },
            }
        ],
    }

    config_text = yaml.dump(cluster_config, default_flow_style=False)
    default_config = os.path.abspath(os.path.expanduser(KUBE_CONFIG_DEFAULT_LOCATION))
    config_dir = os.path.dirname(default_config)
    config_file = os.path.join(config_dir, "mlbench_aws")

    with open(config_file, "w") as f:
        f.write(config_text)

    shell_env = os.environ.copy()
    shell_env["KUBECONFIG"] = "{}:{}".format(default_config, config_file)

    with open(default_config, "w") as out:
        p = subprocess.Popen(
            ["kubectl", "config", "view", "--flatten"],
            stdout=out,
            stderr=subprocess.PIPE,
            env=shell_env,
        )
        p.communicate()

    return (cluster_arn, cluster)


def deploy_nvidia_daemonset_aws():
    """ Deploys the NVIDIA daemon set to the AWS cluster"""
    with request.urlopen(AWS_NVIDIA_DAEMONSET) as r:
        dep = yaml.safe_load(r)
        dep["spec"]["selector"] = {
            "matchLabels": dep["spec"]["template"]["metadata"]["labels"]
        }
        dep = kube_client.ApiClient()._ApiClient__deserialize(dep, "V1DaemonSet")
        k8s_client = kube_client.AppsV1Api()
        k8s_client.create_namespaced_daemon_set("kube-system", body=dep)


def setup_aws_kube_client(name):

    # Keeping this function so we can later configure the python kube client directly

    (kube_context, _) = _create_kube_config_aws_entry(name)
    return kube_context
