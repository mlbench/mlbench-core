from time import sleep

import yaml
from pyhelm.chartbuilder import ChartBuilder
from pyhelm.tiller import Tiller

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


def create_tiller(client):
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


def deploy_tiller(client):
    tiller_service = yaml.safe_load(TILLER_MANIFEST_SERVICE)
    tiller_dep = yaml.safe_load(TILLER_MANIFEST_DEPLOYMENT)
    client.CoreV1Api().create_namespaced_service("kube-system", tiller_service)
    client.ExtensionsV1beta1Api().create_namespaced_deployment(
        "kube-system", tiller_dep
    )

    sleep(5)

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
    return tiller_pod


def deploy_chart(
    num_workers, num_gpus, num_cpus, release_name, custom_value, custom_chart=None
):
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

    tiller.install_release(
        chart.get_helm_chart(),
        name=release_name,
        wait=True,
        dry_run=False,
        namespace="default",
        values=values,
    )
