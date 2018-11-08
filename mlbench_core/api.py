""" MLBench Master/Dashboard API Client Functionality """

import requests
import concurrent.futures
import datetime
import logging

from kubernetes import config, client

OFFICIAL_IMAGES = {
    'Test Image': 'mlbench/mlbench_worker'
}
r"""
Dict of official benchmark images

Note:
    Format: ``{name: image_name}``"""


class ApiClient(object):
    """ Client for the mlbench Master/Dashboard REST API

    When used inside a cluster, will use the API Pod IP directly
    for communication. When used outside of a cluster, will try to
    figure out how to access the API depending on the K8s service
    type, if it's accessible. Endpoint URL can also be set manually.

    All requests are executed in a separate process to ensure
    non-blocking execution. Results are returned as
    ``concurrent.futures.Future`` objects wrapping ``requests``
    responses.

    Expects K8s credentials to be set correctly (automatic inside
    a cluster, through kubectl outside of it)

    Args:
        max_workers (int): maximum number of processes to run in parallel
        in_cluster (bool): Whether the client is run inside the K8s cluster
            or not
        label_selector (str): K8s label selectors to find the master pod
            when running inside a cluster. Default:
            ``component=master,app=mlbench``
        k8s_namespace (str): K8s namespace mlbench is running in.
            Default: ``default``
        service_name (str): Name of the master service, usually something
            like ``release-mlbench-master``. Only needed when running
            outside of a cluster. Default: ``None``
        url (str): ip:port/path or hostname:port/path that overrides
            automatic endpoint detection, pointing to the root of the
            master/dashboard node. Default: ``None``"""

    def __init__(self, max_workers=5, in_cluster=True,
                 label_selector='component=master,app=mlbench',
                 k8s_namespace='default', service_name=None,
                 url=None):
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers)
        self.in_cluster = in_cluster

        if url is not None:
            self.endpoint = "http://{url}/api/".format(url=url)
            return

        class _CustomApiClient(client.ApiClient):
            """
            A bug introduced by a fix.
            https://github.com/kubernetes-client/python/issues/411
            https://github.com/swagger-api/swagger-codegen/issues/6392
            """

            def __del__(self):
                pass

        if self.in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()

        configuration = client.Configuration()
        k8s_client = client.CoreV1Api(_CustomApiClient(configuration))
        if self.in_cluster:
            namespaced_pods = k8s_client.list_namespaced_pod(
                k8s_namespace, label_selector=label_selector)

            assert len(namespaced_pods.items) == 1

            master_pod = namespaced_pods.items[0]
            url = master_pod.status.pod_ip + ":80"
        else:
            # figure out ip and port based on service type
            if service_name is None:
                raise ValueError("'service_name' must be set for "
                                 "out of cluster use")
            service = k8s_client.read_namespaced_service(
                service_name,
                k8s_namespace)
            print(service)
            service_type = service.spec.type

            if service_type == "ClusterIP":
                # cluster ip: up to user to grant access
                ip = service.spec.cluster_ip
                port = service.spec.ports[0].port
            elif service_type == "NodePort":
                port = service.spec.ports[0].node_port
                if (service.spec.external_i_ps
                        and len(service.spec.external_i_ps) > 0):

                    ip = service.spec.external_i_ps[0]
                else:
                    # try and get public node ip
                    namespaced_pods = k8s_client.list_namespaced_pod(
                        k8s_namespace, label_selector=label_selector)

                    assert len(namespaced_pods.items) == 1

                    master_pod = namespaced_pods.items[0]
                    node_name = master_pod.spec.node_name
                    node = k8s_client.read_node(node_name)
                    ip = None
                    internal_ip = None
                    for address in node.status.addresses:
                        if address.type == "ExternalIP":
                            ip = address.address
                            break
                        if address.type == "InternalIP":
                            internal_ip = address.address
                    if not ip:
                        # no external IP found, maybe internal works
                        logging.warning(
                            "No ExternalIP found for NodePort "
                            "Service. Trying InternalIP. This only works "
                            "if the cluster internal IP's are reachable.")
                        ip = internal_ip
            elif service_type == "LoadBalancer":
                port = service.spec.ports[0].port
                if service.status.load_balancer.ingress is None:
                    raise NotImplementedError(
                        "Service Type 'LoadBalancer' only works with an "
                        "'Ingress' ip defined"
                    )
                ip = service.status.load_balancer.ingress.ip
            else:
                raise NotImplementedError

            url = "{ip}:{port}".format(ip=ip, port=port)

        self.endpoint = "http://{url}/api/".format(url=url)

    def get_all_metrics(self):
        """ Get all metrics ever recorded by the master node.

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        request_url = "{endpoint}metrics/".format(endpoint=self.endpoint)
        future = self.executor.submit(
            requests.get,
            request_url)

        return future

    def get_run_metrics(self, run_id, since=None, summarize=None):
        """ Get all metrics for a run.

        Args:
            run_id(str): The id of the run to get metrics for
            since (datetime): Only get metrics newer than this date
                Default: ``None``
            summarize (int): If set, metrics are summarized to at most this
            many entries by averaging the metrics. Default: ``None``

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        return self._get_filtered_metrics(run_id=run_id, since=since,
                                          summarize=summarize)

    def get_pod_metrics(self, pod_id, since=None, summarize=None):
        """ Get all metrics for a worker pod.

        Args:
            pod_id(str): The id of the pod to get metrics for
            since (datetime): Only get metrics newer than this date
                Default: ``None``
            summarize (int): If set, metrics are summarized to at most this
            many entries by averaging the metrics. Default: ``None``

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        return self._get_filtered_metrics(pod_id=pod_id, since=since,
                                          summarize=summarize)

    def _get_filtered_metrics(self, pod_id=None, run_id=None, since=None,
                              summarize=None):
        """Get metrics for a run or pod"""
        if pod_id is None and run_id is None:
            raise ValueError("Either pod_id or pod_id must be specified")

        pk = pod_id
        metric_type = "pod"

        if pk is None:
            pk = run_id
            metric_type = "run"

        params = {
            'metric_type': metric_type
        }

        if since is not None:
            params['since'] = str(since)

        if summarize is not None:
            params['summarize'] = str(summarize)

        request_url = "{endpoint}metrics/{pk}".format(
            endpoint=self.endpoint,
            pk=pk)

        future = self.executor.submit(
            requests.get,
            request_url,
            params=params)

        return future

    def post_metric(self, run_id, name, value, cumulative=False,
                    metadata="", date=None):
        """ Save a metric to the master node for a run.

        Args:
            run_id(str): The id of the run to save a metric for
            name (str): The name of the metric, e.g. ``accuracy``
            value (Number): The metric value to save
            cumulative (bool, optional): Whether this metric is
                cumulative or not. Cumulative metrics are values
                that increment over time, i.e. ``current_calue =
                previous_value + value_difference``. Non-cumulative
                values or discrete values at a certain time. Default: ``False``
            metadata (dict): Optional metadata to attach to a metric.
                Default: ``None``
            date (datetime): The date the metric was gathered.
                Default: ``datetime.now``

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        if date is None:
            date = datetime.datetime.now()

        request_url = "{endpoint}metrics/".format(endpoint=self.endpoint)
        future = self.executor.submit(
            requests.post,
            request_url,
            data={
                "run_id": run_id,
                "name": name,
                "cumulative": cumulative,
                "date": str(date),
                "value": str(value),
                "metadata": metadata
             })
        return future

    def get_runs(self):
        """ Get all active, finished and failed benchmark runs

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        request_url = "{endpoint}runs/".format(endpoint=self.endpoint)
        future = self.executor.submit(
            requests.get,
            request_url)

        return future

    def get_run(self, run_id):
        """ Get a specific benchmark run

        Args:
            run_id(str): The id of the run to get

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        request_url = "{endpoint}runs/{run_id}".format(
            endpoint=self.endpoint,
            run_id=run_id)
        future = self.executor.submit(
            requests.get,
            request_url)

        return future

    def create_run(self, name, num_workers, num_cpus=2.0, max_bandwidth=1000,
                   image=None, custom_image_name=None,
                   custom_image_command=None, custom_image_all_nodes=False):
        """ Create a new benchmark run.

        Available official benchmarks can be found in
        the ``mlbench_core.api.OFFICIAL_IMAGES`` dict.

        Args:
            name (str): The name of the run
            num_workers (int): The number of worker nodes to use
            num_cpus (float): The number of CPU Cores per worker to utilize.
                Default: ``2.0``
            max_bandwidth (int): Maximum bandwidth available for
                communication between worker nodes in mbps. Default: ``1000``
            image (str): Name of the official benchmark image to use (
                see ``mlbench_core.api.OFFICIAL_IMAGES`` keys).
                Default: ``None``
            custom_image_name (str): The name of a custom Docker image
                to run. Can be a dockerhub or private Docker repository url.
                Default: ``None``
            custom_image_command (str): Command to run on the custom image.
                Default: ``None``
            custom_image_all_nodes (bool): Whether to run
                ``custom_image_command`` on all worker nodes or only the
                rank 0 node.

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        request_url = "{endpoint}runs/".format(endpoint=self.endpoint)
        data = {
                "name": name,
                "num_workers": num_workers,
                "num_cpus": num_cpus,
                "max_bandwidth": max_bandwidth
             }

        if custom_image_name is not None:
            data['image_name'] = 'custom_image'
            data['custom_image_name'] = custom_image_name
            data['custom_image_command'] = custom_image_command
            data['custom_image_all_nodes'] = custom_image_all_nodes
        elif image in OFFICIAL_IMAGES:
            data['image_name'] = OFFICIAL_IMAGES[image]
        else:
            raise ValueError("Image {image} not found".format(image=image))

        future = self.executor.submit(
            requests.post,
            request_url,
            data=data
            )
        return future

    def get_worker_pods(self):
        """ Get information on all worker nodes.

        Returns:
            A ``concurrent.futures.Future`` objects wrapping
            ``requests.response`` object. Get the result by calling
            ``return_value.result().json()``
        """
        request_url = "{endpoint}pods/".format(endpoint=self.endpoint)
        future = self.executor.submit(
            requests.get,
            request_url)

        return future

    def __enter__(self):
        self.executor.__enter__()
        return self

    def __exit__(self, *args):
        return self.executor.__exit__(*args)