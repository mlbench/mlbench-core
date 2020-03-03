import datetime
import os

from mlbench_core.api import ApiClient


class LogMetrics(object):
    """ Use to write metric values to the Dashboard API and to Trackers

    Caches API client for performance reasons
    """

    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None

    if in_cluster:
        api = ApiClient()

    @staticmethod
    def log(run_id, rank, epoch, metric_name, value):
        """ Logs metrics to the Metrics API

        Currently only logs inside of a cluster

        Args:
            run_id (str): The id of the run in the dashboard
            rank (int): Rank of the current worker node
            epoch (float): The current epoch (fractional)
            metric_name (str): The name of the metric
            value (float / int / str): The metric value to write
            tracker(:obj:`mlbench_core.utils.Tracker`): The value Tracker
            time (float): The current time (used for Tracker)

        """

        if not LogMetrics.in_cluster:
            return

        metric_name = "{} @ {}".format(metric_name, rank)

        LogMetrics.api.post_metric(
            run_id,
            metric_name,
            value,
            metadata="{{rank: {}, epoch:{}}}".format(rank, epoch),
        )
