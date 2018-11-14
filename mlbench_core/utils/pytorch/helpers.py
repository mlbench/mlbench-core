r"""Helper functions."""

import time
import datetime
import itertools
import shutil
import os
import logging
import socket
import random
import argparse
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch import checkpoint
from mlbench_core.utils.pytorch.topology import FCGraph
from mlbench_core.api import ApiClient


class Timeit(object):
    """Training Time Tracker

    Used to track training time for timing comparison

    Args:
        cumu (float, optional): starting time in seconds. Default: ``0.0``

    Example:
        >>> t = Timeit()
        >>> [... Do some training...]
        >>> t.pause()
        >>> [... do some non-training related things ...]
        >>> t.resume()
        >>> print(t.cumu)
    """

    def __init__(self, cumu=0):
        self.t = time.time()
        self._cumu = cumu
        self._paused = False

    def pause(self):
        """ Pause Time Tracking"""
        if not self._paused:
            self._cumu += time.time() - self.t
            self.t = time.time()
            self._paused = True

    def resume(self):
        """ Resume Time Tracking"""
        if self._paused:
            self.t = time.time()
            self._paused = False

    @property
    def cumu(self):
        """ float: total tracked time in seconds"""
        return self._cumu


def maybe_range(maximum):
    """Map an integer or None to an integer iterator starting from 0 with strid 1.

    If maximum number of batches per epoch is limited, then return an finite
    iterator. Otherwise, return an iterator of infinite length.
    """
    if maximum is None:
        counter = itertools.count(0)
    else:
        counter = range(maximum)
    return counter


def update_best_runtime_metric(tracker, metric_value, metric_name):
    """Update the runtime information to config if the metric value is the best."""
    best_metric_name = "best_{}".format(metric_name)
    if best_metric_name in tracker.records:
        is_best = metric_value > tracker.records[best_metric_name]
    else:
        is_best = True

    if is_best:
        tracker.records[best_metric_name] = metric_value
        tracker.records['best_epoch'] = tracker.current_epoch
    return is_best, best_metric_name


def convert_dtype(dtype, obj):
    # The object should be a ``module`` or a ``tensor``
    if dtype == 'fp32':
        return obj.float()
    elif dtype == 'fp64':
        return obj.double()
    else:
        raise NotImplementedError('dtype {} not supported.'.format(dtype))


def config_logging(logging_level='INFO', logging_file='/mlbench.log'):
    """Setup logging modules.

    A stream handler and file handler are added to default logger `mlbench`.
    """

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank()
            return True

    logger = logging.getLogger('mlbench')
    if len(logger.handlers) >= 2:
        return

    logger.setLevel(logging_level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s',
        "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logging_file)
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def config_pytorch(use_cuda=False, seed=None, cudnn_deterministic=False):
    """Config pytorch packages.

    Fix random number for packages and initialize distributed environment for pytorch.
    Setup cuda environment for pytorch.

    :param config: A global object containing specified config.
    :type config: argparse.Namespace
    """
    # Setting `cudnn.deterministic = True` will turn on
    # CUDNN deterministic setting which can slow down training considerably.
    # Unexpected behavior may also be observed from checkpoint.
    # See: https: // github.com/pytorch/examples/blob/master/imagenet/main.py
    if cudnn_deterministic:
        # cudnn.deterministic = True
        print('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

    if seed:
        random.seed(seed)
        torch.manual_seed(seed)

    # define the graph for the computation.
    if use_cuda:
        assert torch.cuda.is_available()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    graph = FCGraph(rank, world_size, use_cuda)

    # enable cudnn accelerator if we are using cuda.
    if use_cuda:
        graph.assigned_gpu_id()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if torch.backends.cudnn.version() is None:
            print("CUDNN not found on device.")

        print("World size={}, Rank={}, hostname={}, cuda_available={}, cuda_device={}".format(
            world_size, rank, socket.gethostname(), torch.cuda.is_available(),
            torch.cuda.current_device()))

    return rank, world_size, graph


def log_metrics(run_id, rank, epoch, metric_name, value):
    """ Log metrics to mlbench master/dashboard

    Args:
        run_id (str): The id of the current run
        rank (int): The rank of the current worker
        epoch (int): The current epoch
        metric_name (str): The name of the metric to save
        value (Any): The metric value
    """
    in_cluster = os.getenv('MLBENCH_IN_DOCKER') is None
    if in_cluster:
        api = ApiClient()
        api.post_metric(
            run_id,
            metric_name,
            value,
            metadata="{{rank: {}, epoch:{}}}".format(rank, epoch))
    else:
        pass


def config_path(ckpt_run_dir, resume=False):
    """Config the path used during the experiments."""

    if resume:
        print("Remove previous checkpoint directory : {}".format(
            ckpt_run_dir))
        shutil.rmtree(ckpt_run_dir, ignore_errors=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)


def iterate_dataloader(dataloader, dtype, max_batch_per_epoch=None, use_cuda=False,
                       transform_target_type=None):
    for _, (data, target) in zip(maybe_range(max_batch_per_epoch),
                                 dataloader):

        data = convert_dtype(dtype, data)
        if transform_target_type:
            target = convert_dtype(dtype, target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        yield data, target


def maybe_cuda(module, use_cuda):
    if use_cuda:
        module.cuda()
    return module
