r"""Helper functions."""

import logging
import os
import random
import shutil
import socket

import numpy as np
import torch
from torch import distributed as dist

from mlbench_core.utils.pytorch.topology import FCGraph


def config_logging(logging_level="INFO", logging_file="/mlbench.log"):
    """Setup logging modules.
    A stream handler and file handler are added to default logger `mlbench`.

    Args:
        logging_level (str): Log level
        logging_file (str): Log file

    """

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank()
            return True

    logger = logging.getLogger("mlbench")
    if len(logger.handlers) >= 2:
        return

    logger.setLevel(logging_level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(rank)2s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

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

    Args:
        use_cuda (bool): Use CUDA acceleration
        seed (int | None): Random seed to use
        cudnn_deterministic (bool): Set `cudnn.determenistic=True`

    Returns:
        (int, int, `obj`:FCGraph): The rank, world size, and network graph
    """
    # Setting `cudnn.deterministic = True` will turn on
    # CUDNN deterministic setting which can slow down training considerably.
    # Unexpected behavior may also be observed from checkpoint.
    # See: https: // github.com/pytorch/examples/blob/master/imagenet/main.py
    if cudnn_deterministic:
        # cudnn.deterministic = True
        print(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # define the graph for the computation.
    if use_cuda:
        assert torch.cuda.is_available()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    backend = dist.get_backend() if dist.is_initialized() else None
    graph = FCGraph(rank, world_size, use_cuda)

    # enable cudnn accelerator if we are using cuda.
    if use_cuda:
        graph.assigned_gpu_id()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = True

        if torch.backends.cudnn.version() is None:
            print("CUDNN not found on device.")

        print(
            "World size={}, Rank={}, hostname={}, backend={}, cuda_available={}, cuda_device={}".format(
                world_size,
                rank,
                socket.gethostname(),
                backend,
                torch.cuda.is_available(),
                torch.cuda.current_device(),
            )
        )

    return rank, world_size, graph


def config_path(ckpt_run_dir, delete_existing_ckpts=False):
    """Config the path used during the experiments."""
    if delete_existing_ckpts:
        print("Remove previous checkpoint directory : {}".format(ckpt_run_dir))
        shutil.rmtree(ckpt_run_dir, ignore_errors=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
