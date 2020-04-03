import torch.distributed as dist
from .helpers import config_logging
from .helpers import config_pytorch
from .helpers import config_path
from .helpers import Timeit
from .topology import FCGraph

from contextlib import contextmanager

import os
__all__ = ["initialize_backends", "Timeit", "FCGraph"]


@contextmanager
def initialize_backends(
    comm_backend="mpi",
    hosts=None,
    rank=-1,
    logging_level="INFO",
    logging_file="/mlbench.log",
    use_cuda=False,
    seed=None,
    cudnn_deterministic=False,
    ckpt_run_dir="/checkpoints",
    delete_existing_ckpts=False,
):
    """Initializes the backends.

    Sets up logging, sets up pytorch and configures paths
    correctly.

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.

    Returns:
        (:obj:`types.SimpleNamespace`): a global object containing all of the config.
    """

    if not (hasattr(dist, "_initialized") and dist._initialized):
        
        if comm_backend != "mpi":
            hosts = hosts.split(',')
            os.environ["MASTER_ADDR"] = hosts[0]
            os.environ['MASTER_PORT'] = '29500'
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(len(hosts))
        
        dist.init_process_group(comm_backend)

    config_logging(logging_level, logging_file)

    rank, world_size, graph = config_pytorch(use_cuda, seed, cudnn_deterministic)

    config_path(ckpt_run_dir, delete_existing_ckpts)

    yield rank, world_size, graph

    dist.destroy_process_group()
