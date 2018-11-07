import torch.distributed as dist
from .helpers import config_logging
from .helpers import config_pytorch
from .helpers import config_path
from .helpers import Timeit
from .topology import FCGraph

__all__ = ['initialize_backends', 'Timeit', 'FCGraph']


def initialize_backends(comm_backend='mpi', logging_level='INFO',
                        logging_file='/mlbench.log', use_cuda=False,
                        seed=None, cudnn_deterministic=False,
                        ckpt_run_dir='/checkpoints', resume=False):
    """Initializes the backends.

    Sets up logging, sets up pytorch and configures paths
    correctly.

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.

    Returns:
        (:obj:`types.SimpleNamespace`): a global object containing all of the config.
    """

    if not (hasattr(dist, '_initialized') and dist._initialized):
        dist.init_process_group(comm_backend)

    config_logging(logging_level, logging_file)

    rank, world_size, graph = config_pytorch(use_cuda, seed,
                                             cudnn_deterministic)

    config_path(ckpt_run_dir, resume)

    return rank, world_size, graph
