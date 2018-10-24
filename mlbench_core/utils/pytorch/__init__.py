import torch.distributed as dist
from .helpers import config_logging
from .helpers import config_pytorch
from .helpers import config_path
from .helpers import Timeit
from .topology import FCGraph

__all__ = ['initialize_backends', 'Timeit', 'FCGraph']


def initialize_backends(config):
    """Initializes the backends.

    Sets up logging, sets up pytorch and configures paths
    correctly.

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.

    Returns:
        (:obj:`types.SimpleNamespace`): a global object containing all of the config.
    """

    if not (hasattr(dist, '_initialized') and dist._initialized):
        dist.init_process_group(config.comm_backend)

    config_logging(config)

    config_pytorch(config)

    config_path(config)

    return config
