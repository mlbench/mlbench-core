import torch.distributed as dist
from .helpers import config_logging
from .helpers import config_pytorch
from .helpers import config_path

__all__ = ['initialize_backends']


def initialize_backends(config):
    if not (hasattr(dist, '_initialized') and dist._initialized):
        dist.init_process_group(config.comm_backend)

    config_logging(config)

    config_pytorch(config)

    config_path(config)

    return config
