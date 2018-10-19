# -*- coding: utf-8 -*-

"""Top-level package for mlbench_core."""

__version__ = '0.1.0'

# FIXME: remove

import torch.distributed as dist


def red_print(m):
    if dist.get_rank() == 0:
        print("\033[0;31m{}\033[0m".format(m))
