import socket

import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch.distributed import get_backend_tensor


def _ranks_on_same_node(rank, world_size):
    hostname = socket.gethostname()
    hostname_length = get_backend_tensor(torch.IntTensor([len(hostname)]))

    dist.all_reduce(hostname_length, op=dist.ReduceOp.MAX)
    max_hostname_length = hostname_length.item()

    encoding = [ord(c) for c in hostname]
    encoding += [-1 for c in range(max_hostname_length - len(hostname))]
    encoding = get_backend_tensor(torch.IntTensor(encoding))

    all_encodings = [
        get_backend_tensor(torch.IntTensor([0] * max_hostname_length))
        for _ in range(world_size)
    ]
    dist.all_gather(all_encodings, encoding)

    if dist.get_backend() == dist.Backend.NCCL:
        all_encodings = [ec.cpu() for ec in all_encodings]

    all_encodings = [ec.numpy().tolist() for ec in all_encodings]

    ranks = []
    for i in range(world_size):
        if all_encodings[rank] == all_encodings[i]:
            ranks.append(i)
    return ranks


class FCGraph(object):
    """Fully-Connected Network Graph

    Args:
        config (dict): a global object containing all of the config.
    """

    def __init__(self, rank, world_size, use_cuda=False):
        self.rank = rank
        self.world_size = world_size
        self.use_cuda = use_cuda

    @property
    def current_device_name(self):
        return "cuda:{}".format(torch.cuda.current_device()) if self.use_cuda else "cpu"

    @property
    def current_device(self):
        return torch.device(self.current_device_name())

    def assigned_gpu_id(self):
        num_gpus_on_device = torch.cuda.device_count()
        ranks = _ranks_on_same_node(self.rank, self.world_size)
        # raise NotImplementedError(self.rank, ranks)
        assigned_id = ranks.index(self.rank) % num_gpus_on_device
        torch.cuda.set_device(assigned_id)

    def __str__(self):
        return "{}".format(self.current_device_name)

    def __repr__(self):
        return self.__str__()
