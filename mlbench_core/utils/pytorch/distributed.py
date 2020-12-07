import torch
import torch.distributed as dist


def global_average(sum, count):
    def helper(array):
        array = get_backend_tensor(torch.Tensor(array))

        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        return array[0] / array[1]

    avg = helper([sum, count])
    return avg


def get_backend_tensor(tensor):
    if dist.is_initialized() and dist.get_backend() == dist.Backend.NCCL:
        return tensor.cuda()
    return tensor
