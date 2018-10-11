import torch.distributed as dist

# TODO: change the backend of broadcast/... based on pytorch/tensorflow/...


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)
