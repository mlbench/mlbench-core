import torch
import torch.distributed as dist


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.reduce_op.MIN)
    return tensor


def aggregate_gradients(model, world_size, average_models=False):
    """Average gradients of models across all processes."""
    # all_reduce the gradients.
    for ind, param in enumerate(model.parameters()):
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)

        if average_models:
            param.grad.data /= world_size


def global_average(sum, count):
    def helper(array):
        array = torch.Tensor(array)
        dist.all_reduce(array, op=dist.reduce_op.SUM)
        return array[0] / array[1]

    avg = helper([sum, count])
    return avg


##########################################################################################


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def _agg(self, data, op):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op):
        """Aggregate models by model weight.

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.data, op=op)
            param.data = grad

    def agg_grad(self, model, op):
        """Aggregate models gradients.

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op)
            param.grad.data = grad


class AllReduceAggregation(Aggregation):
    """Aggregate udpates / models from different processes."""

    def __init__(self, world_size):
        self.world_size = world_size

    def _agg(self, data, op):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if op == "avg":
            dist.all_reduce(data, op=dist.reduce_op.SUM)
            data /= self.world_size
        else:
            raise NotImplementedError
        return data


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, neighbors):
        """
        Args:
            rank (int): Rank of the current process
            neighbors (list): A list of ranks of its neighbors.
        """
        assert rank not in neighbors
        self.rank = rank
        self.neighbors = neighbors

    def _agg(self, data, op):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        local_data = {i: torch.zeros_like(data) for i in self.neighbors}
        local_data[self.rank] = data

        reqs = []
        for node in self.neighbors:
            reqs.append(dist.isend(tensor=local_data[self.rank], dst=node))
            reqs.append(dist.irecv(tensor=local_data[node], src=node))

        for req in reqs:
            req.wait()

        # Aggregate local_data
        if op == "avg":
            output = sum(local_data.values()) / (len(self.neighbors) + 1)
        else:
            raise NotImplementedError("op {} is not supported yet.".format(op))

        return output


class SparsifiedAggregation(Aggregation):
    """Aggregate sparsified updates."""

    def __init__(self, model):
        pass

    def _agg(self, data, op):
        pass
