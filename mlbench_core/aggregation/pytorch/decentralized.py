import torch
import torch.distributed as dist

from mlbench_core.aggregation.pytorch.aggregation import Aggregation


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, neighbors, use_cuda=False):
        """
        Args:
            rank (int): Rank of the current process
            neighbors (list): A list of ranks of its neighbors.
        """
        assert rank not in neighbors
        self.rank = rank
        self.neighbors = neighbors
        super(DecentralizedAggregation, self).__init__(use_cuda=use_cuda)

    def _agg(self, data, op, denom=None):
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
        if op == "avg_world":
            output = sum(local_data.values()) / (len(self.neighbors) + 1)
        else:
            raise NotImplementedError("op {} is not supported yet.".format(op))

        return output
