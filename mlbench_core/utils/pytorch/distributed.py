import numpy as np
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch.utils import orthogonalize

try:
    import horovod.torch as hvd
except ImportError as e:
    pass

ALLREDUCE_AGGREGATION_OPS = ["avg_world", "custom_avg"]
"""
All possible aggregations for AllReduceAggregation
"""


def global_average(sum, count):
    def helper(array):
        array = get_backend_tensor(torch.Tensor(array))

        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        return array[0] / array[1]

    avg = helper([sum, count])
    return avg


def pack_tensors(tensors, use_cuda=False):
    """
    Packs a list of tensors into one 1-dimensional tensor.

    Args:
        tensors (list[torch.Tensor]): The tensors to pack
        use_cuda (bool): Whether the resulting tensor should be on cuda

    Returns:
        (torch.Tensor, list[int], list[(int, int)]):
            The flattened tensors, the list start indices of each packed tensor,
            and the original shape of each tensor.

            Those values are used to then unpack the tensor
    """
    indices = [0]
    for tensor in tensors:
        new_end = indices[-1] + tensor.nelement()
        indices.append(new_end)

    tensor_sizes = [t.size() for t in tensors]

    vec = torch.empty(
        indices[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
        dtype=tensors[0].dtype,
    )

    for tensor, start_idx, end_idx in zip(tensors, indices[:-1], indices[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)

    return vec, indices, tensor_sizes


def unpack_tensors(aggregated, indices, sizes):
    """
    Unpacks a 1-dimensional tensor into a list of tensors

    Args:
        aggregated (torch.Tensor): The 1-dimensional tensor
        indices (List[Int]): The start index of each tensor
        sizes (List[(Int, Int)]): The size of each resulting tensor

    Returns:
        List[torch.Tensor]: The unpacked tensors
    """
    start_index = indices[:-1]
    end_index = indices[1:]

    tensors = []
    for i, (start, end) in enumerate(zip(start_index, end_index)):
        tensors.append(aggregated[start:end].view(sizes[i]))

    return tensors


##########################################################################################


class Aggregation(object):
    """Aggregate updates / models from different processes.

    Args:
        use_cuda (bool): Whether to use CUDA tensors for communication
    """

    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda

    def _agg(self, data, op, denom=None):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggregated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def _agg_weights_by_model(self, model, op, denom=None):
        """Aggregate models by model weight, all layers at once

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)
        """
        # Pack all layers
        packed, indices, sizes = pack_tensors(
            [t for t in model.parameters()], use_cuda=self.use_cuda
        )
        aggregated = self._agg(packed, op=op, denom=denom)

        tensors = unpack_tensors(aggregated, indices, sizes)
        # Unpack
        for i, param in enumerate(model.parameters()):
            param.data = tensors[i]

    def _agg_gradients_by_model(self, model, op, denom=None):
        """Aggregate models gradients, all layers at once

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)
        """
        # Pack all layers
        packed, indices, sizes = pack_tensors(
            [t.grad for t in model.parameters()], use_cuda=self.use_cuda
        )
        aggregated = self._agg(packed, op=op, denom=denom)

        # Unpack
        tensors = unpack_tensors(aggregated, indices, sizes)
        for i, param in enumerate(model.parameters()):
            param.grad.data = tensors[i]

    def _agg_weights_by_layer(self, model, op, denom=None):
        """Aggregate models by model weight, for each layer individually

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.data, op=op, denom=denom)
            param.data = grad

    def _agg_gradients_by_layer(self, model, op, denom=None):
        """Aggregate models gradients each layer individually

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op, denom=denom)
            param.grad.data = grad

    def agg_model(self, by_layer=False):
        if by_layer:
            return self._agg_weights_by_layer
        else:
            return self._agg_weights_by_model

    def agg_grad(self, by_layer=False):
        if by_layer:
            return self._agg_gradients_by_layer
        else:
            return self._agg_gradients_by_model


class AllReduceAggregation(Aggregation):
    """Aggregate weights / models from different processes using all-reduce aggregation

    Args:
        world_size (int): Current distributed world size
        divide_before (bool): Perform division before reduction (avoid overflow)
        use_cuda (bool): Use cuda tensors for reduction
    """

    def __init__(self, world_size, divide_before=False, use_cuda=False):
        self.world_size = world_size
        self.divide_before = divide_before
        super(AllReduceAggregation, self).__init__(use_cuda=use_cuda)

    def _reduce(self, data):
        """Reduces the given tensor using `torch.distributed` and op=`dist.ReduceOp.SUM`

        Args:
            data (:obj:`torch.Tensor`): The tensor to reduce

        Returns:
            (:obj:`torch.Tensor`): The reduced Tensor
        """
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data

    def _divide(self, data, op, denom=None):
        """Divides the given `data` tensor by

        - `world_size` if op == `avg_world`
        - `denom`, if op == `custom_avg`

        Args:
            data (:obj:`torch.Tensor`): Data tensor to divide
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)

        Returns:
            (:obj:`torch.Tensor`): The resulting tensor
        """
        if op not in ALLREDUCE_AGGREGATION_OPS:
            raise NotImplementedError("Allreduce not implemented for op={}".format(op))

        if op == "avg_world":
            data.div_(self.world_size)
        elif op == "custom_avg":
            if denom is None or denom == 0:
                raise ValueError("Denominator should be one element tensor")
            data.div_(denom)

        return data

    def _agg(self, data, op, denom=None):
        """Aggregate data using `op` operation.

        - If op == `avg_world`, the reduced tensor will be divided by `world_size`,
        - if op == `custom_avg`, the reduced tensor will be divided by `denom`

        If `self.divide_before`, the division is performed before reduction.
        This can be helpful to avoid overflows when using `float16` training

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggregated.
            op (str): Aggregation method. Should be in `ALLREDUCE_AGGREGATION_OPS`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `custom_avg`. (default: `None`)

        Returns:
            (:obj:`torch.Tensor`): The aggregated tensor.
        """
        if self.divide_before:
            data = self._divide(data, op, denom)

        if self.world_size > 1:
            data = self._reduce(data)

        if not self.divide_before:
            data = self._divide(data, op, denom)
        return data


class AllReduceAggregationHVD(AllReduceAggregation):
    def _reduce(self, data):
        data = hvd.allreduce(data, op=hvd.Sum)
        return data


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


class SparsifiedAggregation(Aggregation):
    """Aggregate sparsified updates."""

    def __init__(self, model, use_cuda=False):
        super(SparsifiedAggregation, self).__init__(use_cuda=use_cuda)
        pass

    def _agg(self, data, op, denom=None):
        pass


class PowerAggregation(Aggregation):
    """Aggregate updates using power iteration and error feedback.

    Args:
            model (:obj:`nn.Module`): Model which contains parameters for SGD
            use_cuda (bool): Whether to use cuda tensors for aggregation
            reuse_query (bool): Whether to use warm start to initialize the power iteration
            rank (int): The rank of the gradient approximation
    """

    def __init__(self, model, use_cuda=False, reuse_query=False, rank=1):
        super(PowerAggregation, self).__init__(use_cuda=use_cuda)
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.RandomState(1)
        self.n_workers = dist.get_world_size()
        self.rank = rank
        self.memories = [torch.zeros_like(param) for param in model.parameters()]
        self.send_buffers = [torch.zeros_like(param) for param in model.parameters()]

    def set_random(self, vector):
        """Sets the data in `vector` to random values."""
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)

    def _agg_gradients_by_model(self, model, op, denom=None):
        """Aggregate models gradients, all layers at once

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (None): Not used here

        """
        grads = [t.grad.data for t in model.parameters()]
        aggregated = self._agg(grads, op=op)

        for i, param in enumerate(model.parameters()):
            param.grad.data = aggregated[i]

    def agg_weights(self, by_layer=False):
        raise NotImplementedError("PowerSGD doesn't allow aggregation by weights")

    def agg_model(self, by_layer=False):
        if by_layer:
            raise NotImplementedError("PowerSGD doesn't allow aggregation by layer")
        else:
            return self._agg_gradients_by_model

    def _agg(self, data, op, denom=None):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (None): Not used here

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if op == "avg":
            for grad, memory, send_bfr in zip(data, self.memories, self.send_buffers):
                send_bfr.data[:] = grad + memory
            self.reduce(self.send_buffers, data, self.memories)
        else:
            raise NotImplementedError("op {} is not supported yet.".format(op))
        return data

    def reduce(self, grad_in, grad_out, memory_out):
        """Reduces the gradients between the workers in place and calculates error feedback.

        Args:
            grad_in (list[torch.Tensor]): The gradients to reduce.
            grad_out (list[torch.Tensor]): Used for storing the reduced gradients.
            memory_out (list[torch.Tensor]): Used for storing error feedback.
        """
        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        p_total_size = 0
        q_total_size = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            p_total_size += n * rank
            q_total_size += m * rank
        if self.p_memory is None:
            self.p_memory = torch.empty(p_total_size, device=self.device)
            self.q_memory = torch.empty(q_total_size, device=self.device)

        # Find them again and make lists of pointers
        ps = []
        qs = []
        p_idx = 0
        q_idx = 0
        for tensor, _, _ in high_rank_tensors:
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
            qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
            p_idx += n * rank
            q_idx += m * rank

        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape

            if self.reuse_query and not memory_is_uninitialized:
                pass
            else:
                # Sample a query vector q
                self.set_random(q)

        for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix, q, out=p)

        dist.all_reduce(self.p_memory)

        # Start communicating rank 1 tensors
        rank1_packed, rank1_indices, rank1_sizes = pack_tensors(
            [tensor for (tensor, _, _) in rank1_tensors]
        )

        rank1_handle = dist.all_reduce(rank1_packed, async_op=True)

        for p in ps:
            orthogonalize(p)

        for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
            matrix = tensor.view(tensor.shape[0], -1)
            torch.matmul(matrix.t(), p, out=q)

        dist.all_reduce(self.q_memory)
        self.q_memory.data[:] /= self.n_workers

        for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
            # Set the output gradient
            torch.matmul(p, q.t(), out=out.data[:])
            mem.data[:] = tensor - out

        rank1_handle.wait()
        rank1_packed /= self.n_workers
        rank1_unpacked = unpack_tensors(rank1_packed, rank1_indices, rank1_sizes)
        for i, (_, out, _) in enumerate(rank1_tensors):
            out[:] = rank1_unpacked[i]


def get_backend_tensor(tensor):
    if dist.is_initialized() and dist.get_backend() == dist.Backend.NCCL:
        return tensor.cuda()
    return tensor
