import numpy as np
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch.utils import orthogonalize, pack_tensors, unpack_tensors

try:
    import horovod.torch as hvd
except ImportError as e:
    pass

from mlbench_core.aggregation.pytorch.aggregation import Aggregation

AVG_WORLD = "avg_world"
AVG_CUSTOM = "avg_custom"
ALLREDUCE_AGGREGATION_OPS = [AVG_WORLD, AVG_CUSTOM]
"""
All possible aggregations for AllReduceAggregation
"""


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

        if op == AVG_WORLD:
            data.div_(self.world_size)
        elif op == AVG_CUSTOM:
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
            if dist.get_backend() == dist.Backend.NCCL:
                torch.cuda.synchronize()

        if not self.divide_before:
            data = self._divide(data, op, denom)
        return data


class AllReduceAggregationHVD(AllReduceAggregation):
    """Implements `AllReduceAggregation` using horovod for communication"""

    def _reduce(self, data):
        data = hvd.allreduce(data, op=hvd.Sum)
        return data


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
