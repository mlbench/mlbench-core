import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError as e:
    pass


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
    """Aggregate udpates / models from different processes.

    Args:
        use_cuda (bool): Whether to use CUDA tensors for communication
    """

    def __init__(self, use_cuda=False, has_grad=True):
        self.use_cuda = use_cuda
        self.has_gad = has_grad

    def _agg(self, data, op, denom=None):
        """Aggregate data using `op` operation.

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggregated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)

        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def _agg_weights_by_model(self, model, op, denom=None):
        """Aggregate models by model weight, all layers at once

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)
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
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)
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
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.data, op=op, denom=denom)
            param.data = grad

    def _agg_gradients_by_layer(self, model, op, denom=None):
        """Aggregate models gradients each layer individually

        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)
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
    """Aggregate udpates / models from different processes using all-reduce aggregation

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
            data (`obj`:torch.Tensor): The tensor to reduce

        Returns:
            (`obj`:torch.Tensor): The reduced Tensor
        """
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data

    def _divide(self, data, op, denom=None):
        """Divides the given `data` tensor by

        - `world_size` if op == `avg`
        - `denom`, if op == `avg_batch`

        Args:
            data (`obj`:torch.Tensor): Data tensor to divide
            op (str): One of `avg` or `avg_batch`
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)

        Returns:
            (`obj`:torch.Tensor): The resulting tensor
        """
        if op == "avg":
            data.div_(self.world_size)
        elif op == "avg_batch":
            if denom is None or denom == 0:
                raise ValueError("Denominator should be one element tensor")
            data.div_(denom)
        else:
            raise NotImplementedError("Allreduce not implemented for op={}".format(op))

        return data

    def _agg(self, data, op, denom=None):
        """Aggregate data using `op` operation.

        - If op == `avg`, the reduced tensor will be divided by `world_size`,
        - if op == `avg_batch`, the reduced tensor will be divided by `denom`

        If `self.divide_before`, the division is performed before reduction.
        This can be helpful to avoid overflows when using `float16` training

        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggregated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
            denom (:obj:`torch.Tensor`, optional): Custom denominator to average by
                Use with op == `avg_batch`. (default: `None`)

        Returns:
            (`obj`:torch.Tensor): The aggregated tensor.
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
        if op == "avg":
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


def get_backend_tensor(tensor):
    if dist.is_initialized() and dist.get_backend() == dist.Backend.NCCL:
        return tensor.cuda()
    return tensor
