from mlbench_core.utils.pytorch.utils import pack_tensors, unpack_tensors


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
