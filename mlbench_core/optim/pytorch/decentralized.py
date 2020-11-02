from torch.optim import SGD
from torch.optim.optimizer import required

from mlbench_core.aggregation.pytorch.decentralized import DecentralizedAggregation


class DecentralizedSGD(SGD):
    r"""Implements decentralized stochastic gradient descent (optionally with momentum).

    Args:
        rank (int): rank of current process in the network
        neighbors (list): list of ranks of the neighbors of current process
        model (:obj:`nn.Module`): model which contains parameters for SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_world (bool): Whether to average models on the world_size (default: `True`)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
    """

    def __init__(
        self,
        rank=None,
        neighbors=None,
        model=None,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        average_world=True,
        use_cuda=False,
        by_layer=False,
    ):
        if not rank:
            raise ValueError('"rank" not set for optimizer')
        if not neighbors:
            raise ValueError('"neighbors" not set for optimizer')
        if not model:
            raise ValueError('"model" not set for optimizer')
        super(DecentralizedSGD, self).__init__(
            model.parameters(), lr, momentum, dampening, weight_decay, nesterov
        )

        if average_world:
            self.agg_mode = "avg_world"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.model = model
        self.agg = DecentralizedAggregation(
            rank, neighbors, use_cuda=use_cuda
        ).agg_model(by_layer=by_layer)

    def step(self, closure=None, tracker=None):
        """Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
        """
        loss = super(DecentralizedSGD, self).step(closure=closure)
        if tracker:
            tracker.record_batch_opt_step()
        # Averaging the model after updating the gradient separately.
        self.agg(self.model, self.agg_mode)
        if tracker:
            tracker.record_batch_agg()
        return loss
