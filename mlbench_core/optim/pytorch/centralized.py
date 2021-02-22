from abc import ABC

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam
from torch.optim.optimizer import required

from mlbench_core.aggregation.pytorch.centralized import (
    AVG_CUSTOM,
    AVG_WORLD,
    AllReduceAggregation,
    PowerAggregation,
)
from mlbench_core.optim.pytorch.optim import SparsifiedSGD


class GenericCentralizedOptimizer(ABC):
    """Implements a generic centralized (synchronous) optimizer with `AllReduceAggregation`.
    Averages the reduced parameters over the world size, after aggregation.
    Can aggregate gradients or weights, by layers or all at once.

    Args:
        world_size (int): Size of the network
        model (:obj:`nn.Module`): Model which contains parameters for SGD
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
        agg_grad (bool): Aggregate the gradients before updating weights. If `False`,
            weights will be updated and then reduced across all workers. (default: `True`)
    """

    def __init__(
        self,
        world_size,
        model,
        use_cuda=False,
        by_layer=False,
        divide_before=False,
        agg_grad=True,
    ):
        if not world_size:
            raise ValueError('"world_size" not set for optimizer')
        if not model:
            raise ValueError('"model" not set for optimizer')

        self.agg_mode = AVG_WORLD
        self.model = model
        self.agg_grad = agg_grad
        self.world_size = world_size
        agg = AllReduceAggregation(
            world_size=world_size, use_cuda=use_cuda, divide_before=divide_before
        )  # Divide after reduction
        if agg_grad:
            self.agg = agg.agg_grad(by_layer=by_layer)
        else:
            self.agg = agg.agg_model(by_layer=by_layer)
            self.agg(self.model, self.agg_mode)  # Agg params once at init

        self.optimizer = None

    def step(self, closure=None, tracker=None):
        """Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
        """
        if self.agg_grad:

            if self.world_size > 1:
                self.agg(self.model, self.agg_mode)
                if tracker:
                    tracker.record_batch_agg()

            loss = self.optimizer.step(closure=closure)
            if tracker:
                tracker.record_batch_opt_step()
        else:
            loss = self.optimizer.step(closure=closure)
            if tracker:
                tracker.record_batch_opt_step()

            if self.world_size > 1:
                self.agg(self.model, self.agg_mode)
                if tracker:
                    tracker.record_batch_agg()
        return loss

    def __getattr__(self, item):
        """Any call to unknown function/members will be redirected to the optimizer"""
        if self.optimizer is None:
            raise ValueError("Undefined Optimizer")
        return self.optimizer.__getattribute__(item)

    def zero_grad(self):
        self.optimizer.zero_grad()


class CentralizedSparsifiedSGD(SparsifiedSGD):
    """Implements centralized sparsified version of stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): Learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        sparse_grad_size (int): Size of the sparsified gradients vector (default: 10)
        random_sparse (bool): Whether select random sparsification (default: `False`)
        average_world (bool): Whether to average models on the world_size (default: `True`)

    """

    def __init__(
        self,
        params=None,
        lr=required,
        weight_decay=0,
        sparse_grad_size=10,
        random_sparse=False,
        world_size=1,
        average_world=True,
    ):
        if not params:
            raise ValueError('"params" not set for optimizer')
        self.average_world = average_world
        self.world_size = world_size
        self.random_sparse = random_sparse
        super().__init__(params, lr, weight_decay, sparse_grad_size)

    def step(self, closure=None):
        """Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                # Sparsify the gradients
                sparse_tensor = self.sparsify_gradients(p, lr)
                # Aggregate the gradients
                gathered_list = [
                    torch.zeros_like(sparse_tensor) for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_list, sparse_tensor)
                p.grad.data = torch.zeros_like(p.grad.data)

                if self.random_sparse:
                    for grad_tensor in gathered_list:
                        for index in range(grad_tensor.size()[1]):
                            p.grad.data[0, int(grad_tensor[0, index])] += grad_tensor[
                                1, index
                            ]
                else:
                    for grad_tensor in gathered_list:
                        tensor_size = grad_tensor.size()[1]
                        begin = int(grad_tensor[0, 0])
                        p.grad.data[
                            0, begin : (begin + tensor_size - 1)
                        ] += grad_tensor[0, 1:]

                if self.average_world:
                    p.grad.data /= self.world_size

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-d_p)

        return loss


class CentralizedSGD(GenericCentralizedOptimizer):
    """Implements centralized stochastic gradient descent (optionally with momentum).
    Averages the reduced parameters over the world size.

    Args:
        world_size (int): Size of the network
        model (:obj:`nn.Module`): Model which contains parameters for SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
        agg_grad (bool): Aggregate the gradients before updating weights. If `False`,
            weights will be updated and then reduced across all workers. (default: `True`)
    """

    def __init__(
        self,
        world_size=None,
        model=None,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        use_cuda=False,
        by_layer=False,
        agg_grad=True,
    ):
        super().__init__(
            model=model,
            world_size=world_size,
            use_cuda=use_cuda,
            by_layer=by_layer,
            agg_grad=agg_grad,
        )

        self.optimizer = SGD(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            params=model.parameters(),
        )


class CentralizedAdam(GenericCentralizedOptimizer):
    """Implements centralized Adam algorithm.
    Averages the reduced parameters over the world size

    Args:
        world_size (int): Size of the network
        model (:obj:`nn.Module`): Model which contains parameters for Adam
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper :cite:`adam_convergence`
            (default: False)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
    """

    def __init__(
        self,
        world_size=None,
        model=None,
        lr=required,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        use_cuda=False,
        by_layer=False,
        agg_grad=True,
    ):
        super().__init__(
            model=model,
            world_size=world_size,
            use_cuda=use_cuda,
            by_layer=by_layer,
            agg_grad=agg_grad,
        )

        self.optimizer = Adam(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            params=model.parameters(),
        )


class PowerSGD(SGD):
    """Implements PowerSGD with error feedback (optionally with momentum).

    Args:
        model (:obj:`nn.Module`): Model which contains parameters for SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_world (bool): Whether to average models on the world_size (default: `True`)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
        reuse_query (bool): Whether to use warm start to initialize the power iteration
        rank (int): The rank of the gradient approximation
    """

    def __init__(
        self,
        model=None,
        lr=required,
        momentum=0,
        weight_decay=0,
        dampening=0,
        nesterov=False,
        average_world=True,
        use_cuda=False,
        by_layer=False,
        reuse_query=False,
        world_size=1,
        rank=1,
    ):
        if not model:
            raise ValueError('"model" not set for optimizer')
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        super().__init__(
            model.parameters(), lr, momentum, dampening, weight_decay, nesterov
        )
        if average_world:
            self.agg_mode = "avg"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.world_size = world_size
        self.model = model
        self.agg = PowerAggregation(
            model=model,
            use_cuda=use_cuda,
            reuse_query=reuse_query,
            world_size=world_size,
            rank=rank,
        ).agg_grad(by_layer=by_layer)

    def step(self, closure=None, tracker=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
        """
        if self.world_size > 1:
            self.agg(self.model, self.agg_mode)
            if tracker:
                tracker.record_batch_agg()
        loss = super().step(closure=closure)
        if tracker:
            tracker.record_batch_opt_step()
        return loss


class CustomCentralizedOptimizer(GenericCentralizedOptimizer):
    """Custom Centralized Optimizer. Can be used with any optimizer passed as argument.
    Adds the gradient clipping option, as well as custom average

    Args:
        model (:obj:`torch.nn.Module`): model
        world_size (int): Distributed world size
        use_cuda (bool): Use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer
        agg_grad (bool): Aggregate the gradients before updating weights. If `False`,
            weights will be updated and then reduced across all workers. (default: `True`)
        grad_clip (float): coefficient for gradient clipping, max L2 norm of the gradients
        average_world (bool): Average the gradients by world size
        average_custom (bool): Divide gradients by given denominator at each step, instead
            of `world_size`
        divide_before (bool): Divide gradients before reduction (default: False)
    """

    def __init__(
        self,
        model,
        world_size,
        optimizer,
        use_cuda=False,
        by_layer=False,
        agg_grad=True,
        grad_clip=float("inf"),
        average_world=False,
        average_custom=False,
        divide_before=False,
    ):
        super().__init__(
            model=model,
            world_size=world_size,
            use_cuda=use_cuda,
            by_layer=by_layer,
            agg_grad=agg_grad,
            divide_before=divide_before,
        )
        self.grad_clip = grad_clip

        if average_world:
            self.agg_mode = AVG_WORLD
        elif average_custom:
            self.agg_mode = AVG_CUSTOM
        else:
            raise NotImplementedError("Please choose aggregation method")

        self.optimizer = optimizer

    def step(self, closure=None, tracker=None, denom=None):
        """Performs one step of the optimizer.

        Args:
            closure (callable): Optimizer closure argument
            tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
            denom (Optional[:obj:`torch.Tensor`]): Custom denominator to reduce by
        """

        if self.agg_grad:
            if self.world_size > 1:
                self.agg(self.model, self.agg_mode, denom=denom)
                if tracker:
                    tracker.record_batch_agg()

            # Clip norm
            if self.grad_clip != float("inf"):
                clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step(closure=closure)
            if tracker:
                tracker.record_batch_opt_step()
        else:
            self.optimizer.step(closure=closure)
            if tracker:
                tracker.record_batch_opt_step()
            if self.world_size > 1:
                self.agg(self.model, self.agg_mode)
                if tracker:
                    tracker.record_batch_agg()

        return True
