import numpy as np
import torch
import torch.distributed as dist
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer, required

from mlbench_core.utils.pytorch.distributed import (
    AllReduceAggregation,
    DecentralizedAggregation,
    PowerAggregation,
)


class SparsifiedSGD(Optimizer):
    r"""Implements sparsified version of stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        sparse_grad_size (int): Size of the sparsified gradients vector (default: 10).

    """

    def __init__(self, params, lr=required, weight_decay=0, sparse_grad_size=10):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(SparsifiedSGD, self).__init__(params, defaults)

        self.__create_gradients_memory()
        self.__create_weighted_average_params()

        self.num_coordinates = sparse_grad_size

    def __create_weighted_average_params(self):
        r""" Create a memory to keep the weighted average of parameters in each iteration """
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["estimated_w"] = torch.zeros_like(p.data)
                p.data.normal_(0, 0.01)
                param_state["estimated_w"].copy_(p.data)

    def __create_gradients_memory(self):
        r""" Create a memory to keep gradients that are not used in each iteration """
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["memory"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-d_p)

        return loss

    def sparsify_gradients(self, param, lr):
        """ Calls one of the sparsification functions (random or blockwise)

        Args:
            random_sparse (bool): Indicates the way we want to make the gradients sparse
                (random or blockwise) (default: False)
            param (:obj: `torch.nn.Parameter`): Model parameter
        """
        if self.random_sparse:
            return self._random_sparsify(param, lr)
        else:
            return self._block_sparsify(param, lr)

    def _random_sparsify(self, param, lr):
        """ Sparsify the gradients vector by selecting 'k' of them randomly.

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate

        """

        self.state[param]["memory"] += param.grad.data * lr

        indices = np.random.choice(
            param.data.size()[1], self.num_coordinates, replace=False
        )
        sparse_tensor = torch.zeros(2, self.num_coordinates)

        for i, random_index in enumerate(indices):
            sparse_tensor[1, i] = self.state[param]["memory"][0, random_index]
            self.state[param]["memory"][0, random_index] = 0
        sparse_tensor[0, :] = torch.tensor(indices)

        return sparse_tensor

    def _block_sparsify(self, param, lr):
        """ Sparsify the gradients vector by selecting a block of them.

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate
        """

        self.state[param]["memory"] += param.grad.data * lr

        num_block = int(param.data.size()[1] / self.num_coordinates)

        current_block = np.random.randint(0, num_block)
        begin_index = current_block * self.num_coordinates

        end_index = begin_index + self.num_coordinates - 1
        output_size = 1 + end_index - begin_index + 1

        sparse_tensor = torch.zeros(1, output_size)
        sparse_tensor[0, 0] = begin_index
        sparse_tensor[0, 1:] = self.state[param]["memory"][
            0, begin_index : end_index + 1
        ]
        self.state[param]["memory"][0, begin_index : end_index + 1] = 0

        return sparse_tensor

    def update_estimated_weights(self, iteration, sparse_vector_size):
        """ Updates the estimated parameters

        Args:
            iteration (int): Current global iteration
            sparse_vector_size (int): Size of the sparse gradients vector
        """
        t = iteration
        for group in self.param_groups:
            for param in group["params"]:
                tau = param.data.size()[1] / sparse_vector_size
                rho = (
                    6
                    * ((t + tau) ** 2)
                    / ((1 + t) * (6 * (tau ** 2) + t + 6 * tau * t + 2 * (t ** 2)))
                )
                self.state[param]["estimated_w"] = (
                    self.state[param]["estimated_w"] * (1 - rho) + param.data * rho
                )

    def get_estimated_weights(self):
        """ Returns the weighted average parameter tensor """
        estimated_params = []
        for group in self.param_groups:
            for param in group["params"]:
                estimated_params.append(self.state[param]["estimated_w"])
        return estimated_params


class CentralizedSparsifiedSGD(SparsifiedSGD):
    r"""Implements centralized sparsified version of stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): Learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        sparse_grad_size (int): Size of the sparsified gradients vector (default: 10)
        random_sparse (bool): Whether select random sparsification (default: `False`)
        average_models (bool): Whether to average models together (default: `True`)

    """

    def __init__(
        self,
        params=None,
        lr=required,
        weight_decay=0,
        sparse_grad_size=10,
        random_sparse=False,
        average_models=True,
    ):
        if not params:
            raise ValueError('"params" not set for optimizer')
        self.average_models = average_models
        self.world_size = dist.get_world_size()
        self.random_sparse = random_sparse
        super(CentralizedSparsifiedSGD, self).__init__(
            params, lr, weight_decay, sparse_grad_size
        )

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.

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

                if self.average_models:
                    p.grad.data /= self.world_size

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-d_p)

        return loss


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
        average_models (bool): Whether to average models together. (default: `True`)
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
        average_models=True,
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

        if average_models:
            self.agg_mode = "avg_world"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.model = model
        self.agg = DecentralizedAggregation(
            rank, neighbors, use_cuda=use_cuda
        ).agg_model(by_layer=by_layer)

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = super(DecentralizedSGD, self).step(closure=closure)
        # Averaging the model after updating the gradient separately.
        self.agg(self.model, self.agg_mode)
        return loss


class CentralizedSGD(SGD):
    r"""Implements centralized stochastic gradient descent (optionally with momentum).

    Args:
        world_size (int): Size of the network
        model (:obj:`nn.Module`): Model which contains parameters for SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_models (bool): Whether to average models together. (default: `True`)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
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
        average_models=True,
        use_cuda=False,
        by_layer=False,
    ):
        if not world_size:
            raise ValueError('"world_size" not set for optimizer')
        if not model:
            raise ValueError('"model" not set for optimizer')
        super(CentralizedSGD, self).__init__(
            model.parameters(), lr, momentum, dampening, weight_decay, nesterov
        )
        if average_models:
            self.agg_mode = "avg_world"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.model = model
        self.agg = AllReduceAggregation(
            world_size=world_size, use_cuda=use_cuda
        ).agg_grad(by_layer=by_layer)

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.agg(self.model, self.agg_mode)
        loss = super(CentralizedSGD, self).step(closure=closure)
        return loss


class SignSGD(SGD):
    r"""Implements sign stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_models (bool): Whether to average models together. (default: `True`)

    """

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # Update with the sign
                p.data.add_(-group["lr"], torch.sign(d_p))

        return loss


class CentralizedAdam(Adam):
    r"""Implements centralized Adam algorithm.

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
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        average_models (bool): Whether to average models together. (default: `True`)
        use_cuda (bool): Whether to use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer instead of all layers at once
    """

    def __init__(
        self,
        world_size=None,
        model=None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        average_models=True,
        use_cuda=False,
        by_layer=False,
    ):
        if not world_size:
            raise ValueError('"world_size" not set for optimizer')
        if not model:
            raise ValueError('"model" not set for optimizer')
        super(CentralizedAdam, self).__init__(
            model.parameters(), lr, betas, eps, weight_decay, amsgrad
        )
        if average_models:
            self.agg_mode = "avg_world"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.model = model
        self.agg = AllReduceAggregation(
            world_size=world_size, use_cuda=use_cuda
        ).agg_grad(by_layer=by_layer)

    def step(self, closure=None):
        """ Aggregates the gradients and performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.agg(self.model, self.agg_mode)
        loss = super(CentralizedAdam, self).step(closure=closure)
        return loss


class PowerSGD(SGD):
    r"""Implements PowerSGD with error feedback (optionally with momentum).

        Args:
            model (:obj:`nn.Module`): Model which contains parameters for SGD
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)
            average_models (bool): Whether to average models together. (default: `True`)
            use_cuda (bool): Whether to use cuda tensors for aggregation
            by_layer (bool): Aggregate by layer instead of all layers at once
            reuse_query (bool): Whether to use warm start to initialize the power iteration
            rank (int): The rank of the gradient approximation
        """

    def __init__(
        self,
        model=None,
        lr=0.1,
        momentum=0,
        weight_decay=0,
        dampening=0,
        nesterov=False,
        average_models=True,
        use_cuda=False,
        by_layer=False,
        reuse_query=False,
        rank=1,
    ):
        if not model:
            raise ValueError('"model" not set for optimizer')
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        super(PowerSGD, self).__init__(
            model.parameters(), lr, momentum, dampening, weight_decay, nesterov
        )
        if average_models:
            self.agg_mode = "avg"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        self.model = model
        self.agg = PowerAggregation(
            model=model, use_cuda=use_cuda, reuse_query=reuse_query, rank=rank
        ).agg_grad(by_layer=by_layer)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.agg(self.model, self.agg_mode)
        loss = super(PowerSGD, self).step(closure=closure)
        return loss


optimizers = {
    "centralized_sparsified_sgd": CentralizedSparsifiedSGD,
    "decentralized_sgd": DecentralizedSGD,
    "centralized_sgd": CentralizedSGD,
    "sign_sgd": SignSGD,
    "centralized_adam": CentralizedAdam,
    "power_sgd": PowerSGD,
}


def get_optimizer(optimizer, **kwargs):
    r"""Returns an object of the class specified with the argument `optimizer`.

        Args:
            optimizer (str): name of the optimizer
            **kwargs (dict, optional): additional optimizer-specific parameters. For the list of supported parameters
                for each optimizer, please look at its documentation.
        """
    return optimizers[optimizer](**kwargs)
