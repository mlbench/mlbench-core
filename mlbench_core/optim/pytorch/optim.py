import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim import SGD
import torch.distributed as dist


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
            for p in group['params']:
                param_state = self.state[p]
                param_state['estimated_w'] = torch.zeros_like(p.data)
                p.data.normal_(0, 0.01)
                param_state['estimated_w'].copy_(p.data)

    def __create_gradients_memory(self):
        r""" Create a memory to keep gradients that are not used in each iteration """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)

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

            weight_decay = group['weight_decay']

            for p in group['params']:

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
            random_sparse (bool): Indicates the way we want to make the gradients sparse (random or blockwise) (default: False)
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
        
        self.state[param]['memory'] += param.grad.data * lr

        indices = np.random.choice(param.data.size()[1], self.num_coordinates, replace=False)
        sparse_tensor = torch.zeros(2, self.num_coordinates)

        for i, random_index in enumerate(indices):
            sparse_tensor[1, i] = self.state[param]['memory'][0, random_index]
            self.state[param]['memory'][0, random_index] = 0
        sparse_tensor[0, :] = torch.tensor(indices)

        return sparse_tensor

    def _block_sparsify(self, param, lr):
        """ Sparsify the gradients vector by selecting a block of them.

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate 
        """
    
        self.state[param]['memory'] += param.grad.data * lr

        num_block = int(param.data.size()[1] / self.num_coordinates)

        current_block = np.random.randint(0, num_block)
        begin_index = current_block * self.num_coordinates

        end_index = begin_index + self.num_coordinates - 1
        output_size = 1 + end_index - begin_index + 1

        sparse_tensor = torch.zeros(1, output_size)
        sparse_tensor[0, 0] = begin_index
        sparse_tensor[0, 1:] = self.state[param]['memory'][0, begin_index: end_index + 1]
        self.state[param]['memory'][0, begin_index: end_index + 1] = 0

        return sparse_tensor

    def update_estimated_weights(self, iteration, sparse_vector_size):
        """ Updates the estimated parameters 
        
        Args:
            iteration (int): Current global iteration
            sparse_vector_size (int): Size of the sparse gradients vector 
        """
        t = iteration
        for group in self.param_groups:
            for param in group['params']:
                tau = param.data.size()[1] / sparse_vector_size
                rho = 6 * ((t + tau) ** 2) / ((1 + t) * (6 * (tau ** 2) + t + 6 * tau * t + 2 * (t ** 2)))
                self.state[param]['estimated_w'] = self.state[param]['estimated_w'] * (1 - rho) + param.data * rho

    def get_estimated_weights(self):
        """ Returns the weighted average parameter tensor """
        estimated_params = []
        for group in self.param_groups:
            for param in group['params']:
                estimated_params.append(self.state[param]['estimated_w'])
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
    def __init__(self, params, lr=required, weight_decay=0, sparse_grad_size=10, random_sparse=False, average_models=True):
        self.average_models = average_models
        self.world_size = dist.get_world_size()
        self.random_sparse = random_sparse
        super(CentralizedSparsifiedSGD, self).__init__(params, lr, weight_decay, sparse_grad_size)

        def step(self, closure=None):
            """ Aggregates the gradients and performs a single optimization step.

                Arguments:
                    closure (callable, optional): A closure that reevaluates the model and returns the loss. 
            """

            loss = None

            if closure is not None:
                loss = closure()

            for group in self.param_groups:

                weight_decay = group['weight_decay']
                lr = group['lr']

                for p in group['params']:
                    # Sparsify the gradients
                    sparse_tensor = self.sparsify_gradients(p, lr)
                    # Aggregate the gradients
                    gathered_list = [torch.zeros_like(sparse_tensor) for _ in range(self.world_size)]
                    dist.all_gather(gathered_list, sparse_tensor)
                    p.grad.data = torch.zeros_like(p.grad.data)

                    if self.sparse_tensor:
                        for grad_tensor in gathered_list:
                            for index in range(grad_tensor.size()[1]):
                                p.grad.data[0, int(grad_tensor[0, index])] += grad_tensor[1, index]
                    else:            
                        for grad_tensor in gathered_list:
                            tensor_size = grad_tensor.size()[1]
                            begin = int(grad_tensor[0, 0])
                            p.grad.data[0, begin:(begin + tensor_size - 1)] += grad_tensor[0, 1:]

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
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        average_models (bool): Whether to average models together. (default: `True`)

    """
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, average_models=True):
        self.average_models = average_models
        self.world_size = dist.get_world_size()
        super(DecentralizedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)


                # all_reduce the parameters.
                dist.all_reduce(p.data, op=dist.reduce_op.SUM)

                if self.average_models:
                    p.data /= self.world_size

        return loss


class CentralizedSGD(SGD):
    r"""Implements centralized stochastic gradient descent (optionally with momentum).

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
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, average_models= True):
        self.average_models = average_models
        self.world_size = dist.get_world_size()
        super(CentralizedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                # all_reduce the gradients.
                dist.all_reduce(p.grad.data, op=dist.reduce_op.SUM)

                if self.average_models:
                    p.grad.data /= self.world_size

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss




