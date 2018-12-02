import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class SparsifiedSGD(Optimizer):
    r"""Implements sparsified version of stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        sparse_grad_size (int): Size of the sparsified gradients vector.

    """

    def __init__(self, params, lr=required, weight_decay=0, sparse_grad_size=10):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(sparsified_SGD, self).__init__(params, defaults)

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

    def sparsify_gradients(self, lr, random_sparse):
        """ Calls one of the sparsification functions (random or blockwise)

        Args:
            lr (float): Learning rate
            random_sparse (boolean): Indicates the way we want to make the gradients sparse (random or blockwise)
        """
        if random_sparse:
            return self._random_sparsify(lr)
        else:
            return self._block_sparsify(lr)

    def _random_sparsify(self, lr):
        """
        Sparsify the gradients vector by selecting 'k' of them randomly.

        Args:
            lr (float): Learning rate
        """
        params_sparse_tensors = []

        for group in self.param_groups:

            for param in group['params']:
                self.state[param]['memory'] += param.grad.data * lr[0]

                indices = np.random.choice(param.data.size()[1], self.num_coordinates, replace=False)
                sparse_tensor = torch.zeros(2, self.num_coordinates)

                for i, random_index in enumerate(indices):
                    sparse_tensor[1, i] = self.state[param]['memory'][0, random_index]
                    self.state[param]['memory'][0, random_index] = 0
                sparse_tensor[0, :] = torch.tensor(indices)

                params_sparse_tensors.append(sparse_tensor)

        return params_sparse_tensors

    def _block_sparsify(self, lr):
        """
        Sparsify the gradients vector by selecting a block of them.

        Args:
            lr (float): Learning rate
        """
        params_sparse_tensors = []

        
        for group in self.param_groups:

            for param in group['params']:
                self.state[param]['memory'] += param.grad.data * lr[0]

                num_block = int(param.data.size()[1] / self.num_coordinates)

                current_block = np.random.randint(0, num_block)
                begin_index = current_block * self.num_coordinates

                end_index = begin_index + self.num_coordinates - 1
                output_size = 1 + end_index - begin_index + 1

                sparse_tensor = torch.zeros(1, output_size)
                sparse_tensor[0, 0] = begin_index
                sparse_tensor[0, 1:] = self.state[param]['memory'][0, begin_index: end_index + 1]
                self.state[param]['memory'][0, begin_index: end_index + 1] = 0

                params_sparse_tensors.append(sparse_tensor)

        return params_sparse_tensors

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
