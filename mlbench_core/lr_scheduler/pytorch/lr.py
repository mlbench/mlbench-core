# -*- coding: utf-8 -*-

import argparse
import numpy as np
import re
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from bisect import bisect_left, bisect_right


def const(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)


def triangular_learning_rates(optimizer, base_lr, max_lr, cycle_length, scale_fn, extra, mode):
    """ Linearily Scale Learning Rate

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        base_lr (float): Lower bound and initial learning rate in a cycle.
        max_lr (float): Upper bound in a cycle
        cycle_length (int): Length of a cycle in terms of batches.
        scale_fn(:func:`Function`): The scaling function.
        extra (int): The number of extra epochs to perform after a cycle
        mode (str): The scaling mode to use. One of `linear`, `triangular`, `one_cycle`,
            `triangular2` or `exp_range`

    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """
    step_size = cycle_length / 2

    if mode == 'one_cycle':
        def f(iterations):
            if iterations <= cycle_length:
                cycle = np.floor(1 + iterations / (2 * step_size))
                x = np.abs(iterations / step_size - 2 * cycle + 1)
                lr = base_lr + (max_lr - base_lr) * np.maximum(0,
                                                               (1 - x)) * scale_fn(cycle, iterations)
            else:
                lr = base_lr * extra
            return lr / base_lr
    else:
        def f(iterations):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0,
                                                           (1 - x)) * scale_fn(cycle, iterations)
            return lr / base_lr

    # Use base_lr to overwrite the --lr
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def cyclical_learning_rates(optimizer, mode, gamma, cycle_length, base_lr, max_lr, extra_epochs):
    """ Cyclically Scale Learning Rate

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

    Since :cite:`smith2017cyclical` mentioned that triangular, Welch, Hann windows produce equivalent results,
    we only implement triangular learning rate policy, also known as **linear cycle**.

    The original implementation of :cite:`smith2017cyclical` can be found from `here <https://github.com/bckenstler/CLR>`_.

    :cite:`smith2017super` uses one cycle with extra epochs.

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        mode (str): The scaling mode to use. One of `linear`, `triangular`, `one_cycle`,
            `triangular2` or `exp_range`
        base_lr (float): Lower bound and initial learning rate in a cycle.
        max_lr (float): Upper bound in a cycle
        max_lr (float): The maximum learning rate
        extra_epochs (int): The number of extra epochs to perform after a cycle

    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """
    if mode in ['linear', 'triangular', 'one_cycle']:
        def scale_fn(cycle, iterations): return 1.
    elif mode == 'triangular2':
        def scale_fn(cycle, iterations): return 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        def scale_fn(cycle, iterations): return gamma ** iterations
    else:
        raise ValueError("Cycle mode {} not support.".format(mode))

    return triangular_learning_rates(optimizer, base_lr, max_lr,
                                     cycle_length=cycle_length, scale_fn=scale_fn,
                                     extra=extra_epochs,
                                     mode=mode)


def multistep_learning_rates_with_warmup(optimizer, world_size, lr, gamma, milestones, warmup_duration=None, warmup_lr=None, warmup_linear_scaling=False):
    """ Multistep Learning Rate Schedule with warmup

    In :cite:`goyal2017accurate`, warmup is used in order to apply the ``Linear Scaling Rule``.
    Starting from the ``base_lr``, lr gradually increases to ``base_lr * scaling_factor``.
    Then use multiply the learning rate by ``gamma`` at specified milestones.
    See :cite:`ginsburg2018large`

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        world_size (int): The total number of workers
        lr (float): The initial learning rate
        gamma (float): Decay factor for learning rate
        milestones (:obj:`list` of :obj:`int`): The epochs/steps at which to reduce the
            learning rate
        warmup_duration (int): The number of epochs to perform warmup before regular
            lr scaling starts. Default: `None`
        warmup_lr (float): The learning rate to use for the warmup epochs. Default: `None`
        warmup_linear_scaling (bool): Whether or not to linearily scale lr during
            warmup. Default: `False`
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """
    if bool(warmup_duration) != bool(warmup_lr):
        raise ValueError("Either both or none of warmup_duration and warmup_lr have to be set")

    scaling_factor = 1

    if warmup_linear_scaling:
        scaling_factor = world_size

    base_lr = lr * scaling_factor

    warmup_init_lr = lr
    if warmup_lr:
        warmup_init_lr = warmup_lr

    if not list(milestones) == sorted(milestones):
        raise ValueError('Milestones should be a list of increasing integers.'
                         'Got {}'.format(milestones))

    if warmup_duration >= milestones[0]:
        raise ValueError("The scaling phase should be earlier than the first milestone."
                         "Got {} and {}".format(warmup_duration, milestones[0]))

    def f(duration):
        if warmup_lr and duration <= warmup_duration:
            warmup_progress = duration / warmup_duration
            lr = warmup_progress * base_lr + (1 - warmup_progress) * warmup_init_lr
        else:
            lr = base_lr * gamma ** bisect_right(milestones, duration)
        return lr / base_lr

    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)
