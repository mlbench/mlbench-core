# -*- coding: utf-8 -*-
"""Scheduling Learning Rates.

.. rubric:: References

.. [ginsburg2018large] Ginsburg, Boris and Gitman, Igor and You, Yang
    Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling

.. [leslie2017cyclical] Leslie N. Smith
    Cyclical Learning Rates for Training Neural Networks

.. [goyal2017accurate] Goyal, Priya, et al.
    Accurate, large minibatch SGD: training imagenet in 1 hour.

.. [smith2017super] Smith, Leslie N., and Nicholay Topin.
    Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates.


"""
import argparse
import numpy as np
import re
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from bisect import bisect_left, bisect_right


def const(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)


def triangular_learning_rates(optimizer, base_lr, max_lr, cycle_length, scale_fn, extra, mode):
    """Linearly scale the learning rates.

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

    :param optimizer: an optimizer whose learning rate is scheduled.
    :type optimizer: torch.nn.optim.optimizer
    :param base_lr: lower bound and initial lr in a cycle.
    :type base_lr: float
    :param max_lr: upper bound in a cycle
    :type max_lr: float
    :param cycle_length: length of a cycle in terms of batches.
    :type cycle_length: int
    :param scale_fn: custom scaling policy defined by a single argument lambda function, defaults to None
    :type scale_fn: callable, optional
    :returns: a learning rate scheduler
    :rtype: LambdaLR
    """
    step_size = cycle_length / 2

    if mode == 'one_cycle':
        def f(iterations):
            if iterations <= cycle_length:
                cycle = np.floor(1 + iterations / (2 * step_size))
                x = np.abs(iterations/step_size - 2 * cycle + 1)
                lr = base_lr + (max_lr-base_lr) * np.maximum(0, (1-x)) * scale_fn(cycle, iterations)
            else:
                lr = base_lr * extra
            return lr / base_lr
    else:
        def f(iterations):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations/step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr-base_lr) * np.maximum(0, (1-x)) * scale_fn(cycle, iterations)
            return lr / base_lr

    # Use base_lr to overwrite the --lr
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def cyclical_learning_rates(config, optimizer):
    """
    Since leslie2017cyclical_ mentioned that traingular, Welch, Hann windows produce equivalent results,
    we only implement triangular learning rate policy, also known as **linear cycle**.

    The original implementation of leslie2017cyclical_ can be found from `here <https://github.com/bckenstler/CLR>`_.

    smith2017super_ uses one cycle with extra epochs.
    """
    if config.lr_scheduler_level != 'batch':
        raise ValueError("The scheduler should be updated at batch level. Got {}."
                         .format(config.lr_scheduler_level))

    mode = config.clr_mode
    gamma = config.clr_gamma
    if mode in ['linear', 'triangular', 'one_cycle']:
        def scale_fn(cycle, iterations): return 1.
    elif mode == 'triangular2':
        def scale_fn(cycle, iterations): return 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        def scale_fn(cycle, iterations): return gamma ** iterations
    else:
        raise ValueError("Cycle mode {} not support.".format(mode))

    _cycle_unit, _cycle_length = config.lr_scheduler_level, config.clr_cycle_length[config.lr_scheduler_level]
    cycle_length = int(_cycle_length) if _cycle_unit == 'batch' \
        else float(_cycle_length) * config.train_num_batches

    return triangular_learning_rates(optimizer, config.clr_base_lr, config.clr_max_lr,
                                     cycle_length=cycle_length, scale_fn=scale_fn,
                                     extra=config.clr_extra,
                                     mode=mode)


def multistep_learning_rates_with_warmup(config, optimizer):
    """Use multistep learning rate schedule with warmup.

    In goyal2017accurate_, warmup is used in order to apply the ``Linear Scaling Rule``.
    Starting from the ``base_lr``, lr gradually increases to ``base_lr * scaling_factor``.
    Then use multiply the learning rate by ``gamma`` at specified milestones.

    :param config: all configs
    :type config: argparse.Namespace
    :param optimizer: optimizer associated with the scheduler
    :type optimizer: torch.nn.optim.optimizer
    :returns: a learning rate scheduler
    :rtype: LambdaLR
    :raises: ValueError, ValueError, ValueError
    """
    scaling_factor = config.world_size if config.warmup_linear_scaling else 1
    if config.warmup_init_lr_nonscale and (not config.lr):
        lr = config.lr_per_sample * config.batch_size
    else:
        lr = config.lr

    base_lr = lr * scaling_factor

    warmup_durations = config.warmup_durations.get(config.lr_scheduler_level, 0)
    milestones = config.multisteplr_milestones[config.lr_scheduler_level]

    gamma = config.multisteplr_gamma
    warmup = config.warmup

    if config.warmup_init_lr_nonscale:
        warmup_init_lr = lr
    else:
        warmup_init_lr = config.warmup_init_lr

    if not list(milestones) == sorted(milestones):
        raise ValueError('Milestones should be a list of increasing integers.'
                         'Got {}'.format(milestones))

    if warmup_durations >= milestones[0]:
        raise ValueError("The scaling phase should be earlier than the first milestone."
                         "Got {} and {}".format(warmup_durations, milestones[0]))

    def f(durations):
        if warmup and durations <= warmup_durations:
            warmup_progress = durations / warmup_durations
            lr = warmup_progress * base_lr + (1 - warmup_progress) * warmup_init_lr
        else:
            lr = base_lr * gamma ** bisect_right(milestones, durations)
        return lr / base_lr

    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)
