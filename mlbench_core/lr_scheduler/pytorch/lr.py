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
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
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
    """ Cyclically Scale Learning Rate

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

    Since :cite:`smith2017cyclical` mentioned that triangular, Welch, Hann windows produce equivalent results,
    we only implement triangular learning rate policy, also known as **linear cycle**.

    The original implementation of :cite:`smith2017cyclical` can be found from `here <https://github.com/bckenstler/CLR>`_.

    :cite:`smith2017super` uses one cycle with extra epochs.

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
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
    """ Multistep Learning Rate Schedule with warmup

    In :cite:`goyal2017accurate`, warmup is used in order to apply the ``Linear Scaling Rule``.
    Starting from the ``base_lr``, lr gradually increases to ``base_lr * scaling_factor``.
    Then use multiply the learning rate by ``gamma`` at specified milestones.
    See :cite:`ginsburg2018large`

    Args:
        config (:obj:`types.SimpleNamespace`): a global object containing all of the config.
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
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
