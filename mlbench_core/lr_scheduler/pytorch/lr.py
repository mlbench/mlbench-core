# -*- coding: utf-8 -*-

import math
from bisect import bisect_right

import numpy as np
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def const(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)


def triangular_learning_rates(
    optimizer, base_lr, max_lr, cycle_length, scale_fn, extra, mode
):
    """Linearily Scale Learning Rate

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

    if mode == "one_cycle":

        def f(iterations):
            if iterations <= cycle_length:
                cycle = np.floor(1 + iterations / (2 * step_size))
                x = np.abs(iterations / step_size - 2 * cycle + 1)
                lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(
                    cycle, iterations
                )
            else:
                lr = base_lr * extra
            return lr / base_lr

    else:

        def f(iterations):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(
                cycle, iterations
            )
            return lr / base_lr

    # Use base_lr to overwrite the --lr
    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


def cyclical_learning_rates(
    optimizer, mode, gamma, cycle_length, base_lr, max_lr, extra_epochs
):
    """Cyclically Scale Learning Rate

    If one cycle is applied with length smaller than the total number of iterations, then
    use small learning rate for the remaining iterations.

    Since :cite:`smith2017cyclical` mentioned that triangular, Welch, Hann windows produce equivalent results,
    we only implement triangular learning rate policy, also known as **linear cycle**.

    The original implementation of :cite:`smith2017cyclical` can be found from
    `here <https://github.com/bckenstler/CLR>`_.

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
    if mode in ["linear", "triangular", "one_cycle"]:

        def scale_fn(cycle, iterations):
            return 1.0

    elif mode == "triangular2":

        def scale_fn(cycle, iterations):
            return 1 / (2.0 ** (cycle - 1))

    elif mode == "exp_range":

        def scale_fn(cycle, iterations):
            return gamma ** iterations

    else:
        raise ValueError("Cycle mode {} not support.".format(mode))

    return triangular_learning_rates(
        optimizer,
        base_lr,
        max_lr,
        cycle_length=cycle_length,
        scale_fn=scale_fn,
        extra=extra_epochs,
        mode=mode,
    )


def multistep_learning_rates_with_warmup(
    optimizer,
    world_size,
    lr,
    gamma,
    milestones,
    warmup_duration=None,
    warmup_lr=None,
    warmup_linear_scaling=False,
):
    """Multistep Learning Rate Schedule with warmup

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
        raise ValueError(
            "Either both or none of warmup_duration and warmup_lr have to be set"
        )

    scaling_factor = 1

    if warmup_linear_scaling:
        scaling_factor = world_size

    base_lr = lr * scaling_factor

    warmup_init_lr = lr
    if warmup_lr:
        warmup_init_lr = warmup_lr

    if list(milestones) != sorted(milestones):
        raise ValueError(
            "Milestones should be a list of increasing integers."
            "Got {}".format(milestones)
        )

    if warmup_duration >= milestones[0]:
        raise ValueError(
            "The scaling phase should be earlier than the first milestone."
            "Got {} and {}".format(warmup_duration, milestones[0])
        )

    def f(duration):
        if warmup_lr and duration <= warmup_duration:
            warmup_progress = duration / warmup_duration
            lr = warmup_progress * base_lr + (1 - warmup_progress) * warmup_init_lr
        else:
            lr = base_lr * gamma ** bisect_right(milestones, duration)
        return lr / base_lr

    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
    optimizer.base_lrs = [base_lr for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambda=f)


class MultistepLearningRatesWithWarmup(LambdaLR):
    """Multistep Learningrate Scheduler with Warmup Period

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        gamma (float): Decay factor for learning rate
        milestones (:obj:`list` of :obj:`int`): The epochs/steps at which to reduce the
            learning rate
        scaled_lr (float): The LR to reach after warmup
        warmup_init_lr (float): The initial learning rate to use for the warmup epochs. Default: 0
        warmup_duration (int): The number of epochs to perform warmup before regular
            lr scaling starts. Default: 0
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """

    def __init__(
        self,
        optimizer,
        gamma,
        milestones,
        scaled_lr,
        warmup_init_lr=0,
        warmup_duration=0,
    ):
        if list(milestones) != sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers."
                "Got {}".format(milestones)
            )

        if warmup_duration >= milestones[0]:
            raise ValueError(
                "The scaling phase should be earlier than the first milestone."
                "Got {} and {}".format(warmup_duration, milestones[0])
            )

        self.optimizer = optimizer
        self.gamma = gamma
        self.milestones = milestones
        self.warmup_duration = warmup_duration
        self.warmup_init_lr = warmup_init_lr

        self.warmup_scaled_lr = scaled_lr

        # overwrite initial lr
        self.base_lr = warmup_init_lr if warmup_duration > 0 else scaled_lr
        for group in self.optimizer.param_groups:
            group["initial_lr"] = self.base_lr
            group["lr"] = self.base_lr

        super(MultistepLearningRatesWithWarmup, self).__init__(self.optimizer, self.f)

    def f(self, duration):
        # warmup_lr => lr or lr * world_size => ....
        if duration <= self.warmup_duration and self.warmup_duration > 0:
            progress = duration / self.warmup_duration
            lr = progress * self.warmup_scaled_lr + (1 - progress) * self.warmup_init_lr
        else:
            lr = self.warmup_scaled_lr * self.gamma ** bisect_right(
                self.milestones, duration
            )
        return lr / self.base_lr


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """ReduceLROnPlateau but with a linear warm-up period.

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        warmup_init_lr (float): LR at beginning of warm-up
        scaled_lr (float): LR at end of warm-up
        warmup_epochs (int): Number of epochs for warm-up
        batches_per_epoch (int, optional): Number of batches per epoch if we want a warm-up per batch
        **kwargs: Arguments for ReduceLROnPlateau
    """

    def __init__(
        self,
        optimizer,
        warmup_init_lr,
        scaled_lr,
        warmup_epochs,
        batches_per_epoch=None,
        **kwargs
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_duration = warmup_epochs * (
            batches_per_epoch or 1
        )  # To get finer warmup
        self.warmup_init_lr = warmup_init_lr

        self.scaled_lr = scaled_lr
        self.optimizer = optimizer

        self.batch_idx = 0
        self.finished_warmup = warmup_epochs <= 0  # If no warmup

        self.base_lr = scaled_lr if self.finished_warmup else warmup_init_lr
        self._set_lr(self.base_lr)

        super(ReduceLROnPlateauWithWarmup, self).__init__(optimizer, **kwargs)

    def batch_step(self):
        """Function to call when the warm-up is per batch.

        This function will change the learning rate to
        ``
        progress = batch_idx / warmup_duration
        new_lr = progress * scaled_lr + (1 - progress) * warmup_init_lr
        ``
        """
        if self.batch_idx >= self.warmup_duration:
            return
        else:
            self.batch_idx += 1
            progress = self.batch_idx / self.warmup_duration
            new_lr = progress * self.scaled_lr + (1 - progress) * self.warmup_init_lr
            self._set_lr(new_lr)

        # Check if warmup done
        self.finished_warmup = (
            self.finished_warmup or self.batch_idx == self.warmup_duration
        )

    def step(self, metrics, epoch=None):
        """Scheduler step at end of epoch.

        This function will pass the arguments to ReduceLROnPlateau if the warmup is done, and call
        `self.batch_step` if the warm-up is per epoch, to update the LR.

        Args:
            metrics (float): Current loss

        """
        if self.finished_warmup:  # Reduce only if we finished warmup
            super(ReduceLROnPlateauWithWarmup, self).step(metrics, epoch=None)
        else:  # Still in warmup
            if epoch is not None:
                raise ValueError("Epoch argument must be none")
            self.last_epoch += 1

            # This means the warm-up is per epoch not batch, so we need to update it
            if (
                self.warmup_epochs > 0 and self.warmup_epochs == self.warmup_duration
            ):  # warmup per epoch
                self.batch_step()

    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class SparsifiedSGDLR(LambdaLR):
    """Learning rate schedule for sparsifiedSGD (gamma / l2_coef * (t + shifting_param))

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        gamma (float): The constant value in the numerator of the learning rate schedule formula
        l2_coef (float): The regularization rate which is used in the denominator of the learning rate schedule formula
        shifting_param (float): The constant value in the denominator of the learning rate schedule formula
    """

    def __init__(self, optimizer, gamma, l2_coef, shifting_param):
        self.shifting_param = shifting_param
        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            group["initial_lr"] = gamma / l2_coef

        self.optimizer.base_lrs = [gamma / l2_coef for _ in self.optimizer.param_groups]

        super(SparsifiedSGDLR, self).__init__(self.optimizer, self.f)

    def f(self, iteration):
        return 1 / max(1, (self.shifting_param + iteration))


class TimeDecayLR(LambdaLR):
    """
    Time based decay learning rate schedule for SGD (alpha / (t + beta))

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        alpha (float): The constant value in the numerator of the learning rate schedule formula
        beta (float): The constant value in the denominator of the learning rate schedule formula
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """

    def __init__(self, optimizer, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            group["initial_lr"] = alpha / beta

        self.optimizer.base_lrs = [alpha / beta for _ in self.optimizer.param_groups]

        super(TimeDecayLR, self).__init__(self.optimizer, self.f)

    def f(self, iteration):
        return self.beta / (self.beta + iteration)


class SQRTTimeDecayLR(LambdaLR):
    """
    Time based decay learning rate schedule for SGD (alpha / sqrt(t))

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        alpha (float): The constant value in the numerator of the learning rate schedule formula
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """

    def __init__(self, optimizer, alpha):
        self.alpha = alpha
        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            group["initial_lr"] = alpha

        self.optimizer.base_lrs = [alpha for _ in self.optimizer.param_groups]

        super(SQRTTimeDecayLR, self).__init__(self.optimizer, self.f)

    def f(self, iteration):
        return 1.0 / math.sqrt(max(1, iteration))


class ExponentialWarmupMultiStepLR(LambdaLR):
    """
    Learning rate scheduler with exponential warmup and step decay.

    Parameters: warmup_steps, remain_steps and decay_interval accept both
    integers and floats as an input. Integer input is interpreted as
    absolute index of iteration, float input is interpreted as a fraction
    of total training iterations (epochs * steps_per_epoch).

    If decay_interval is None then the decay will happen at regulary spaced
    intervals ('decay_steps' decays between iteration indices
    'remain_steps' and 'iterations').

    Args:
        optimizer: instance of optimizer
        iterations (int): total number of training iterations
        warmup_steps (int): number of warmup iterations
        remain_steps (int|float): start decay at 'remain_steps' iteration
        decay_interval (int|float): interval between LR decay steps
        decay_steps (int): max number of decay steps
        decay_factor (float): decay factor
    """

    def __init__(
        self,
        optimizer,
        iterations,
        warmup_steps=0,
        remain_steps=1.0,
        decay_interval=None,
        decay_steps=4,
        decay_factor=0.5,
    ):
        # iterations before learning rate reaches base LR
        self.warmup_steps = self.convert_relative_stepsize(warmup_steps, iterations)

        # iteration at which decay starts
        self.remain_steps = self.convert_relative_stepsize(remain_steps, iterations)

        # number of steps between each decay
        if decay_interval is None:
            # decay at regulary spaced intervals
            decay_iterations = iterations - self.remain_steps
            self.decay_interval = decay_iterations // decay_steps
            self.decay_interval = max(self.decay_interval, 1)
        else:
            self.decay_interval = self.convert_relative_stepsize(
                decay_interval, iterations
            )

        # multiplicative decay factor
        self.decay_factor = decay_factor

        # max number of decay steps
        self.decay_steps = decay_steps

        if self.warmup_steps > self.remain_steps:
            self.warmup_steps = self.remain_steps

        super(ExponentialWarmupMultiStepLR, self).__init__(optimizer, self.f)

    @staticmethod
    def convert_relative_stepsize(param, total):
        if isinstance(param, float):
            param = int(param * total)
        return param

    def f(self, duration):
        factor = 1
        if duration <= self.warmup_steps:
            # exponential lr warmup
            if self.warmup_steps != 0:
                warmup_factor = math.exp(math.log(0.01) / self.warmup_steps)
            else:
                warmup_factor = 1.0
            factor = warmup_factor ** (self.warmup_steps - self.last_epoch)

        elif self.last_epoch >= self.remain_steps:
            # step decay
            decay_iter = self.last_epoch - self.remain_steps
            num_decay_steps = decay_iter // self.decay_interval + 1
            num_decay_steps = min(num_decay_steps, self.decay_steps)
            factor = self.decay_factor ** num_decay_steps
        return factor


class SQRTTimeDecayLRWithWarmup(LambdaLR):
    """SQRT learning rate scheduler with warm-up steps

        During warmup:
          ```
          lrs = torch.linspace(warmup_init_lr, base_lr, warmup_steps)
          lr = lrs[update_num]
          ```
        After warmup:
          ```
          lr = decay_factor / sqrt(update_num)
          ```
        where
          ```decay_factor = base_lr * sqrt(warmup_steps)```

    Args:
        optimizer (:obj:`torch.optim`): The optimizer
        base_lr (float): The base LR after warm-up
        warmup_init_lr (float): LR at start of training
        warmup_steps (int): Number of warm-up steps

    """

    def __init__(self, optimizer, base_lr, warmup_init_lr, warmup_steps):
        self.warmup_steps = warmup_steps

        self.lr_step = (base_lr - warmup_init_lr) / warmup_steps
        self.init_lr = warmup_init_lr
        self.base_lr = base_lr
        self.optimizer = optimizer

        super(SQRTTimeDecayLRWithWarmup, self).__init__(optimizer, self.f)

    def f(self, iteration):
        # Warmup
        if iteration < self.warmup_steps:
            # Linear increase with the steps
            lr = self.init_lr + self.lr_step * iteration
            factor = lr / self.base_lr
        else:
            factor = (self.warmup_steps / iteration) ** 0.5
        return factor
