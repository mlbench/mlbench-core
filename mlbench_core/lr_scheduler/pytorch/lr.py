# -*- coding: utf-8 -*-

import math
from bisect import bisect_right

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


class LRLinearWarmUp(LambdaLR):
    """Applies linear warmup to learning rate.

    At the first iteration, lr will be `initial_lr`, and will linearly increase to `scaled_lr`
    at iteration `warmup_duration + 1` (i.e `warmup_duration` steps of warm-up)

    In :cite:`goyal2017accurate`, warmup is used in order to apply the ``Linear Scaling Rule``.
    Starting from the ``base_lr``, lr gradually increases to ``base_lr * scaling_factor``.

    Args:
        init_lr (float): Initial LR at beginning of warmup
        scaled_lr (float): LR at end of warmup
        warmup_duration (float): Duration of warmup
    """

    def __init__(self, optimizer, init_lr, scaled_lr, warmup_duration):
        self.warmup_duration = warmup_duration
        self.scaled_lr = scaled_lr
        self.init_lr = init_lr
        self.optimizer = optimizer

        # overwrite initial lr
        for group in self.optimizer.param_groups:
            group["initial_lr"] = self.scaled_lr
            group["lr"] = self.scaled_lr

        super().__init__(self.optimizer, self.f)

    def f(self, duration):
        factor = 1
        if self.warmup_duration > 0 and duration <= self.warmup_duration:
            progress = duration / self.warmup_duration
            factor = progress + ((1 - progress) * self.init_lr) / self.scaled_lr
        return factor

    @property
    def duration(self):
        return self.warmup_duration


class MultiStepLRLinearWarmUp(LambdaLR):
    """Multi-step Learning rate Scheduler with Linear Warm-up Period

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        gamma (float): Decay factor for learning rate
        milestones (:obj:`list` of :obj:`int`): The epochs/steps at which to reduce the
            learning rate
        scaled_lr (float): The LR to reach after warmup
        warmup_init_lr (float): The initial learning rate to use for the warmup epochs. Default: 0
        warmup_duration (int): The number of epochs to perform warmup before regular
            lr scaling starts. Default: 0
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

        self.gamma = gamma
        self.milestones = milestones
        self.warmup = LRLinearWarmUp(
            optimizer=optimizer,
            init_lr=warmup_init_lr,
            scaled_lr=scaled_lr,
            warmup_duration=warmup_duration,
        )

        super(MultiStepLRLinearWarmUp, self).__init__(optimizer, self.f)

    def f(self, duration):
        # warmup_lr => lr or lr * world_size => ....
        if duration <= self.warmup.duration:
            factor = self.warmup.f(duration)
        else:
            factor = self.gamma ** bisect_right(self.milestones, duration)
        return factor


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
        beta (float): The constant value in the denominator of the learning rate schedule formula
    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """

    def __init__(self, optimizer, beta):
        self.beta = beta
        super(TimeDecayLR, self).__init__(optimizer, self.f)

    def f(self, iteration):
        return 1 / (self.beta + iteration)


class SQRTTimeDecayLR(LambdaLR):
    """
    Time based decay learning rate schedule for SGD (alpha / sqrt(t))

    Returns:
        A learning rate scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`)
    """

    def __init__(self, optimizer):
        super(SQRTTimeDecayLR, self).__init__(optimizer, self.f)

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
    """SQRT learning rate scheduler with Linear warm-up steps

        During warmup:
          ```
          lrs = torch.linspace(warmup_init_lr, base_lr, warmup_steps)
          lr = lrs[update_num]
          ```
        After warmup:
          ```
          lr = base_lr * decay_factor
          ```
        where
          ```decay_factor = sqrt(warmup_steps / current_iteration)```

    Args:
        optimizer (:obj:`torch.optim`): The optimizer
        base_lr (float): The base LR after warm-up
        warmup_init_lr (float): LR at start of training
        warmup_steps (int): Number of warm-up steps

    """

    def __init__(self, optimizer, base_lr, warmup_init_lr, warmup_steps):
        self.warmup = LRLinearWarmUp(
            optimizer=optimizer,
            init_lr=warmup_init_lr,
            scaled_lr=base_lr,
            warmup_duration=warmup_steps,
        )

        super(SQRTTimeDecayLRWithWarmup, self).__init__(optimizer, self.f)

    def f(self, iteration):
        # Warmup
        if iteration <= self.warmup.duration:
            factor = self.warmup.f(iteration)
        else:
            factor = (self.warmup.duration / iteration) ** 0.5
        return factor


class PolyDecayLRLinearWarmup(LambdaLR):
    """Polynomial decay of learning rate with linear warmup.

    During warmup:
      ```
      lrs = torch.linspace(warmup_init_lr, base_lr, warmup_steps)
      lr = lrs[update_num]
      ```
    After warmup:
      ```
      lr = base_lr * decay_factor
      ```
    where
      ```decay_factor = (1 - ((iteration - warmup_duration) / decay_steps))^pow```

    """

    def __init__(
        self,
        optimizer,
        init_lr,
        scaled_lr,
        warmup_duration,
        decay_steps,
        min_lr=0,
        power=2,
    ):
        """

        Args:
            optimizer (:obj:`torch.optim.Optimizer`): Optimizer to use
            init_lr (float): Initial LR at start of training
            scaled_lr (float): Scaled LR to reach after warmup
            warmup_duration (int): Warm-up steps
            decay_steps (int): Decay steps for power decay
            power (float): Power to use
        """
        self.warmup = LRLinearWarmUp(
            optimizer=optimizer,
            init_lr=init_lr,
            scaled_lr=scaled_lr,
            warmup_duration=warmup_duration,
        )
        self.pow = power
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        super(PolyDecayLRLinearWarmup, self).__init__(optimizer, self.f)

        self.factor = 1

    def f(self, iteration):
        # Warmup
        if iteration <= self.warmup.duration:
            self.factor = self.warmup.f(iteration)
        else:
            diff = iteration - self.warmup.duration

            # If remaining decay steps
            if diff <= self.decay_steps:
                progress = diff / self.decay_steps
                self.factor = math.pow(
                    (1 - progress)
                    + math.pow(self.min_lr / self.warmup.scaled_lr, 1 / self.pow)
                    * progress,
                    self.pow,
                )
        return self.factor
