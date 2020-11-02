import logging

import torch
from torch.nn.utils import clip_grad_norm_

from mlbench_core.aggregation.pytorch.centralized import (
    AVG_CUSTOM,
    AVG_WORLD,
    AllReduceAggregation,
    AllReduceAggregationHVD,
)

logger = logging.getLogger("mlbench")


class DynamicLossScaler:
    def __init__(
        self, init_scale=2.0 ** 15, scale_factor=2.0, scale_window=2000, max_scale=None
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = -1
        self.max_scale = max_scale

    def update_scale(self, overflow):
        if overflow:
            self.loss_scale /= self.scale_factor
            self._last_overflow_iter = self._iter
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            if self.max_scale is not None:
                self.loss_scale = min(self.loss_scale, self.max_scale)
        self._iter += 1

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float("inf") or grad_norm != grad_norm:
            return True
        return False


class FP16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor

    Args:
        fp16_model (:obj:`torch.nn.Module`): model (previously casted to half)
        world_size (int): Distributed world size
        use_cuda (bool): Use cuda tensors for aggregation
        use_horovod (bool): Use Horovod for aggregation
        by_layer (bool): Aggregate by layer
        grad_clip (float): coefficient for gradient clipping, max L2 norm of the gradients
        init_scale (int):  initial loss scale
        scale_factor (float): Factor for upscale/dowscale
        scale_window (int): interval for loss scale upscaling
        average_world (bool): Average the gradients by world size
        average_custom (bool): Divide gradients by given denominator at each step, instead
            of `world_size`
        divide_before (bool): Divide gradients before reduction (default: False)
    """

    def __init__(
        self,
        fp16_model,
        world_size,
        use_cuda=False,
        use_horovod=False,
        by_layer=False,
        grad_clip=float("inf"),
        init_scale=1024,
        scale_factor=2,
        scale_window=128,
        max_scale=None,
        min_scale=1e-4,
        average_world=False,
        average_custom=False,
        divide_before=False,
    ):
        self.use_cuda = use_cuda

        self.fp16_model = fp16_model
        self.fp32_params = self.initialize_flat_fp32_weight()

        self.loss_scaler = DynamicLossScaler(
            init_scale=init_scale,
            scale_factor=scale_factor,
            scale_window=scale_window,
            max_scale=max_scale,
        )
        self.min_scale = min_scale
        self.grad_clip = grad_clip

        self.optimizer = None

        if use_horovod:
            self.agg = AllReduceAggregationHVD(
                world_size=world_size, use_cuda=use_cuda, divide_before=divide_before
            ).agg_grad(by_layer=by_layer)
        else:
            self.agg = AllReduceAggregation(
                world_size=world_size, use_cuda=use_cuda, divide_before=divide_before
            ).agg_grad(by_layer=by_layer)

        if average_world:
            self.agg_mode = AVG_WORLD
        elif average_custom:
            self.agg_mode = AVG_CUSTOM
        else:
            raise NotImplementedError("Only average model is supported right now.")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # Flattening master weight
    def initialize_flat_fp32_weight(self):
        """Initializes the model's parameters in fp32

        Returns:
            (:obj:`torch.Tensor`): The Parameters in fp32
        """
        # Get all params that require gradient
        params = [p for p in self.fp16_model.parameters() if p.requires_grad]
        total_param_size = sum(p.data.numel() for p in params)

        # Create new fp32 params
        fp32_params = params[0].new(0).float().new(total_param_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            fp32_params[offset : offset + numel].copy_(p.data.view(-1))
            offset += numel

        fp32_params = torch.nn.Parameter(fp32_params, requires_grad=True)

        fp32_params.grad = torch.autograd.Variable(
            fp32_params.data.new(*fp32_params.size())
        )

        return fp32_params

    @staticmethod
    def fp16_to_fp32_flat_grad(fp32_params, fp16_model):
        """Copies the parameters in `fp16_model` into `fp32_params` in-place

        Args:
            fp32_params (torch.Tensor): Parameters in fp32
            fp16_model (torch.nn.Module): Model in fp16

        """
        flat_grads = torch.cat(
            [p.grad.data.view(-1) for p in fp16_model.parameters() if p.requires_grad]
        )
        fp32_params.grad = flat_grads.to(torch.float32)

    @staticmethod
    def fp32_to_fp16_weights(fp16_model, fp32_params):
        """Copies the parameters in `fp32_params` into `fp16_model` in-place

        Args:
            fp16_model (torch.nn.Module): Model in fp16
            fp32_params (torch.Tensor): Parameters in fp32

        """
        with torch.no_grad():
            pointer = 0
            for p in fp16_model.parameters():
                if not p.requires_grad:
                    continue
                nelem = p.numel()
                p.data.copy_(
                    fp32_params.data[pointer : pointer + nelem].view_as(p.data)
                )
                pointer += nelem

    def backward_loss(self, loss):
        """Scales and performs backward on the given loss

        Args:
            loss (torch.nn.Module): The loss

        """
        loss *= self.loss_scaler.loss_scale
        loss.backward()

    def step(self, closure=None, tracker=None, multiplier=1, denom=None):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
            multiplier (float): Multiplier for gradient scaling. Gradient will be scaled using
                `scaled_grad = reduced_grad / (loss_scaler * multiplier)`
            denom (Optional[:obj:`torch.Tensor`]): Custom denominator to average by
                Use with `average_batch`. (default: `None`)
        """

        scaling_factor = self.loss_scaler.loss_scale * multiplier

        # Aggregate gradients
        self.agg(self.fp16_model, self.agg_mode, denom=denom)

        if tracker:
            tracker.record_batch_agg()

        # Cast fp16 grads to fp32 for optimizer
        self.fp16_to_fp32_flat_grad(self.fp32_params, self.fp16_model)

        # UnScale gradients
        if scaling_factor != 1.0:
            self.fp32_params.grad.data.div_(scaling_factor)

        # Clip and compute gradient norm
        norm = clip_grad_norm_([self.fp32_params], self.grad_clip)
        updated = False
        overflow = self.loss_scaler.has_overflow(norm)
        self.loss_scaler.update_scale(overflow)

        if not overflow:
            self.optimizer.step(closure=closure)
            self.fp32_to_fp16_weights(self.fp16_model, self.fp32_params)
            updated = True
        else:
            if self.loss_scaler.loss_scale <= self.min_scale:
                raise Exception(
                    "Minimum loss scale ({}) reached".format(self.min_scale)
                )
            logger.info(f"Skipped batch, new scale: {self.loss_scaler.loss_scale}")

        if tracker:
            tracker.record_batch_opt_step()
        return updated

    def zero_grad(self):
        """Resets the gradients of the optimizer and fp16_model"""
        self.optimizer.zero_grad()
        self.fp16_model.zero_grad()
