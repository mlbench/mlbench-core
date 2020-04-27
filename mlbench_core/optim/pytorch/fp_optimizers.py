# import ctypes
import logging
import math

import torch
import torch.distributed as dist
from mlbench_core.utils.pytorch.distributed import (
    AllReduceAggregationHVD,
    AllReduceAggregation,
)
from torch.nn.utils import clip_grad_norm_

try:
    from apex.optimizers import FusedAdam
    from apex import amp
except ImportError as e:
    pass

logger = logging.getLogger("mlbench")


class FP16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor

    Args:
        fp16_model (`obj`:torch.nn.Module): model (previously casted to half)
        world_size (int): Distributed world size
        use_cuda (bool): Use cuda tensors for aggregation
        use_horovod (bool): Use Horovod for aggregation
        by_layer (bool): Aggregate by layer
        grad_clip (float): coefficient for gradient clipping, max L2 norm of the gradients
        loss_scale (int):  initial loss scale
        dls_downscale (int): loss downscale factor, loss scale is divided by this factor when NaN/INF occurs in the gradients
        dls_upscale (int): loss upscale factor, loss scale is multiplied by this factor if previous dls_upscale_interval batches finished successfully
        dls_upscale_interval (int): interval for loss scale upscaling
        average_models (bool): Average the models
    """

    def __init__(
        self,
        fp16_model,
        world_size,
        use_cuda=False,
        use_horovod=False,
        by_layer=False,
        grad_clip=float("inf"),
        loss_scale=1024,
        dls_downscale=2,
        dls_upscale=2,
        dls_upscale_interval=128,
        average_models=True,
    ):
        self.use_cuda = use_cuda

        self.fp16_model = fp16_model
        self.fp16_params, self.fp32_params = self.initialize_flat_fp32_weight()
        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip
        self.world_size = dist.get_world_size()

        self.optimizer = None

        if use_horovod:
            self.agg = AllReduceAggregationHVD(
                world_size=world_size, use_cuda=use_cuda
            ).agg_grad(by_layer=by_layer)
        else:
            self.agg = AllReduceAggregation(
                world_size=world_size, use_cuda=use_cuda
            ).agg_grad(by_layer=by_layer)

        if average_models:
            self.agg_mode = "avg"
        else:
            raise NotImplementedError("Only average model is supported right now.")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # Flattening master weight
    def initialize_flat_fp32_weight(self):
        """ Initializes the model's parameters in fp32 and fp16

        Returns:
            (torch.Tensor, torch.Tensor): The Parametrs in fp16 and fp32
        """
        # Set all gradients to None
        for p in self.fp16_model.parameters():
            p.grad = None

        # Count number of parameters per layer
        nelem = 0
        for p in self.fp16_model.parameters():
            nelem += p.numel()
        fp32_params = torch.empty(
            nelem,
            dtype=torch.float32,
            device=torch.device("cuda" if self.use_cuda else "cpu"),
        )
        fp16_params = torch.empty(
            nelem,
            dtype=torch.float16,
            device=torch.device("cuda" if self.use_cuda else "cpu"),
        )

        pointer = 0
        for p in self.fp16_model.parameters():
            nelem = p.numel()
            fp32_params[pointer : pointer + nelem].copy_(p.data.view(-1))
            fp16_params[pointer : pointer + nelem].copy_(p.data.view(-1))
            pointer += nelem

        fp32_params = torch.nn.Parameter(fp32_params, requires_grad=True)
        fp32_params.grad = torch.autograd.Variable(
            fp32_params.data.new(*fp32_params.size())
        )

        fp16_params = torch.nn.Parameter(fp16_params, requires_grad=True)
        fp16_params.grad = torch.autograd.Variable(
            fp16_params.data.new(*fp16_params.size())
        )

        return fp16_params, fp32_params

    @staticmethod
    def fp16_to_fp32_flat_grad(fp32_params, fp16_model):
        """ Copies the parameters in `fp16_model` into `fp32_params` in-place

        Args:
            fp32_params (torch.Tensor): Parameters in fp32
            fp16_model (torch.nn.Module): Model in fp16

        """
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            fp32_params.grad.data[pointer : pointer + nelem].copy_(p.grad.data.view(-1))
            pointer += nelem

    @staticmethod
    def fp32_to_fp16_grads(fp16_model, fp32_params):
        """ Copies the parameters in `fp32_params` into `fp16_model` in-place

         Args:
             fp16_model (torch.nn.Module): Model in fp16
             fp32_params (torch.Tensor): Parameters in fp32

         """
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            p.data.view(-1).copy_(fp32_params.data[pointer : pointer + nelem])
            pointer += nelem

    def backward_loss(self, loss):
        """ Scales and performs backward on the given loss

        Args:
            loss (torch.nn.Module): The loss

        """
        loss *= self.loss_scale
        loss.backward()

    def step(self, closure=None):
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
        """

        scaling_factor = self.loss_scale

        # Aggregate gradients
        self.agg(self.fp16_model, self.agg_mode)
        # Cast fp16 params to fp32 for optimizer
        self.fp16_to_fp32_flat_grad(self.fp32_params, self.fp16_model)

        if scaling_factor != 1.0:
            self.fp32_params.grad.data /= scaling_factor
        norm = clip_grad_norm_([self.fp32_params], self.grad_clip)

        updated = False
        if math.isfinite(norm):
            self.optimizer.step(closure=closure)
            self.fp32_to_fp16_grads(self.fp16_model, self.fp32_params)
            self.since_last_invalid += 1
            updated = True
        else:
            self.loss_scale /= self.dls_downscale
            self.since_last_invalid = 0
            logger.info(f"Skipped batch, new scale: {self.loss_scale}")

        if self.since_last_invalid >= self.dls_upscale_interval:
            self.loss_scale *= self.dls_upscale
            self.loss_scale = min(self.loss_scale, 8192.0)
            self.since_last_invalid = 0

        for p in self.fp16_model.parameters():
            p.grad = None

        return updated

    def zero_grad(self):
        self.optimizer.zero_grad()


class FP32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.

    Args:
        model (`obj`:torch.nn.Module): model
        world_size (int): Distributed world size
        use_cuda (bool): Use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer
        grad_clip (float): coefficient for gradient clipping, max L2 norm of the gradients
        average_models (bool): Average the models
    """

    def __init__(
        self,
        model,
        world_size,
        use_cuda=False,
        by_layer=False,
        grad_clip=None,
        average_models=True,
    ):
        self.model = model
        self.grad_clip = grad_clip
        self.optimizer = None
        self.agg = AllReduceAggregation(
            world_size=world_size, use_cuda=use_cuda
        ).agg_grad(by_layer=by_layer)
        if average_models:
            self.agg_mode = "avg"
        else:
            raise NotImplementedError("Only average model is supported right now.")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def step(self, closure=None):
        """
        Performs one step of the optimizer.
        """
        if self.grad_clip != float("inf"):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.agg(self.model, self.agg_mode)
        self.optimizer.step(closure=closure)
        return True

    def backward_loss(self, loss):
        loss.backward()

    def zero_grad(self):
        self.optimizer.zero_grad()


class AMPOptimizer:
    """
    Optimizer compatible with AMP.
    Uses AMP to apply loss scaling, computes backward and applies weight
    update.

    Args:
        model (`obj`:torch.nn.Module): model
        grad_clip (float): coefficient for gradient clipping, max L2 norm of the gradients
        loss_scale (int):  initial loss scale
        dls_upscale_interval (int): interval for loss scale upscaling
        average_models (bool): Average the models
        world_size (int): Distributed world size
        use_cuda (bool): Use cuda tensors for aggregation
        by_layer (bool): Aggregate by layer
        use_horovod (bool): Use Horovod for aggregation
    """

    def __init__(
        self,
        model,
        grad_clip=None,
        loss_scale=8192,
        dls_upscale_interval=128,
        average_models=True,
        world_size=1,
        use_cuda=False,
        by_layer=False,
        use_horovod=False,
    ):
        self.model = model
        self.grad_clip = grad_clip
        self.optimizer = None
        loss_scaler = amp._amp_state.loss_scalers[0]
        loss_scaler._loss_scale = loss_scale
        loss_scaler._scale_seq_len = dls_upscale_interval

        if average_models:
            self.agg_mode = "avg"
        else:
            raise NotImplementedError("Only average model is supported right now.")

        if use_horovod:
            self.agg = AllReduceAggregationHVD(
                world_size=world_size, use_cuda=use_cuda
            ).agg_grad(by_layer=by_layer)
        else:
            self.agg = AllReduceAggregation(
                world_size=world_size, use_cuda=use_cuda
            ).agg_grad(by_layer=by_layer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def backward_loss(self, loss):
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

    def step(self, closure=None):
        """
        Performs one step of the optimizer.
        """
        if self.grad_clip != float("inf"):
            clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)

        self.agg(self.model, self.agg_mode)
        self.optimizer.step(closure=closure)
        return True

    def zero_grad(self):
        self.optimizer.zero_grad()
