"""Customized Loss Functions."""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

try:
    from apex.contrib.xentropy import SoftmaxCrossEntropyLoss

    apex_installed = True
except ImportError as e:
    apex_installed = False


def _l2_regularization(l2, model):
    """Computes the L2 regularization for the given model

    Args:
        l2 (float): L2 parameter
        model (:obj:`torch.nn.Module`): Model to use

    Returns:
        float: L2 loss (i.e. (l2 * l2_norm(params)) / 2)
    """
    l2_loss = sum(param.norm(2) ** 2 for param in model.parameters())
    return l2 / 2 * l2_loss


def _l1_regularization(l1, model):
    """Computes the L1 regularization for the given model

    Args:
        l1 (float): L1 parameter
        model (:obj:`torch.nn.Module`): Model to use

    Returns:
        float: L1 loss (i.e. l1 * l1_norm(params))
    """
    l1_loss = sum(param.norm(1) for param in model.parameters())
    return l1 * l1_loss


class BCELossRegularized(_WeightedLoss):
    """Binary Cross Entropy (BCE) with l1/l2 regularization.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise,
            it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        By default,
            the losses are averaged over each loss element in the batch.
            Note that for
            some losses, there multiple elements per sample. If the field
            :attr:`size_average`
            is set to ``False``, the losses are instead summed for each
            minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By
        default, the
            losses are averaged or summed over observations for each
            minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``,
            returns a loss per
            batch element instead and ignores :attr:`size_average`. Default:
            ``True``
        l1 (float, optional): The scale of the L1 regularization. Default:
        ``0.0``
        l2 (float, optional): The scale of the L2 regularization. Default:
        ``0.0``
        model (:obj:`torch.nn.Module`): a pytorch model to be trained and
        validated.
        reduction (string, optional): Specifies the reduction to apply to
        the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will
            be applied,
            'mean': the sum of the output will be divided by the
            number of
            elements in the output, 'sum': the output will be summed. Note:
            :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated,
            and in the meantime,
            specifying either of those two args will override
            :attr:`reduction`. Default: 'mean'
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        l1=0.0,
        l2=0.0,
        model=None,
        reduction="mean",
    ):
        super(BCELossRegularized, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input_, target):
        output = F.binary_cross_entropy(
            input_, target.float(), weight=self.weight, reduction=self.reduction
        )
        l2_loss = _l2_regularization(self.l2, self.model)
        l1_loss = _l1_regularization(self.l1, self.model)
        output += l1_loss + l2_loss
        return output


class MSELossRegularized(_WeightedLoss):
    """Mean Squared Error (MSE) with l1/l2 regularization.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise,
            it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
        By default,
            the losses are averaged over each loss element in the batch.
            Note that for
            some losses, there multiple elements per sample. If the field
            :attr:`size_average`
            is set to ``False``, the losses are instead summed for each
            minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By
        default, the
            losses are averaged or summed over observations for each
            minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``,
            returns a loss per
            batch element instead and ignores :attr:`size_average`. Default:
            ``True``
        l1 (float, optional): The scale of the L1 regularization. Default:
        ``0.0``
        l2 (float, optional): The scale of the L2 regularization. Default:
        ``0.0``
        model (:obj:`torch.nn.Module`): a pytorch model to be trained and
        validated.
        reduction (string, optional): Specifies the reduction to apply to
        the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will
            be applied,
            'mean': the sum of the output will be divided by the
            number of
            elements in the output, 'sum': the output will be summed. Note:
            :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated,
            and in the meantime,
            specifying either of those two args will override
            :attr:`reduction`. Default: 'mean'
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        l1=0.0,
        l2=0.0,
        model=None,
        reduction="mean",
    ):
        super(MSELossRegularized, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input_, target):
        output = F.mse_loss(input_, target, reduction=self.reduction)
        l2_loss = _l2_regularization(self.l2, self.model)
        l1_loss = _l1_regularization(self.l1, self.model)
        output += l1_loss + l2_loss
        return output


class LabelSmoothing(_Loss):
    """
    NLL loss with label smoothing.

    Args:
        padding_idx (int): Code for padding char
        smoothing (float): Smoothing value
        fast_xentropy (bool): Use `apex.contrib.xentropy.SoftmaxCrossEntropyLoss`
    """

    def __init__(self, padding_idx, smoothing=0.0, fast_xentropy=False):

        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.fast_xentropy = fast_xentropy

        if fast_xentropy:
            if apex_installed:
                self.xentropy_func = SoftmaxCrossEntropyLoss.apply
            else:
                raise ImportError("Cannot use fast xentropy without apex")
        else:
            self.xentropy_func = None

    def forward(self, x, target):
        if self.fast_xentropy:
            assert (x.dtype == torch.float16) or (
                x.dtype == torch.float32
            ), "Unsupported data types"
            loss = self.xentropy_func(
                x,
                target,
                self.smoothing,
                self.padding_idx,
                x.dtype == torch.float16,
            )

        else:
            logprobs = torch.nn.functional.log_softmax(x, dim=-1).type(torch.float32)
            non_pad_mask = target != self.padding_idx
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)[non_pad_mask]
            smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()
