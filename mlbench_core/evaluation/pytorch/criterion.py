"""Customized Loss Functions."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

try:
    from apex.contrib.xentropy import SoftmaxCrossEntropyLoss

    apex_installed = True
except ImportError as e:
    apex_installed = False


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
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will
            be applied,
            'elementwise_mean': the sum of the output will be divided by the
            number of
            elements in the output, 'sum': the output will be summed. Note:
            :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated,
            and in the meantime,
            specifying either of those two args will override
            :attr:`reduction`. Default: 'elementwise_mean'
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        l1=0.0,
        l2=0.0,
        model=None,
        reduction="elementwise_mean",
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
        l2_loss = sum(param.norm(2) ** 2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
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
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will
            be applied,
            'elementwise_mean': the sum of the output will be divided by the
            number of
            elements in the output, 'sum': the output will be summed. Note:
            :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated,
            and in the meantime,
            specifying either of those two args will override
            :attr:`reduction`. Default: 'elementwise_mean'
    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        l1=0.0,
        l2=0.0,
        model=None,
        reduction="elementwise_mean",
    ):
        super(MSELossRegularized, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input_, target):
        output = F.mse_loss(input_, target, reduction=self.reduction)
        l2_loss = sum(param.norm(2) ** 2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
        return output


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, padding_idx, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        Args:
            padding_idx (int): Code for padding char
            smoothing (float): Smoothing value
        """
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1).type(torch.float32)

        non_pad_mask = target != self.padding_idx
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()


# Taken from MLPerf
class LabelSmoothedCrossEntropy(_Loss):
    def __init__(
        self,
        model,
        label_smoothing,
        padding_idx,
        sentence_avg=False,
        fast_xentropy=False,
    ):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.eps = label_smoothing
        self.padding_idx = padding_idx
        self.sentence_avg = sentence_avg

        if fast_xentropy and apex_installed:
            self.xentropy_func = SoftmaxCrossEntropyLoss.apply
        else:
            self.xentropy_func = None

        self.model = model

    def forward(self, sample, output, reduce=True):

        target = sample["target"]
        target = target.view(-1, 1)
        if self.xentropy_func is not None:
            assert (output[0].dtype == torch.float16) or (
                output[0].dtype == torch.float32
            ), "Unsupported data types"
            output = output[0].view(
                output[0].size(0) * output[0].size(1), output[0].size(2)
            )
            labels = target.view(target.size(0) * target.size(1))
            losses = self.xentropy_func(
                output,
                labels,
                self.eps,
                self.padding_idx,
                output[0].dtype == torch.float16,
            )
            loss = losses.sum()
        else:
            lprobs = self.model.get_normalized_probs(output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            non_pad_mask = target.ne(self.padding_idx)
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            if reduce:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            loss = (1.0 - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = target.size(0) if self.sentence_avg else sample["ntokens"]
        return loss, sample_size
