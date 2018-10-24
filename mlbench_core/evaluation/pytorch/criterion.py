"""Customized Loss Functions."""

import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class BCELossRegularized(_WeightedLoss):
    """Binary Cross Entropy (BCE) with l1/l2 regularization.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        l1 (float, optional): The scale of the L1 regularization. Default: ``0.0``
        l2 (float, optional): The scale of the L2 regularization. Default: ``0.0``
        model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    """

    def __init__(self, weight=None, size_average=None, reduce=None, l1=0.0, l2=0.0, model=None,
                 reduction='elementwise_mean'):
        super(BCELossRegularized, self).__init__(
            weight, size_average, reduce, reduction)
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input_, target):
        output = F.binary_cross_entropy(
            input_, target, weight=self.weight, reduction=self.reduction)
        l2_loss = sum(param.norm(2)**2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
        return output


class MSELossRegularized(_WeightedLoss):
    """Mean Squared Error (MSE) with l1/l2 regularization.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        l1 (float, optional): The scale of the L1 regularization. Default: ``0.0``
        l2 (float, optional): The scale of the L2 regularization. Default: ``0.0``
        model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
    """

    def __init__(self, weight=None, size_average=None, reduce=None, l1=0.0, l2=0.0, model=None,
                 reduction='elementwise_mean'):
        super(MSELossRegularized, self).__init__(
            weight, size_average, reduce, reduction)
        self.l2 = l2
        self.l1 = l1
        self.model = model

    def forward(self, input_, target):
        output = F.mse_loss(input_, target, reduction=self.reduction)
        l2_loss = sum(param.norm(2)**2 for param in self.model.parameters())
        output += self.l2 / 2 * l2_loss
        l1_loss = sum(param.norm(1) for param in self.model.parameters())
        output += self.l1 * l1_loss
        return output
