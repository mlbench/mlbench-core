r"""Customized Loss Functions."""

import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class BCELossRegularized(_WeightedLoss):
    """
    Add l1/l2 regularized terms to Binary Cross Entropy (BCE).
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
    """
    Add l1/l2 regularized terms to Mean Squared Error (MSE).
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
