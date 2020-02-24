"""Utilities for measuring the performance of a model."""

import math

from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average


class TopKAccuracy(object):
    r"""Top K accuracy of an output.

    Counts a prediction as correct if the target value is in the top ``k``
    predictions, false otherwise, and returns the number of correct
    instances relative to total instances (0.0 to 100.0).

    Args:
        topk (int, optional): The number of top predictions to consider.
            Default: ``1``

    """

    def __init__(self, topk=1):
        self.topk = topk
        self.reset()

    def __call__(self, loss, output, target):
        """Computes the precision@k for the specified values of k

        Args:
            loss (:obj:`torch.Tensor`): Not used for accuracy
            output (:obj:`torch.Tensor`): Predictions of a model
            target (:obj:`torch.Tensor`): Target labels

        Example:
                >>> m = nn.Softmax()
                >>> input = torch.randn(10, 50)
                >>> preds = m(input)
                >>> targets = torch.randint(0, 1, (10,50))
                >>> topk = TopKAccuracy(5)
                >>> precision = topk(preds, targets)

        Returns:
            float
        """
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def reset(self):
        """Reset metric tracking stats"""
        self.top = AverageMeter()

    def update(self, prec, size):
        """Add new measurement to running stats"""
        self.top.update(prec, size)

    def average(self):
        """Average stats."""
        return global_average(self.top.sum, self.top.count)

    @property
    def name(self):
        """str: Name of this metric."""
        return "Prec@{}".format(self.topk)


class Perplexity(object):
    """Language Model Perplexity score."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric tracking stats"""
        self.ppl = AverageMeter()

    def update(self, ppl, size):
        """Add new measurement to running stats"""
        self.ppl.update(ppl, size)

    def average(self):
        """Average stats."""
        return global_average(self.ppl.sum, self.ppl.count)

    @property
    def name(self):
        """str: Name of this metric."""
        return "Perplexity"

    def __call__(self, loss, output, target):
        """Computes the perplexity

        Args:
            loss (:obj:`torch.Tensor`): The loss of a language model.
            output (:obj:`torch.Tensor`): Not Used
            target (:obj:`torch.Tensor`): Not Used

        Returns:
            float
        """
        return math.exp(loss)