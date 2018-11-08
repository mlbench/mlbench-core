"""Utilities for measuring the performance of a model."""

from mlbench_core.utils.pytorch.distributed import global_average
from mlbench_core.utils import AverageMeter


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

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k

        Args:
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
        r"""Name of the metric."""
        return "Prec@{}".format(self.topk)
