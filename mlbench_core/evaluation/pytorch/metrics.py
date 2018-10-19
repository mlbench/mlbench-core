r"""Unilities for measuring the performance of a model."""

from mlbench_core.utils.pytorch.distributed import global_average


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stats."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update stats given input val and n."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TopKAccuracy(object):
    r"""Compute Top K accuracy of an output."""

    def __init__(self, topk=1):
        self.topk = topk
        self.reset()

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def reset(self):
        r"""Reset the stats."""
        self.top = AverageMeter()

    def update(self, prec, size):
        r"""Update stats."""
        self.top.update(prec, size)

    def average(self):
        r"""Average stats."""
        return global_average(self.top.sum, self.top.count)

    @property
    def name(self):
        r"""Name of the metric."""
        return "Prec@{}".format(self.topk)
