import argparse
import time

from .log_metrics import LogMetrics


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


class Tracker(object):
    """A class to track running stats."""
    batch_times = []
    validation_times = []
    epoch_stats = {}
    records = []
    history = []
    cumulative_train_time = []
    cumulative_val_time = []
    best_epoch = 0
    current_epoch = 0
    best_metric_value = 0
    is_training = True

    train_prefix = 'train_'
    val_prefix = 'val_'

    def __init__(self, metrics, run_id, rank):
        self.metrics = metrics
        self.run_id = run_id
        self.rank = rank
        self.reset_epoch_stats()

        self.primary_metric = metrics[0]

    def train(self):
        self.is_training = True

    def validation(self):
        self.is_training = False

    def validation_start(self):
        self.validation_times = [('start', time.time())]

    def validation_end(self):
        self.validation_times = [('end', time.time())]
        self.cumulative_val_time.append(
            self.validation_times[-1][1]
            - self.validation_times[0][1])

    def batch_start(self):
        self.batch_times = []
        self.record_batch_step("start")

    def batch_end(self):
        self.record_batch_step("end")

        if len(self.batch_times) > 2:
            self.cumulative_train_time.append(
                self.batch_times[-1][1]
                - self.batch_times[0][1])

    def epoch_end(self):
        self.current_epoch += 1
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        stat_names = [k for k in ["loss"] + [m.name for m in self.metrics]]

        stat_names = [self.train_prefix + k for k in stat_names] \
            + [self.val_prefix + k for k in stat_names]

        self.epoch_stats = {
            k: AverageMeter()
            for k in stat_names}

    def record_batch_step(self, name):
        self.batch_times.append((name, time.time()))

    def record_stat(self, name, value, n=1, log_to_api=False):
        prefix = self.train_prefix

        if not self.is_training:
            prefix = self.val_prefix

        name = prefix + name

        self.epoch_stats[name].update(value, n)

        self.history.append((self.run_id, self.rank, name, value, time.time()))

        if log_to_api:
            LogMetrics.log(
                self.run_id,
                self.rank,
                self.current_epoch,
                name,
                value
            )

    def record_loss(self, value, n=1, log_to_api=False):
        self.record_stat("loss", value, n, log_to_api)

    def record_metric(self, metric, value, n=1, log_to_api=False):
        self.record_stat(metric.name, value, n, log_to_api)

        if metric.name == self.primary_metric.name and not self.is_training:
            self.update_primary_metric(value)

    def update_primary_metric(self, new_metric_value):

        if new_metric_value > self.best_metric_value:
            self.best_metric_value = new_metric_value
            self.best_epoch = self.current_epoch

    def is_best(self):
        return self.current_epoch == self.best_epoch

    def __str__(self):
        prefix = self.train_prefix

        if not self.is_training:
            prefix = self.val_prefix

        # loss
        str_builder = ['loss={:6.2e}'.format(self.epoch_stats[prefix + 'loss'].avg)]

        # metrics
        for metric in self.metrics:
            str_builder.append("{} {:.2e}".format(
                metric.name, self.epoch_stats[prefix + metric.name].avg))

        # batch times
        self.batch_times.sort(key=lambda x: x[0])

        tracker_stats = zip(
            self.batch_times[:-1],
            self.batch_times[1:])

        str_builder.append(
            ' | '.join(
                '{} {:.1e}'.format(name, t2 - t1)
                for (_, t1), (name, t2) in tracker_stats))

        return ' | '.join(str_builder)


