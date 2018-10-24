r"""Control flow for pytorch applications."""
import torch
import logging
import time
import torch.distributed as dist
from collections import defaultdict

from mlbench_core.utils.pytorch import checkpoint
from mlbench_core.evaluation.pytorch.metrics import AverageMeter
from mlbench_core.utils.pytorch.distributed import aggregate_gradients, global_average
from mlbench_core.utils.pytorch.helpers import Timeit, update_best_runtime_metric, \
    iterate_dataloader, log_metrics, Tracker

logger = logging.getLogger('mlbench')


def record_train_batch_stats(batch_idx, loss, output, target, metrics, config, timeit, tracker):
    r"""Record the stats in a training batch."""
    progress = batch_idx / config.num_batches_per_device_train
    progress += tracker.current_epoch

    tracker.epoch_stats["loss"].update(loss, output.size())

    str_builder = ["Epoch {:5.2f} Batch {:4}: loss={:6.2e}".format(
        progress, batch_idx, tracker.epoch_stats["loss"].avg)]

    # Compute metrics for one batch
    for metric in metrics:
        metric_value = metric(output, target).item()
        tracker.epoch_stats[metric.name].update(metric_value, output.size())
        str_builder.append("{} {:.2e}".format(
            metric.name, tracker.epoch_stats[metric.name].avg))

    # Compute time spent on each step
    for (_, t1), (name, t2) in zip(tracker.batch_stats[:-1], tracker.batch_stats[1:]):
        str_builder.append("{} {:.1e}".format(name, t2 - t1))

    logger.info(" | ".join(str_builder))

    if not hasattr(tracker, 'cumu_time_train'):
        tracker.cumu_time_train = []
    tracker.cumu_time_train.append(
        tracker.batch_stats[-1][1] - tracker.batch_stats[0][1])

    log_metrics(config, tracker, 'train_loss', loss)


def train_epoch(model, optimizer, loss_function, scheduler, config, metrics, timeit, dataloader, tracker):
    """Train model for one epoch of data."""
    tracker.epoch_stats = {k: AverageMeter()
                           for k in ["loss"] + [m.name for m in metrics]}
    # switch to train mode
    model.train()
    for batch_idx, (data, target) in enumerate(iterate_dataloader(dataloader, config)):
        tracker.batch_stats = [("start", time.time())]

        if config.lr_scheduler_level == 'batch':
            scheduler.step()

        # Clear gradients in the optimizer.
        optimizer.zero_grad()
        tracker.batch_stats.append(('init', time.time()))

        # Compute the output
        output = model(data)
        tracker.batch_stats.append(('fwd_pass', time.time()))

        # Compute the loss
        loss = loss_function(output, target)
        tracker.batch_stats.append(('comp_loss', time.time()))

        # Backprop
        loss.backward()
        tracker.batch_stats.append(('backprop', time.time()))

        # Aggregate gradients from all workers
        aggregate_gradients(model, config)
        tracker.batch_stats.append(('aggr_grad', time.time()))

        # Apply updates to model
        optimizer.step()
        tracker.batch_stats.append(('opt_step', time.time()))

        record_train_batch_stats(batch_idx, loss.item(
        ), output, target, metrics, config, timeit, tracker)


def validate(model, loss_function, metrics, config, dataloader):
    r"""Validate the quality of the model in terms of loss and metrics.

    :param model: PyTorch Models.
    :type model: nn.Module
    :param loss_function: A loss function
    :type loss_function: nn.modules.loss
    :param metrics: metrics to measure
    :type metrics: list
    :param config: configurations of the training.
    :type config: argparse.Namespace
    :param dataloader: load data in batches.
    :type dataloader: torch.utils.data.dataloader.DataLoader
    :returns: global metrics and loss related to current model
    :rtype: dict, float
    """
    # Turn on evaluation mode for the model
    model.eval()

    # Initialize the accumulators for loss and metrics
    losses = AverageMeter()
    for metric in metrics:
        metric.reset()

    # Each worker computer their own losses and metrics
    with torch.no_grad():
        for data, target in iterate_dataloader(dataloader, config):
            # Inference
            output = model(data)

            # Compute loss
            loss = loss_function(output, target)

            # Update loss
            losses.update(loss.item(), data.size(0))

            # Update metrics
            for metric in metrics:
                metric_value = metric(output, target)
                metric.update(metric_value, data.size(0))

    # Aggregate metrics and loss for all workers
    metrics_averages = {metric.name: metric.average().item()
                        for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()
    return metrics_averages, loss_average


def do_validate(model, optimizer, loss_function, metrics, scheduler, config, timeit, dataloader, tracker):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    metrics_values, loss = validate(
        model, loss_function, metrics, config, dataloader)

    if not hasattr(tracker, 'cumu_time_val'):
        tracker.cumu_time_val = []
    tracker.cumu_time_val.append(time.time() - tracker.start_time)

    if len(metrics_values) > 0:
        # Assume the first metric is used to determine the best model to checkpoint.
        prim_metric = metrics[0]
        prim_metric_value = metrics_values[prim_metric.name]

        is_best, best_metric_name = update_best_runtime_metric(
            config, tracker, prim_metric_value, prim_metric.name)

        # Save
        for name, value in metrics_values.items():
            log_metrics(config, tracker, name, value)

        if config.rank == 0:
            logger.info('{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
                best_metric_name,
                config.rank,
                tracker.records['best_epoch'],
                tracker.current_epoch,
                tracker.records[best_metric_name]))
    else:
        is_best = False
        if config.rank == 0:
            logger.info("Validation loss={:.3f}".format(loss))

    log_metrics(config, tracker, 'val_loss', loss)

    checkpoint.save(config, tracker, model, optimizer, scheduler, is_best)


class TrainValidation(object):
    r"""Train and validate a model."""

    def __call__(self, model, optimizer, loss_function, metrics, scheduler, config, dataloader_fn):
        """Train models and perform validation.

        Args:
            model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
            optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
            loss_function (:obj:`torch.nn.modules.loss._Loss`): loss function.
            metrics (:obj:`list` of :obj:`mlbench_core.evaluation.pytorch.*`): metrics like TopKAccuracy.
            scheduler (:obj:`mlbench_core.lr_scheduler.pytorch.lr.*`): a scheduler for hyperparameters.
            config (:obj:`types.SimpleNamespace`): a global object containing all of the config.
            dataloader_fn (:func:`Function`): A function returning a :obj:`torch.utils.data.DataLoader`.
        """
        # TODO: resume a tracker
        tracker = Tracker(config)

        dataloader_train = dataloader_fn(train=True, config=config)
        dataloader_val = dataloader_fn(train=False, config=config)

        # define some parameters for training.
        logger.info("There are {train_epochs} epochs, {num_batches_per_device_train} "
                    "mini-batches per epoch (batch size: {batch_size})."
                    .format(**config.__dict__))

        # train the model and evaluate the model per args.eval_freq
        max_epochs = min(config.train_epochs, config.max_train_steps)\
            if config.max_train_steps else config.train_epochs

        # Initialize Tracker or resume from checkpoint
        if config.resume:
            # TODO: Update the resume part in checkpoint.py
            start_epoch = tracker.current_epoch + 1 if config.resume else 0
            timeit = Timeit(config.runtime['cumu_time_val'][-1])
            raise NotImplementedError
        else:
            start_epoch = 0
            timeit = Timeit(0.)

            # Initialize Tracker
            tracker.current_epoch = 0
            tracker.best_epoch = 0
            tracker.records = defaultdict(list)
            tracker.start_time = time.time()

        dist.barrier()
        for epoch in range(start_epoch, max_epochs):
            tracker.current_epoch = epoch

            # schedule learning rates
            if config.lr_scheduler_level == 'epoch':
                scheduler.step()

            # Per epoch information.
            logger.info("Current epoch : {} : lr={} : time={:10.3e}"
                        .format(epoch, scheduler.get_lr(), timeit.cumu))

            # FIXME: The Timeit object can be a problem.
            train_epoch(model, optimizer, loss_function, scheduler,
                        config, metrics, timeit, dataloader_train, tracker)

            if config.validation:
                do_validate(model, optimizer, loss_function, metrics,
                            scheduler, config, timeit, dataloader_val, tracker)

            # Shuffle the dataset across nodes
            if config.repartition_per_epoch:
                dataloader_train = dataloader_fn(train=True, config=config)
