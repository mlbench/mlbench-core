r"""Control flow for pytorch applications."""
import logging

import torch

from mlbench_core.controlflow.pytorch.helpers import iterate_dataloader
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average

logger = logging.getLogger("mlbench")

LOG_EVERY_N_BATCHES = 25


def compute_train_batch_metrics(output, target, metrics):
    """Computes the given metrics on the given batch

    Args:
        output (:obj:`torch.Tensor`): The model output
        target (:obj:`torch.Tensor`): The labels for the current batch
        metrics (list): List of metrics to track

    Returns:
        (dict of :obj:`mlbench_core.evaluation.pytorch.metrics.MLBenchMetric`: float): The metric
            and its computed value
    """
    # Compute metrics for one batch
    result = {}
    for metric in metrics:
        metric_value = metric(output, target).item()
        result[metric] = metric_value
    return result


def record_train_batch_stats(
    batch_idx, loss, output, metric_results, tracker, num_batches_per_device_train
):
    """Record the stats in a training batch.

    Args:
        batch_idx (int): The id of the current batch
        loss (float): The loss of the batch
        output (:obj:`torch.Tensor`): The model output
        metric_results (dict): of :obj:`mlbench_core.evaluation.pytorch.metrics.MLBenchMetric`: float Metrics and their values
        tracker (:obj:`mlbench_core.utils.Tracker`): Tracker object to use.
        num_batches_per_device_train (int): Number of batches per train epoch
    """
    progress = batch_idx / num_batches_per_device_train
    progress += tracker.current_epoch

    log_to_api = (
        batch_idx % LOG_EVERY_N_BATCHES == 0
        or batch_idx == num_batches_per_device_train
    )

    if tracker:
        tracker.record_loss(loss, output.size()[0], log_to_api=log_to_api)

        for metric, metric_value in metric_results.items():
            tracker.record_metric(
                metric, metric_value, output.size()[0], log_to_api=log_to_api
            )
    status = "Epoch {:5.2f} Batch {:4}: ".format(progress, batch_idx)

    logger.info(status + str(tracker))


def validation_round(
    dataloader,
    model,
    loss_function,
    metrics,
    dtype,
    tracker=None,
    transform_target_type=False,
    use_cuda=False,
    max_batches=None,
):
    """Evaluate the model on the test dataset.

    Args:
        dataloader (`obj`:torch.utils.data.DataLoader): The validation set
        model (`obj`:torch.nn.Module): The model to train
        loss_function (`obj`:torch.nn.Module): The loss function
        metrics (list): List of metrics to track
        dtype (str): The datatype to use, one of `fp32`or `fp64`
        tracker (`obj`:mlbench_core.utils.Tracker | None): Tracker object to use.
        transform_target_type (bool): Convert target to `dtype`. Default `False`
        use_cuda (bool): Whether to use GPU for training, default: `False`
        max_batches (int | None): Maximum number of batches to validate on

    Returns:
          (dict, float): Dictionary of average of each metric, and average validation loss
    """

    model.eval()
    if tracker:
        tracker.validation()
        tracker.validation_start()

    # Initialize the accumulators for loss and metrics
    losses = AverageMeter()
    for metric in metrics:
        metric.reset()

    # Each worker computer their own losses and metrics
    with torch.no_grad():

        data_iter = iterate_dataloader(
            dataloader, dtype, max_batches, use_cuda, transform_target_type
        )

        for data, target in data_iter:
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
    metrics_averages = {metric: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()

    if tracker:
        tracker.validation_end()
    return metrics_averages, loss_average


def record_validation_stats(metrics_values, loss, tracker=None, rank=0):
    """Records the stats of a previously run validation

    Args:
        metrics_values (dict): Dictionary of each metric's average.
        loss (float): Validation loss
        tracker (`obj`:mlbench_core.utils.Tracker, optional): Tracker object to use.
        rank (int): Current distributed rank

    Returns:
        (bool): Whether this validation round is the best
    """
    if len(metrics_values) > 0:
        # Save
        if tracker:
            for metric, value in metrics_values.items():
                tracker.record_metric(metric, value, log_to_api=rank == 0)

                tracker.record_stat(
                    "global_{}".format(metric.name),
                    value,
                    log_to_api=rank == 0,
                )

        if rank == 0 and tracker:
            logger.info(
                "{} for rank {}:(best epoch {}, current epoch {}): {:.3f}".format(
                    tracker.primary_metric.name,
                    tracker.rank,
                    tracker.best_epoch,
                    tracker.current_epoch,
                    tracker.best_metric_value,
                )
            )
    else:
        if rank == 0:
            logger.info("Validation loss={:.3f}".format(loss))

    if tracker:
        tracker.record_loss(loss, log_to_api=True)

        global_loss = global_average(loss, 1).item()

        if rank == 0:
            tracker.record_stat("global_loss", global_loss, log_to_api=True)

    return tracker.is_best() if tracker else False
