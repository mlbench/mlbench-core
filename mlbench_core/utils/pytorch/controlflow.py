import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch import checkpoint
from mlbench_core.utils.pytorch.metrics import AverageMeter
from mlbench_core.utils.pytorch.distributed import aggregate_gradients, global_average
from mlbench_core.utils.pytorch.utils import convert_dtype, Timeit, maybe_range, update_best_runtime_metric
from mlbench_core.utils.pytorch.utils import log_metrics


def train_epoch(model, optimizer, criterion, scheduler, config, timeit, dataloader):
    """Train model for one epoch of data."""
    # switch to train mode
    model.train()

    for batch_idx, (data, target) in zip(maybe_range(config.max_batch_per_epoch),
                                         dataloader):
        if config.lr_scheduler_level == 'batch':
            scheduler.step()

        data = convert_dtype(config.dtype, data)
        if config.transform_target_type:
            target = convert_dtype(config.dtype, target)

        if config.use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        aggregate_gradients(model, config.world_size)
        optimizer.step()

        with torch.no_grad():
            loss = loss.item()
            loss = global_average(loss, 1).item()
            if config.rank == 0:
                print("Train Batch {:5}: loss={:.3f}".format(batch_idx, loss))
            log_metrics(config, 'train_loss', loss)

            timeit.pause()
            config.runtime['cumu_time_train'].append(timeit.cumu)
            timeit.resume()


def validate(model, optimizer, criterion, metrics, config, dataloader):
    model.eval()

    losses = AverageMeter()
    for metric in metrics:
        metric.reset()

    for batch_idx, (data, target) in zip(maybe_range(config.max_batch_per_epoch),
                                         dataloader):
        data = convert_dtype(config.dtype, data)
        if config.transform_target_type:
            target = convert_dtype(config.dtype, target)

        if config.use_cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)

            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            for metric in metrics:
                metric_value = metric(output, target)
                metric.update(metric_value, data.size(0))

    metrics_averages = {metric.name: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()
    return metrics_averages, loss_average


def do_validate(model, optimizer, criterion, metrics, scheduler, config, timeit, dataloader):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # evaluate the model.
    metrics_values, loss = validate(model, optimizer, criterion, metrics, config, dataloader)

    config.runtime['cumu_time_val'].append(timeit.cumu)

    if len(metrics_values) > 0:
        # Assume the first metric is used to determine the best model to checkpoint.
        prim_metric = metrics[0]
        prim_metric_value = metrics_values[prim_metric.name]

        is_best, best_metric_name = update_best_runtime_metric(config, prim_metric_value, prim_metric.name)

        if config.rank == 0:
            print('{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
                best_metric_name,
                config.rank,
                config.runtime['best_epoch'],
                config.runtime['current_epoch'],
                config.runtime[best_metric_name]))

        for name, value in metrics_values.items():
            log_metrics(config, name, value)
    else:
        is_best = False
        if config.rank == 0:
            print("Validation loss={:.3f}".format(loss))

    log_metrics(config, 'val_loss', loss)

    checkpoint.save(config, model, optimizer, scheduler, is_best)


class TrainValidation(object):
    def __call__(self, model, optimizer, criterion, metrics, scheduler, config, dataloader_fn):
        """Train models and perform validation.

        :param model: a pytorch model to be trained and validated.
        :type model: nn.Module
        :param optimizer: an optimizer for the given model.
        :param criterion: loss function. 
        :param metrics: metrics like TopKAccuracy.
        :param scheduler: a scheduler for hyperparameters.
        :param config: a global object containing all of the config.
        :type config: argparse.Namespace
        """
        dataloader_train, dataloader_val = dataloader_fn(config)

        # define some parameters for training.
        print('There are {} epochs, {} mini-batches per epoch (batch size:{}).'
              .format(config.train_epochs, config.num_batches_per_device_train,
                      config.batch_size))

        # train the model and evaluate the model per args.eval_freq
        max_epochs = min(config.train_epochs, config.max_train_steps)\
            if config.max_train_steps else config.train_epochs
        start_epoch = config.runtime['current_epoch'] if config.resume else 0
        config.runtime['records'] = config.runtime.get('records', [])
        config.runtime['cumu_time_val'] = config.runtime.get('cumu_time_val', [])
        config.runtime['cumu_time_train'] = config.runtime.get('cumu_time_train', [])

        dist.barrier()

        timeit = Timeit(0 if len(config.runtime['cumu_time_val']) == 0
                        else config.runtime['cumu_time_val'][-1])
        for epoch in range(start_epoch, max_epochs):
            config.runtime['current_epoch'] = epoch

            # schedule learning rates
            if config.lr_scheduler_level == 'epoch':
                scheduler.step()

            # Per epoch information.
            print("Current epoch : {} : lr={} : time={:10.3e}"
                  .format(epoch, scheduler.get_lr(), timeit.cumu))

            train_epoch(model, optimizer, criterion, scheduler, config, timeit, dataloader_train)

            if config.validation:
                timeit.pause()
                do_validate(model, optimizer, criterion, metrics, scheduler, config, timeit, dataloader_val)
                timeit.resume()

            if config.repartition_per_epoch:
                dataloader_train, dataloader_val = dataloader_fn(config)


class ControlFlow(object):
    @staticmethod
    def create(config):
        # TODO:
        return TrainValidation()
