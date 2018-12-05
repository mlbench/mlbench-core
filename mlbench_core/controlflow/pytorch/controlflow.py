r"""Control flow for pytorch applications."""
import torch
import logging
import time
import math
import torch.distributed as dist
from collections import defaultdict

from mlbench_core.utils import AverageMeter, Tracker
from mlbench_core.utils.pytorch.distributed import aggregate_gradients, global_average
from mlbench_core.utils.pytorch.helpers import Timeit, update_best_runtime_metric, \
    iterate_dataloader, log_metrics

logger = logging.getLogger('mlbench')


class TrainValidation(object):
    r"""Train and validate a model.

    Args:
        model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        loss_function (:obj:`torch.nn.modules.loss._Loss`): loss function.
        metrics (:obj:`list` of :obj:`mlbench_core.evaluation.pytorch.*`): metrics like TopKAccuracy.
        scheduler (:obj:`mlbench_core.lr_scheduler.pytorch.lr.*`): a scheduler for hyperparameters.
        batch_size (int): The size of batches provided by the dataloader
        train_epochs (int): The number of epochs to train for
        rank (int): The rank of the current workers
        world_size (int): The total number of workers
        run_id (str): The id of the current run
        dtype (str): The datatype to use for the dataloader data
        validate (bool): Whether to run validation on the val dataset. Default: `True`
        schedule_per (str): When to perform a step for the lr scheduler, one of
            `epoch` or `batch`. Default: `epoch`
        checkpoint (:obj:`Checkpointer`): Class that handles checkpointing. Default: `None`
        transform_target_type (str): dtype to transform the target to. Not used. Default: `None`
        average_models (bool): Whether to average models together. Default: `False`
        use_cuda (bool): Whether to train on GPU or not. Default: `False`
        max_batch_per_epoch (int): Maximum number of batches per epoch. Whole dataset
            is used if not specified. Default: `None`
    """

    def __init__(self, model, optimizer, loss_function, metrics, scheduler,
                 batch_size, train_epochs, rank, world_size, run_id, dtype,
                 validate=True, schedule_per='epoch', checkpoint=None,
                 transform_target_type=None, average_models=False,
                 use_cuda=False, max_batch_per_epoch=None):
        self.tracker = Tracker()
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.scheduler = scheduler
        self.schedule_per = schedule_per
        self.perform_validation = validate
        self.rank = rank
        self.world_size = world_size
        self.checkpoint = checkpoint
        self.run_id = run_id
        self.transform_target_type = transform_target_type
        self.average_models = average_models
        self.use_cuda = use_cuda
        self.max_batch_per_epoch = max_batch_per_epoch
        self.dtype = dtype

    def _get_dataloader_stats(self, dataloader_train, dataloader_val):
        """ Sets the stats for the supplied dataloaders

        Args:
            dataloader_train (:obj:`torch.utils.data.DataLoader`): The train set
            dataloader_val (:obj:`torch.utils.data.DataLoader`): The validation set
        """
        self.num_batches_per_device_train = len(dataloader_train)
        self.num_batches_per_device_val = len(dataloader_val)

    def run(self, dataloader_train=None, dataloader_val=None,
            dataloader_train_fn=None, dataloader_val_fn=None, resume=False,
            repartition_per_epoch=False):
        """Execute training and (possibly) validation

        `dataloader_train` and `dataloader_train_fn` are mutually exclusive.
        `dataloader_val` and `dataloader_val_fn` are mutually exclusive.

        Args:
            dataloader_train (:obj:`torch.utils.data.DataLoader`): A dataloader for the train set.
                Default: `None`
            dataloader_val (:obj:`torch.utils.data.DataLoader`): A dataloader for the val set.
                Default: `None`
            dataloader_train_fn (:func:`Function`): A function returning a :obj:`torch.utils.data.DataLoader`
                for the train set. Default: `None`
            dataloader_val_fn (:func:`Function`): A function returning a :obj:`torch.utils.data.DataLoader`
                for the val set. Default: `None`
            resume (bool): Whether this is a resume of a previous run or not. Default: `False`
            repartition_per_epoch (bool): Whether to repartition the dataset again every epoch.
                Requires dataloader_train_fn and/or dataloader_val_fn to be set. Default: `False`
        """

        if not dataloader_train_fn and not dataloader_train:
            raise ValueError(
                "One of dataloader_train_fn or dataloader_train must be set")

        if not dataloader_val_fn and not dataloader_val:
            raise ValueError(
                "One of dataloader_val_fn or dataloader_val must be set")

        if dataloader_train_fn:
            dataloader_train = dataloader_train_fn()

        if dataloader_val_fn:
            dataloader_val = dataloader_val_fn()

        self._get_dataloader_stats(dataloader_train, dataloader_val)

        # define some parameters for training.
        logger.info("There are {train_epochs} epochs, {num_batches} "
                    "mini-batches per epoch (batch size: {batch_size})."
                    .format(
                        train_epochs=self.train_epochs,
                        num_batches=self.num_batches_per_device_train,
                        batch_size=self.batch_size))

        # Initialize Tracker or resume from checkpoint
        if resume:
            # TODO: Update the resume part in checkpoint.py
            start_epoch = self.tracker.current_epoch + 1 if resume else 0
            self.timeit = Timeit(self.checkpoint.runtime['cumu_time_val'][-1])
            raise NotImplementedError
        else:
            start_epoch = 1
            self.timeit = Timeit(0.)

            # Initialize Tracker
            self.tracker.current_epoch = 1
            self.tracker.best_epoch = 0
            self.tracker.records = defaultdict(list)
            self.tracker.start_time = time.time()

        dist.barrier()
        for epoch in range(start_epoch, 1 + self.train_epochs):
            self.tracker.current_epoch = epoch

            # schedule learning rates
            if self.schedule_per == 'epoch':
                self.scheduler.step()

            # Per epoch information.
            logger.info("Current epoch : {} : lr={} : time={:10.3e}"
                        .format(
                            epoch,
                            self.scheduler.get_lr(),
                            self.timeit.cumu))

            # FIXME: The Timeit object can be a problem.
            self.train_epoch(dataloader_train)

            if self.perform_validation:
                self.do_validate(dataloader_val)

            # Shuffle the dataset across nodes
            if repartition_per_epoch:
                if dataloader_train_fn:
                    dataloader_train = dataloader_train_fn()

                if dataloader_val_fn:
                    dataloader_val = dataloader_val_fn()

                self._get_dataloader_stats(dataloader_train, dataloader_val)

    def train_epoch(self, dataloader):
        """Train model for one epoch of data.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The train set
        """
        self.tracker.epoch_stats = {
            k: AverageMeter()
            for k in ["loss"] + [m.name for m in self.metrics]}
        # switch to train mode
        self.model.train()
        data_iter = iterate_dataloader(
            dataloader,
            self.dtype,
            self.max_batch_per_epoch,
            self.use_cuda,
            self.transform_target_type)

        for batch_idx, (data, target) in enumerate(data_iter):
            self.tracker.batch_stats = [("start", time.time())]

            if self.schedule_per == 'batch':
                self.scheduler.step()

            # Clear gradients in the optimizer.
            self.optimizer.zero_grad()
            self.tracker.batch_stats.append(('init', time.time()))

            # Compute the output
            output = self.model(data)
            self.tracker.batch_stats.append(('fwd_pass', time.time()))

            # Compute the loss
            loss = self.loss_function(output, target)
            self.tracker.batch_stats.append(('comp_loss', time.time()))

            # Backprop
            loss.backward()
            self.tracker.batch_stats.append(('backprop', time.time()))
            
            # Aggregate gradients/parameters from all workers and apply updates to model
            self.optimizer.step()
            self.tracker.batch_stats.append(('opt_step', time.time()))

            self.record_train_batch_stats(
                batch_idx,
                loss.item(),
                output,
                target)

    def record_train_batch_stats(self, batch_idx, loss, output, target):
        r"""Record the stats in a training batch.

        Args:
            batch_idx (int): The id of the current batch
            loss (float): The loss of the batch
            output (:obj:`torch.Tensor`): The model output
            target (:obj:`torch.Tensor`): The labels for the current batch
        """
        progress = batch_idx / self.num_batches_per_device_train
        progress += self.tracker.current_epoch

        self.tracker.epoch_stats["loss"].update(loss, output.size()[0])

        str_builder = ["Epoch {:5.2f} Batch {:4}: loss={:6.2e}".format(
            progress, batch_idx, self.tracker.epoch_stats["loss"].avg)]

        # Compute metrics for one batch
        for metric in self.metrics:
            metric_value = metric(output, target).item()

            self.tracker.epoch_stats[metric.name].update(
                metric_value,
                output.size()[0])

            str_builder.append("{} {:.2e}".format(
                metric.name, self.tracker.epoch_stats[metric.name].avg))

        # Compute time spent on each step
        tracker_stats = zip(
            self.tracker.batch_stats[:-1],
            self.tracker.batch_stats[1:])
        for (_, t1), (name, t2) in tracker_stats:
            str_builder.append("{} {:.1e}".format(name, t2 - t1))

        logger.info(" | ".join(str_builder))

        if not hasattr(self.tracker, 'cumu_time_train'):
            self.tracker.cumu_time_train = []

        self.tracker.cumu_time_train.append(
            self.tracker.batch_stats[-1][1] - self.tracker.batch_stats[0][1])

        log_metrics(
            self.run_id,
            self.rank,
            self.tracker.current_epoch,
            'train_loss',
            loss)

    def do_validate(self, dataloader):
        """Evaluate the model on the test dataset and save to the checkpoint.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The validation set
        """
        # evaluate the model.
        metrics_values, loss = self.validate(dataloader)

        if not hasattr(self.tracker, 'cumu_time_val'):
            self.tracker.cumu_time_val = []
        self.tracker.cumu_time_val.append(
            time.time() - self.tracker.start_time)

        if len(metrics_values) > 0:
            # Assume the first metric is used to determine the best model to checkpoint.
            prim_metric = self.metrics[0]
            prim_metric_value = metrics_values[prim_metric.name]

            is_best, best_metric_name = update_best_runtime_metric(
                self.tracker, prim_metric_value, prim_metric.name)

            # Save
            for name, value in metrics_values.items():
                log_metrics(
                    self.run_id,
                    self.rank,
                    self.tracker.current_epoch,
                    name,
                    value)

            if self.rank == 0:
                logger.info('{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
                    best_metric_name,
                    self.rank,
                    self.tracker.records['best_epoch'],
                    self.tracker.current_epoch,
                    self.tracker.records[best_metric_name]))
        else:
            is_best = False
            if self.rank == 0:
                logger.info("Validation loss={:.3f}".format(loss))

        log_metrics(
            self.run_id,
            self.rank,
            self.tracker.current_epoch,
            'val_loss',
            loss)

        if self.checkpoint:
            self.checkpoint.save(self.tracker, self.model,
                                 self.optimizer, self.scheduler,
                                 self.tracker.current_epoch, is_best)

    def validate(self, dataloader):
        r"""Validate the quality of the model in terms of loss and metrics.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The validation set
        """
        # Turn on evaluation mode for the model
        self.model.eval()

        # Initialize the accumulators for loss and metrics
        losses = AverageMeter()
        for metric in self.metrics:
            metric.reset()

        # Each worker computer their own losses and metrics
        with torch.no_grad():
            data_iter = iterate_dataloader(
                dataloader,
                self.dtype,
                self.max_batch_per_epoch,
                self.use_cuda,
                self.transform_target_type)

            for data, target in data_iter:
                # Inference
                output = self.model(data)

                # Compute loss
                loss = self.loss_function(output, target)

                # Update loss
                losses.update(loss.item(), data.size(0))

                # Update metrics
                for metric in self.metrics:
                    metric_value = metric(output, target)
                    metric.update(metric_value, data.size(0))

        # Aggregate metrics and loss for all workers
        metrics_averages = {metric.name: metric.average().item()
                            for metric in self.metrics}
        loss_average = global_average(losses.sum, losses.count).item()
        return metrics_averages, loss_average
