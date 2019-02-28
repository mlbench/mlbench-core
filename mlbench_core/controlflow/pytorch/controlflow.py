r"""Control flow for pytorch applications."""
import logging
import time

from mlbench_core.utils import AverageMeter, Tracker
from mlbench_core.utils.pytorch.distributed import aggregate_gradients, global_average
from mlbench_core.utils.pytorch.helpers import Timeit, update_best_runtime_metric, \
    iterate_dataloader
from mlbench_core.utils import LogMetrics

import torch
import torch.distributed as dist

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
        tracker (:obj:`mlbench_core.utils.Tracker`): Tracker for the controlflow. Default: `None`
    """

    def __init__(self, model, optimizer, loss_function, metrics, scheduler,
                 batch_size, train_epochs, rank, world_size, run_id, dtype,
                 validate=True, schedule_per='epoch', checkpoint=None,
                 transform_target_type=None, average_models=False,
                 use_cuda=False, max_batch_per_epoch=None, tracker=None):
        self.tracker = tracker or Tracker()
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
            start_epoch = self.tracker.current_epoch + 1
        else:
            start_epoch = 0

        dist.barrier()
        for epoch in range(start_epoch, self.train_epochs):
            # schedule learning rates
            if self.schedule_per == 'epoch':
                self.scheduler.step()

            # Per epoch information.
            logger.info("Current epoch : {} : lr={}"
                        .format(epoch, self.scheduler.get_lr()))

            # FIXME: The Timeit object can be a problem.
            self.train_epoch(dataloader_train)

            is_best = False
            if self.perform_validation:
                is_best = self.do_validate(dataloader_val)

            if self.checkpoint:
                self.checkpoint.save(self.tracker, self.model,
                                     self.optimizer, self.scheduler,
                                     self.tracker.current_epoch, is_best)

            # Shuffle the dataset across nodes
            if repartition_per_epoch:
                if dataloader_train_fn:
                    dataloader_train = dataloader_train_fn()

                if dataloader_val_fn:
                    dataloader_val = dataloader_val_fn()

                self._get_dataloader_stats(dataloader_train, dataloader_val)

            self.tracker.epoch_end()

    def train_epoch(self, dataloader):
        """Train model for one epoch of data.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The train set
        """
        # switch to train mode
        self.model.train()
        self.tracker.train()
        data_iter = iterate_dataloader(
            dataloader,
            self.dtype,
            self.max_batch_per_epoch,
            self.use_cuda,
            self.transform_target_type)

        for batch_idx, (data, target) in enumerate(data_iter):
            self.tracker.batch_start()

            if self.schedule_per == 'batch':
                self.scheduler.step()

            # Clear gradients in the optimizer.
            self.optimizer.zero_grad()
            self.tracker.record_batch_step('init')

            # Compute the output
            output = self.model(data)
            self.tracker.record_batch_step('fwd_pass')

            # Compute the loss
            loss = self.loss_function(output, target)
            self.tracker.record_batch_step('comp_loss')

            # Backprop
            loss.backward()
            self.tracker.record_batch_step('backprop')

            # Aggregate gradients/parameters from all workers and apply updates to model
            self.optimizer.step()
            self.tracker.record_batch_step('opt_step')

            self.tracker.batch_end()

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

        self.tracker.record_loss(loss, output.size()[0], log_to_api=True)

        # Compute metrics for one batch
        for metric in self.metrics:
            metric_value = metric(output, target).item()

            self.tracker.record_metric(
                metric,
                metric_value,
                output.size()[0],
                log_to_api=True)

        logger.info(str(self.tracker))

    def do_validate(self, dataloader):
        """Evaluate the model on the test dataset and save to the checkpoint.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The validation set
        """
        # evaluate the model.
        self.tracker.validation()

        self.tracker.validation_start()
        metrics_values, loss = self.validate(dataloader)
        self.tracker.validation_end()

        if len(metrics_values) > 0:
            # Save
            for metric, value in metrics_values.items():
                self.tracker.record_metric(metric, value, log_to_api=True)

            if self.rank == 0:
                logger.info('{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
                    self.tracker.primary_metric,
                    self.rank,
                    self.tracker.best_epoch,
                    self.tracker.current_epoch,
                    self.tracker.best_metric_value))
        else:
            if self.rank == 0:
                logger.info("Validation loss={:.3f}".format(loss))

        self.tracker.record_loss(loss, log_to_api=True)

        return self.tracker.is_best()

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
        metrics_averages = {metric: metric.average().item()
                            for metric in self.metrics}
        loss_average = global_average(losses.sum, losses.count).item()
        return metrics_averages, loss_average
