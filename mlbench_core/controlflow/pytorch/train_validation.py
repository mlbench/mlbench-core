"""Deprecated class, please use functions in `controlflow.py`"""

import logging

import deprecation
from torch import distributed as dist

from mlbench_core.utils import Tracker

from . import (
    compute_train_batch_metrics,
    record_train_batch_stats,
    record_validation_stats,
    validation_round,
)
from .helpers import iterate_dataloader

logger = logging.getLogger("mlbench")


@deprecation.deprecated()
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

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        metrics,
        scheduler,
        batch_size,
        train_epochs,
        rank,
        world_size,
        run_id,
        dtype,
        validate=True,
        schedule_per="epoch",
        checkpoint=None,
        transform_target_type=None,
        average_models=False,
        use_cuda=False,
        max_batch_per_epoch=None,
        tracker=None,
    ):
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_per = schedule_per
        self.perform_validation = validate
        self.checkpoint = checkpoint
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.rank = rank
        self.run_id = run_id
        self.dtype = dtype
        self.schedule_per = schedule_per
        self.transform_target_type = transform_target_type
        self.use_cuda = use_cuda
        self.max_batch_per_epoch = max_batch_per_epoch
        if tracker:
            self.tracker = tracker
        else:
            self.tracker = Tracker(metrics, run_id, rank)

    def _get_dataloader_stats(self, dataloader_train, dataloader_val):
        """Sets the stats for the supplied dataloaders

        Args:
            dataloader_train (:obj:`torch.utils.data.DataLoader`): The train set
            dataloader_val (:obj:`torch.utils.data.DataLoader`): The validation set
        """
        self.num_batches_per_device_train = len(dataloader_train)
        self.num_batches_per_device_val = len(dataloader_val)

    def run(
        self,
        dataloader_train=None,
        dataloader_val=None,
        dataloader_train_fn=None,
        dataloader_val_fn=None,
        resume=False,
        repartition_per_epoch=False,
    ):
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
                "One of dataloader_train_fn or dataloader_train must be set"
            )

        if not dataloader_val_fn and not dataloader_val:
            raise ValueError("One of dataloader_val_fn or dataloader_val must be set")

        if dataloader_train_fn:
            dataloader_train = dataloader_train_fn()

        if dataloader_val_fn:
            dataloader_val = dataloader_val_fn()

        self._get_dataloader_stats(dataloader_train, dataloader_val)

        # define some parameters for training.
        logger.info(
            "There are {train_epochs} epochs, {num_batches} "
            "mini-batches per epoch (batch size: {batch_size}).".format(
                train_epochs=self.train_epochs,
                num_batches=self.num_batches_per_device_train,
                batch_size=self.batch_size,
            )
        )

        # Initialize Tracker or resume from checkpoint
        if resume:
            start_epoch = self.tracker.current_epoch + 1
        else:
            start_epoch = 0

        dist.barrier()
        for epoch in range(start_epoch, self.train_epochs):
            # Per epoch information.
            logger.info(
                "Current epoch : {} : lr={}".format(epoch, self.scheduler.get_lr())
            )

            train_round(
                dataloader_train,
                self.model,
                self.optimizer,
                self.loss_function,
                self.metrics,
                self.scheduler,
                self.dtype,
                self.schedule_per,
                self.transform_target_type,
                self.use_cuda,
                self.max_batch_per_epoch,
                self.tracker,
            )

            is_best = False
            if self.perform_validation:
                metrics, loss = validation_round(
                    dataloader_val,
                    self.model,
                    self.loss_function,
                    self.metrics,
                    self.dtype,
                    self.tracker,
                    self.transform_target_type,
                    self.use_cuda,
                )
                is_best = record_validation_stats(
                    metrics, loss, self.tracker, self.rank
                )

            if self.checkpoint:
                self.checkpoint.save(
                    self.tracker,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.tracker.current_epoch,
                    is_best,
                )

            # Shuffle the dataset across nodes
            if repartition_per_epoch:
                if dataloader_train_fn:
                    dataloader_train = dataloader_train_fn()

                if dataloader_val_fn:
                    dataloader_val = dataloader_val_fn()

                self._get_dataloader_stats(dataloader_train, dataloader_val)

            self.tracker.epoch_end()


def train_round(
    dataloader,
    model,
    optimizer,
    loss_function,
    metrics,
    scheduler,
    dtype,
    schedule_per="epoch",
    transform_target_type=None,
    use_cuda=False,
    max_batch_per_epoch=None,
    tracker=None,
):
    """Performs max_batch_per_epoch batches of training (or full trainset if
    not specified)

    Args:
        dataloader (:obj:`torch.utils.data.DataLoader`): The train set
        model (`obj`:torch.nn.Module): The model to train
        optimizer (`obj`:torch.optim): The optimizer
        loss_function (`obj`:torch.nn.Module): The loss function
        metrics (list): List of metrics to track
        scheduler (`obj`:torch.optim.lr_scheduler): Learning Rate scheduler
        dtype (str): The datatype to use, one of `fp32`or `fp64`
        schedule_per (str): Learning Rate scheduler mode, one of `batch` or `epoch`
        transform_target_type (str): Datatype to convert data to, default: `None`
        use_cuda (bool): Whether to use GPU for training, default: `False`
        max_batch_per_epoch (int): Maximum number of batches tot rain for per epoch,
                                   default: `None` (all batches)
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use.
    """
    model.train()

    if tracker:
        tracker.train()

    data_iter = iterate_dataloader(
        dataloader, dtype, max_batch_per_epoch, use_cuda, transform_target_type
    )

    num_batches_per_device_train = len(dataloader)

    for batch_idx, (data, target) in enumerate(data_iter):
        if tracker:
            tracker.batch_start()

        # Clear gradients in the optimizer.
        optimizer.zero_grad()
        if tracker:
            tracker.record_batch_init()

        # Compute the output
        output = model(data)
        if tracker:
            tracker.record_batch_fwd_pass()

        # Compute the loss
        loss = loss_function(output, target)
        if tracker:
            tracker.record_batch_comp_loss()

        # Backprop
        loss.backward()
        if tracker:
            tracker.record_batch_backprop()

        # Aggregate gradients/parameters from all workers and apply updates to model
        optimizer.step()
        if tracker:
            tracker.record_batch_opt_step()

        if schedule_per == "batch":
            scheduler.step()

        metrics_results = compute_train_batch_metrics(
            output,
            target,
            metrics,
        )

        if tracker:
            tracker.record_batch_comp_metrics()
            tracker.batch_end()

        record_train_batch_stats(
            batch_idx,
            loss.item(),
            output,
            metrics_results,
            tracker,
            num_batches_per_device_train,
        )

    if schedule_per == "epoch":
        scheduler.step()
