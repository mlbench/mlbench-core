r"""A controlflow which train and evaluate a model."""
import logging

from mlbench_core.utils import AverageMeter, Tracker


def train_round(
    session,
    train_set_init_op,
    train_op,
    loss_op,
    metrics,
    batch_size,
    num_batches_per_epoch_for_train,
    tracker,
    lr_scheduler_level=None,
    lr_tensor=None,
):
    """ Performs num_batches_per_epoch_for_train batches of training (or full trainset if
    not specified)

    Args:
        session (obj): The tensorflow session
        train_set_init_op (obj): The trainset initialisation tf operation
        train_op (obj): The tensorflow training operation
        loss_op (obj): The tensorflow loss operation
        metrics (list): List of metrics to track
        batch_size (int): The batch size
        num_batches_per_epoch_for_train (int): Maximum number of batches tot rain for per epoch,
                                   default: `None` (all batches)
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use
        lr_scheduler_level (str): Learning Rate scheduler mode, one of `batch` or `epoch`
        lr_tensor (obj): The learningrate schedule tensorflow operation
    """
    logging.info("Initialize training dataset.")
    session.run(train_set_init_op)
    tracker.train()

    loss_meter = AverageMeter()
    metrics_meter = [AverageMeter() for _ in metrics]

    if lr_scheduler_level == "epoch" and lr_tensor is not None:
        lr = session.run(lr_tensor)
        logging.debug(
            "Epoch {} Learning Rate : {:10.3e}".format(tracker.current_epoch, lr)
        )

    for i_batch in range(num_batches_per_epoch_for_train):
        # for i_batch in range(1):
        tracker.batch_start()

        if lr_scheduler_level == "batch" and lr_tensor is not None:
            lr = session.run(lr_tensor)
            logging.debug(
                "Epoch {} Learning Rate : {:10.3e}".format(tracker.current_epoch, lr)
            )

        out = session.run(
            {
                "metrics": [m.metric_op for m in metrics],
                "loss": loss_op,
                "train_op": train_op,
            }
        )

        tracker.batch_end()

        # Update tracker
        loss_meter.update(out["loss"], n=batch_size)
        tracker.record_loss(loss_meter.avg, log_to_api=True)

        for metric, meter, o in zip(metrics, metrics_meter, out["metrics"]):
            meter.update(o, n=batch_size)
            tracker.record_metric(metric, meter.avg, log_to_api=True)

        # Print logging information.
        progress = i_batch / num_batches_per_epoch_for_train
        progress += tracker.current_epoch

        status = "Epoch {:5.2f} Batch {:4}: ".format(progress, i_batch)

        logging.info(status + str(tracker))

    # Record training loss and metrics.
    tracker.record_loss(loss_meter.avg, log_to_api=True)

    for metric, meter in zip(metrics, metrics_meter):
        tracker.record_metric(metric, meter.avg, log_to_api=True)

    logging.info("Finish training for one epoch.")


def validation_round(
    session,
    validation_set_init_op,
    loss_op,
    metrics,
    batch_size,
    num_batches_per_epoch_for_validation,
    tracker,
):
    """ Handles one full iteration of validation on the whole validation set.

    Args:
        session (obj): The tensorflow session
        validation_set_init_op (obj): The trainset initialisation tf operation
        loss_op (obj): The tensorflow loss operation
        metrics (list): List of metrics to track
        batch_size (int): The batch size
        num_batches_per_epoch_for_validation (int): Maximum number of batches to validate
            for per epoch, default: `None` (all batches)
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use
    """
    session.run(validation_set_init_op)
    tracker.validation()

    loss_meter = AverageMeter()
    metrics_meter = [AverageMeter() for _ in metrics]

    for i_batch in range(num_batches_per_epoch_for_validation):
        out = session.run({"metrics": [m.metric_op for m in metrics], "loss": loss_op})

        # Update tracker
        loss_meter.update(out["loss"], n=batch_size)
        for meter, o in zip(metrics_meter, out["metrics"]):
            meter.update(o, n=batch_size)

        logging.debug(
            "{}/{} Validation loss={:10.3e} | metrics: [{}]".format(
                tracker.current_epoch,
                i_batch,
                loss_meter.avg,
                ",".join([format(m.avg, "10.3e") for m in metrics_meter]),
            )
        )

    tracker.record_loss(loss_meter.avg, log_to_api=True)

    if tracker.rank == 0:
        tracker.record_stat("global_loss", loss_meter.avg, log_to_api=True)

    for i, metric, meter in zip(range(len(metrics)), metrics, metrics_meter):
        tracker.record_metric(metric, meter.avg, log_to_api=True)

        if tracker.rank == 0:
            tracker.record_stat(
                "global_{}".format(metric.name), meter.avg, log_to_api=True
            )


class TrainValidation(object):
    """A control flow to train and evaluate a model.

    Args:
        train_op (:obj:`tf.Operation`): An operation for training models.
        sess (:obj:`tf.Session`): A session which the control flow will communicate.
        loss (:obj:`tf.Tensor`): The loss tensor.
        metrics (list of :obj:`tf.Tensor`): A list of metrics tensors.
        max_train_steps (int): Number of steps for training (independent of lr)
        train_epochs (int): Number of steps for training (may related to lr).
        batch_size (int): Size of a batch.
        num_batches_per_epoch_for_train (int): Number of batches in one training epoch
        num_batches_per_epoch_for_validation (int): Number of batches in one validation epoch
        train_set_init_op (:obj:`tf.Operation`): Op for initializing training dataset.
        validation_set_init_op (:obj:`tf.Operation`): Op for initializing validation dataset.
        run_id (str): the id of the run in the dashboard
        rank (int): the rank of the current worker
        lr_scheduler_level (str): Learning rate is updated based on `epoch` or `batch`.

    """

    def __init__(
        self,
        train_op,
        sess,
        loss,
        metrics,
        max_train_steps,
        train_epochs,
        batch_size,
        num_batches_per_epoch_for_train,
        num_batches_per_epoch_for_validation,
        train_set_init_op,
        validation_set_init_op,
        run_id,
        rank,
        lr_scheduler_level="epoch",
        tracker=None,
    ):
        self.batch_size = batch_size
        self.num_batches_per_epoch_for_train = num_batches_per_epoch_for_train
        self.num_batches_per_epoch_for_validation = num_batches_per_epoch_for_validation
        self.sess = sess
        self.loss = loss
        self.metrics = metrics
        self.train_op = train_op
        self.lr_scheduler_level = lr_scheduler_level
        self.max_train_steps = max_train_steps
        self.train_epochs = train_epochs
        self.train_set_init_op = train_set_init_op
        self.validation_set_init_op = validation_set_init_op
        self.run_id = run_id
        self.rank = rank
        if tracker:
            self.tracker = tracker
        else:
            self.tracker = Tracker(metrics, run_id, rank)

    def train_one_epoch(self, lr_tensor_name=None):
        """Train a model for an epoch and use tracker to log stats.

        Args:
            lr_tensor (obj): The learningrate schedule tensorflow operation"""
        train_round(
            self.sess,
            self.train_set_init_op,
            self.train_op,
            self.loss,
            self.metrics,
            self.batch_size,
            self.num_batches_per_epoch_for_train,
            self.tracker,
            lr_tensor=lr_tensor_name,
            lr_scheduler_level=self.lr_scheduler_level,
        )

    def valid_one_epoch(self):
        """Validate a model for an epoch and use tracker to log stats."""
        validation_round(
            self.sess,
            self.validation_set_init_op,
            self.loss,
            self.metrics,
            self.batch_size,
            self.num_batches_per_epoch_for_validation,
            self.tracker,
        )

    def train_and_eval(self, initial_epoch=0, lr_tensor_name=None):
        """Train and evaluate one epoch.

        Args:
            initial_epoch (int, optional): Defaults to 0. Initial epoch of training.
            lr_tensor_name (:obj:`tf.Tensor`, optional): Defaults to None.
                A (scalar) float tensor representing name of learning rate
        """
        final_epoch = min(self.max_train_steps, self.train_epochs)
        for i_epoch in range(initial_epoch, final_epoch):
            logging.debug("=> Epoch {}".format(i_epoch))

            self.train_one_epoch()
            self.valid_one_epoch()
            self.tracker.epoch_end()

        return self.tracker
