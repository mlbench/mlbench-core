r"""A controlflow which train and evaluate a model."""
import logging
import tensorflow as tf
from collections import defaultdict

from mlbench_core.utils import Tracker, AverageMeter


class TrainValidation(object):
    """A control flow to train and evaluate a model."""

    def __init__(self, train_op, sess, loss, metrics,
                 lr_scheduler_level, max_train_steps, train_epochs,
                 batch_size, num_batches_per_epoch_for_train, num_batches_per_epoch_for_validation,
                 train_set_init_op, validation_set_init_op):
        """
        Args:
            train_op (:obj:`tf.Operation`): An operation for training models.
            sess (:obj:`tf.Session`): A session which the control flow will communicate.
            is_training (bool or :obj:`tf.Tensor`): training the model with the number of atrs
            loss (:obj:`tf.Tensor`): The loss tensor.
            metrics (list of :obj:`tf.Tensor`): A list of metrics tensors.
        """
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

    def train_one_epoch(self, tracker):
        """Train a model for an epoch and use tracker to log stats."""
        logging.info("Initialize training dataset.")
        self.sess.run(self.train_set_init_op)

        loss_meter = AverageMeter()
        metrics_meter = [AverageMeter() for _ in self.metrics]

        for i_batch in range(self.num_batches_per_epoch_for_train):
            # for i_batch in range(1):
            out = self.sess.run({
                "metrics": [m['value'] for m in self.metrics],
                "loss": self.loss,
                "train_op": self.train_op,
            })

            # Update tracker
            loss_meter.update(out["loss"], n=self.batch_size)
            for meter, o in zip(metrics_meter, out['metrics']):
                meter.update(o, n=self.batch_size)

            # Print logging information.
            logging.debug(
                "E{}:B{}/{} loss={:10.3e} | metrics: [{}] | best epoch {} ({:10.3e})"
                .format(tracker.current_epoch,
                        i_batch,
                        self.num_batches_per_epoch_for_train,
                        loss_meter.avg,
                        ",".join([format(m.avg, "10.3e")
                                  for m in metrics_meter]),
                        tracker.best_epoch,
                        tracker.best_epoch_value))

        # Record training loss and metrics.
        tracker.records['train_loss'].append(loss_meter.avg)
        for metric, meter in zip(self.metrics, metrics_meter):
            tracker.records['train_' + metric['name']].append(meter.avg)

        logging.info("Finish training for one epoch.")

    def valid_one_epoch(self, tracker):
        self.sess.run(self.validation_set_init_op)

        loss_meter = AverageMeter()
        metrics_meter = [AverageMeter() for _ in self.metrics]

        for i_batch in range(self.num_batches_per_epoch_for_validation):
            out = self.sess.run({
                "metrics": [m['value'] for m in self.metrics],
                "loss": self.loss})

            # Update tracker
            loss_meter.update(out["loss"], n=self.batch_size)
            for meter, o in zip(metrics_meter, out['metrics']):
                meter.update(o, n=self.batch_size)

            logging.debug(
                "{}/{} Validation loss={:10.3e} | metrics: [{}]"
                .format(tracker.current_epoch, i_batch, loss_meter.avg,
                        ",".join([format(m.avg, "10.3e")
                                  for m in metrics_meter])))

        tracker.records['val_loss'].append(loss_meter.avg)
        for i, metric, meter in zip(range(len(self.metrics)), self.metrics, metrics_meter):
            metric_name = 'val_' + metric['name']

            # Here we implicitly assume the larger metrics value means better results
            if (i == 0) and (len(tracker.records[metric_name]) == 0 or
                             meter.avg > max(tracker.records[metric_name])):
                tracker.best_epoch = tracker.current_epoch
                tracker.best_epoch_value = meter.avg

            tracker.records[metric_name].append(meter.avg)

    def train_and_eval(self, initial_epoch=0, lr_tensor_name=None):
        """Train and evaluate one epoch.

        Args:
            initial_epoch (int, optional): Defaults to 0. Initial epoch of training.
            lr_tensor_name (:obj:`tf.Tensor`, optional): Defaults to None.
                A (scalar) float tensor representing name of learning rate
        """
        # Tracker for stats
        tracker = Tracker()
        tracker.current_epoch = 0
        tracker.best_epoch = 0
        tracker.best_epoch_value = 0
        tracker.records = defaultdict(list)

        final_epoch = min(self.max_train_steps,
                          self.train_epochs)
        for i_epoch in range(initial_epoch, final_epoch):
            logging.debug("=> Epoch {}".format(i_epoch))
            tracker.current_epoch = i_epoch

            if self.lr_scheduler_level == "epoch" and lr_tensor_name is not None:
                lr = self.sess.run(lr_tensor_name)
                logging.debug("Epoch {} Learning Rate : {:10.3e}".format(
                    i_epoch, lr))

            self.train_one_epoch(tracker)
            self.valid_one_epoch(tracker)

        return tracker
