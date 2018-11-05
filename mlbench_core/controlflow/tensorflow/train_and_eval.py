import tensorflow as tf
from collections import defaultdict

from mlbench_core.evaluation import AverageMeter
from mlbench_core.utils import Tracker


class ControlFlow(object):
    def __init__(self, train_op, data_loader, sess, is_training, config, loss, metrics):
        # Save the placeholders
        self.is_training = is_training

        self.data_loader = data_loader
        self.sess = sess
        self.loss = loss
        self.metrics = metrics
        self.train_op = train_op
        self.config = config

    def train_one_epoch(self, tracker):
        self.sess.run(self.data_loader.tr_data_init_op)

        num_batches = self.data_loader.num_batches_per_epoch_for_train

        loss_meter = AverageMeter()
        metrics_meter = [AverageMeter() for _ in self.metrics]

        for i_batch in range(num_batches):
            g_batch = tracker.current_epoch * num_batches + i_batch

            # Get dataset and target from a batch
            data, target = self.sess.run(self.data_loader.tr_next_batch)

            out = self.sess.run(
                {
                    "metrics": [m['value'] for m in self.metrics],
                    "loss": self.loss,
                    "train_op": self.train_op
                },
                feed_dict={
                    self.data_loader.inputs: data,
                    self.data_loader.labels: target,
                    self.is_training: True
                })

            # Update tracker
            loss_meter.update(out["loss"], n=target.shape[0])
            for meter, o in zip(metrics_meter, out['metrics']):
                meter.update(o, n=target.shape[0])

            print(("{}/{} loss={:10.3e} | metrics: [" + " ".join
                   (["{: 10.2e}" for _ in metrics_meter]) + "] | best epoch {} ({:10.2e})")
                  .format(tracker.current_epoch, i_batch, loss_meter.avg,
                          *[m.avg for m in metrics_meter], tracker.best_epoch, tracker.best_epoch_value))

        # Record
        tracker.records['train_loss'].append(loss_meter.avg)
        for metric, meter in zip(self.metrics, metrics_meter):
            tracker.records['train_' + metric['name']].append(meter.avg)

    def valid_one_epoch(self, tracker):
        self.sess.run(self.data_loader.val_data_init_op)

        num_batches = self.data_loader.num_batches_per_epoch_for_eval

        loss_meter = AverageMeter()
        metrics_meter = [AverageMeter() for _ in self.metrics]

        for i_batch in range(num_batches):
            # Get dataset and target from a batch
            data, target = self.sess.run(self.data_loader.val_next_batch)

            out = self.sess.run(
                {
                    "metrics": [m['value'] for m in self.metrics],
                    "loss": self.loss,
                },
                feed_dict={
                    self.data_loader.inputs: data,
                    self.data_loader.labels: target,
                    self.is_training: False
                })

            # Update tracker
            loss_meter.update(out["loss"], n=target.shape[0])
            for meter, o in zip(metrics_meter, out['metrics']):
                meter.update(o, n=target.shape[0])

            print(("{}/{} Validation loss={:10.3e} | metrics: [" + " ".join
                   (["{: 10.2f}" for _ in metrics_meter]) + "]")
                  .format(tracker.current_epoch, i_batch, loss_meter.avg,
                          *[m.avg for m in metrics_meter]))

        # Record
        tracker.records['val_loss'].append(loss_meter.avg)
        for i, metric, meter in zip(range(len(self.metrics)), self.metrics, metrics_meter):
            metric_name = 'val_' + metric['name']

            # Here we implicitly assume the larger metrics value means better results
            if i == 0:
                # primary metrics
                if (len(tracker.records[metric_name]) == 0 or
                        meter.avg > max(tracker.records[metric_name])):
                    tracker.best_epoch = tracker.current_epoch
                    tracker.best_epoch_value = meter.avg

            tracker.records[metric_name].append(meter.avg)

    def train_and_eval(self, initial_epoch=0, lr_scheduler=None):
        # Initialize Variables
        self.sess.run(tf.group(tf.global_variables_initializer()))

        # Tracker for stats
        tracker = Tracker()
        tracker.current_epoch = 0
        tracker.best_epoch = 0
        tracker.best_epoch_value = 0
        tracker.records = defaultdict(list)

        final_epoch = min(self.config.max_train_steps,
                          self.config.train_epochs)
        for i_epoch in range(initial_epoch, final_epoch):
            print("=> Epoch {}".format(i_epoch))
            tracker.current_epoch = i_epoch

            if (self.config.lr_scheduler_level == "epoch" and lr_scheduler is not None):
                self.sess.run(lr_scheduler.assign(i_epoch))
                print(i_epoch, self.sess.run(
                    tf.get_default_graph().get_tensor_by_name("learning_rate:0")
                ))

            self.train_one_epoch(tracker)
            self.valid_one_epoch(tracker)

        return tracker
