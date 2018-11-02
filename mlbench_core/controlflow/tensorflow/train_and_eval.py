import tensorflow as tf


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


class ControlFlow(object):
    def __init__(self, train_op, data_loader, sess, is_training, config, loss, metrics):
        # Save the placeholders
        self.is_training = is_training

        # TODO: Distinguish train dataloader with validation dataloader
        self.data_loader = data_loader
        self.sess = sess
        self.loss = loss
        self.metrics = metrics
        self.train_op = train_op
        self.config = config

    def train_one_epoch(self, i_epoch):
        self.sess.run(self.data_loader.tr_data_init_op)

        num_batches = self.data_loader.num_batches_per_epoch_for_train

        losses = AverageMeter()
        prec1 = AverageMeter()
        prec5 = AverageMeter()

        for i_batch in range(num_batches):
            g_batch = i_epoch * num_batches + i_batch

            # Get dataset and target from a batch
            data, target = self.sess.run(self.data_loader.tr_next_batch)

            out = self.sess.run(
                {
                    "metrics": self.metrics,
                    "loss": self.loss,
                    "train_op": self.train_op
                },
                feed_dict={
                    self.data_loader.inputs: data,
                    self.data_loader.labels: target,
                    self.is_training: True
                })

            p1, p5 = out['metrics']
            # TODO: Hide Loss
            losses.update(out["loss"], n=target.shape[0])
            prec1.update(p1, n=target.shape[0])
            prec5.update(p5, n=target.shape[0])

        print("{}/{} loss={:10.3e} prec1={:10.2f}% prec5={:10.2f}%"
              .format(i_epoch, i_batch, losses.val, prec1.val * 100,
                      prec5.val * 100))

    def valid_one_epoch(self, i_epoch):
        self.sess.run(self.data_loader.val_data_init_op)

        num_batches = self.data_loader.num_batches_per_epoch_for_eval

        losses = AverageMeter()
        prec1 = AverageMeter()
        prec5 = AverageMeter()

        for i_batch in range(num_batches):
            # Get dataset and target from a batch
            data, target = self.sess.run(self.data_loader.val_next_batch)

            out = self.sess.run(
                {
                    "metrics": self.metrics,
                    "loss": self.loss,
                },
                feed_dict={
                    self.data_loader.inputs: data,
                    self.data_loader.labels: target,
                    self.is_training: False
                })

            p1, p5 = out['metrics']
            losses.update(out["loss"], n=target.shape[0])
            prec1.update(p1, n=target.shape[0])
            prec5.update(p5, n=target.shape[0])

            print("{}/{} Validation loss={:10.3e} prec1={:10.2f}% prec5={:10.2f}%"
                  .format(i_epoch, i_batch, losses.val,
                          prec1.val * 100, prec5.val * 100))

        print("{} Validation loss={:10.3e} prec1={:10.2f}% prec5={:10.2f}%"
              .format(i_epoch, losses.val, prec1.val * 100,
                      prec5.val * 100))

    def train_and_eval(self, initial_epoch=0):
        # Initialize Variables
        self.sess.run(tf.group(tf.global_variables_initializer()))
        for i_epoch in range(initial_epoch, self.config.train_epochs):
            print("=> Epoch {}".format(i_epoch))

            self.train_one_epoch(i_epoch)
            self.valid_one_epoch(i_epoch)
