import logging

import torch
import torch.optim
import torch.utils.data
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average

logger = logging.getLogger("mlbench")
LOG_EVERY_N_BATCHES = 25


def _record_train_batch_stats(
    batch_idx, loss, batch_size, tracker, num_batches_per_device_train
):
    r"""Record the stats in a training batch.

    Args:
        batch_idx (int): The id of the current batch
        loss (float): The loss of the batch
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use.
        num_batches_per_device_train (int): Number of batches per train epoch
    """
    progress = batch_idx / num_batches_per_device_train
    progress += tracker.current_epoch

    log_to_api = (
        batch_idx % LOG_EVERY_N_BATCHES == 0
        or batch_idx == num_batches_per_device_train
    )

    if tracker:
        tracker.record_loss(loss, batch_size, log_to_api=log_to_api)
    status = "Epoch {:5.2f} Batch {:4}: ".format(progress, batch_idx)

    logger.info(status + str(tracker))


class GNMTTrainer:
    def __init__(
        self,
        model,
        criterion,
        fp_optimizer,
        scheduler,
        translator,
        rank,
        schedule_per,
        tracker,
        metrics,
        iter_size,
    ):
        self.model = model
        self.batch_first = model.batch_first
        self.criterion = criterion
        self.epoch = 0
        self.rank = rank
        # Optimizers & Scheduler
        self.fp_optimizer = fp_optimizer

        self.schedule_per = schedule_per
        self.scheduler = scheduler
        self.device = next(model.parameters()).device

        self.translator = translator
        self.metrics = metrics
        self.iter_size = iter_size

        self.tracker = tracker

    def compute_model_output(self, src, tgt):
        src, src_length = src
        tgt, tgt_length = tgt

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
        else:
            output = self.model(src, src_length, tgt[:-1])

        return output

    def set_tracker(self, tracker):
        self.tracker = tracker

    def compute_loss(self, src, tgt, output):
        src, src_length = src
        tgt, tgt_length = tgt
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        num_toks = {"tgt": int(sum(tgt_length - 1)), "src": int(sum(src_length))}

        if self.batch_first:
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)

        else:
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1), tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= B * self.iter_size

        loss_per_token = loss_per_batch / num_toks["tgt"]
        loss_per_sentence = loss_per_batch / B

        return loss, loss_per_token, loss_per_sentence, num_toks

    def optimize(self, batch_idx, data, num_batches_per_device_train):

        # Whether to update the weights at this iteration
        update = (batch_idx % self.iter_size) == self.iter_size - 1
        if self.tracker:
            self.tracker.batch_start()

        # Clear gradients in the optimizer.
        if (batch_idx % self.iter_size) == 0:
            self.fp_optimizer.zero_grad()
            if self.tracker:
                self.tracker.record_batch_step("init")

        # Compute the output
        src, tgt = data.src, data.trg
        output = self.compute_model_output(src, tgt)

        if self.tracker:
            self.tracker.record_batch_step("fwd_pass")

        # Compute the loss
        stats = self.compute_loss(src, tgt, output)
        loss, loss_per_token, loss_per_sentence, num_toks = stats
        #

        if self.tracker:
            self.tracker.record_batch_step("comp_loss")

        # Backprop
        self.fp_optimizer.backward_loss(loss)

        if self.tracker:
            self.tracker.record_batch_step("backprop")

        # Opt step
        if update:
            self.fp_optimizer.step()
            if self.tracker:
                self.tracker.record_batch_step("opt_step")

            # Learning rate scheduler
            if self.schedule_per == "batch":
                self.scheduler.step()

        if self.tracker:
            self.tracker.batch_end()

        if self.batch_first:
            batch_size = output.size(0)
        else:
            batch_size = output.size(1)

        _record_train_batch_stats(
            batch_idx=batch_idx,
            loss=loss_per_token,
            batch_size=batch_size,
            tracker=self.tracker,
            num_batches_per_device_train=num_batches_per_device_train,
        )

    def _training(self):
        torch.set_grad_enabled(True)
        self.model.train()

        if self.tracker:
            self.tracker.train()

    def _eval(self):

        torch.set_grad_enabled(False)
        self.model.eval()

        # Set tracker in validation mode
        if self.tracker:
            self.tracker.validation()
            self.tracker.validation_start()

    def train_round(
        self, train_loader, val_loader, bleu_score=False, validate_every=None
    ):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.
        Args:
            data_loader: Data loader

        Returns:

        """
        # Set in training mode
        self._training()

        num_batches_per_device_train = len(train_loader)

        for batch_idx, data in enumerate(train_loader):
            self.optimize(batch_idx, data, num_batches_per_device_train)

            if bleu_score and (batch_idx + 1) % validate_every == 0:
                self.validation_round(val_loader)
                self._training()

        if self.schedule_per == "epoch":
            self.scheduler.step()

    def validate(self, loader):
        losses = AverageMeter()

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for data in loader:

                # Inference
                src, trg = data.src, data.trg
                output = self.compute_model_output(src, trg)

                # Compute loss
                stats = self.compute_loss(src, trg, output)
                loss, loss_per_token, loss_per_sentence, num_toks = stats

                # Update loss
                losses.update(loss_per_token, num_toks["tgt"])

                # Update metrics
                translated, targets = self.translator.translate(src, trg)
                for metric in self.metrics:
                    metric_value = metric(loss, translated, targets)
                    size = src[0].shape[0] if self.batch_first else src[0].shape[1]

                    metric.update(metric_value, size)

        metrics_averages = {metric: metric.average().item() for metric in self.metrics}
        loss_average = global_average(losses.sum, losses.count).item()
        return metrics_averages, loss_average

    def validation_round(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tracker = self.tracker
        self._eval()
        # Gather metrics and loss average
        metrics_values, loss = self.validate(data_loader)
        if tracker:
            tracker.validation_end()

        if len(metrics_values) > 0:
            # Save metrics
            if tracker:
                for metric, value in metrics_values.items():
                    tracker.record_metric(metric, value, log_to_api=True)

                    global_metric_value = global_average(value, 1).item()

                    if self.rank == 0:
                        tracker.record_stat(
                            "global_{}".format(metric.name),
                            global_metric_value,
                            log_to_api=True,
                        )

            #
            if self.rank == 0 and tracker:
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
            if self.rank == 0:
                logger.info("Validation loss={:.3f}".format(loss))

        if tracker:
            tracker.record_loss(loss, log_to_api=True)

            global_loss = global_average(loss, 1).item()

            if self.rank == 0:
                tracker.record_stat("global_loss", global_loss, log_to_api=True)

        return tracker.is_best() if tracker else False
