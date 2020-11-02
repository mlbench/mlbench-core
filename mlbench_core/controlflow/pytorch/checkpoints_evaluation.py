"""Evaluate training/validation set using models in checkpoints"""
import logging

import torch

from mlbench_core.aggregation.pytorch.centralized import AllReduceAggregation
from mlbench_core.controlflow.pytorch.helpers import iterate_dataloader
from mlbench_core.utils.pytorch.distributed import global_average

logger = logging.getLogger("mlbench")


class CheckpointsEvaluationControlFlow(object):
    """Evaluate models on training / validation dataset.

    Args:
        ckpt_dir (str): Path to checkpoints.
        rank (int): The rank of the current process
        world_size (int): The total number of workers
        checkpointer (:obj:`Checkpointer`): Used to load checkpoints.
        model (:obj:`torch.optim.Optimizer`): An optimizer for the given model.
        epochs (int): Number of epochs to traing.
        loss_function (:obj:`torch.nn.modules.loss._Loss`): loss function.
        metrics (:obj:`list` of :obj:`mlbench_core.evaluation.pytorch.*`): metrics like TopKAccuracy.
        use_cuda (bool): Whether to train on GPU or not. Default: `False`
        dtype (str): The datatype to use for the dataloader data
        max_batch_per_epoch (int): Maximum number of batches per epoch. Whole dataset
        is used if not specified. Default: `None`
    """

    def __init__(
        self,
        ckpt_dir,
        rank,
        world_size,
        checkpointer,
        model,
        epochs,
        loss_function,
        metrics,
        use_cuda=False,
        dtype=None,
        max_batch_per_epoch=None,
    ):
        self.ckpt_dir = ckpt_dir
        self.rank = rank
        self.checkpointer = checkpointer
        self.model = model
        self.epochs = epochs
        self.loss_function = loss_function
        self.metrics = metrics
        self.dtype = dtype
        self.max_batch_per_epoch = max_batch_per_epoch
        self.use_cuda = use_cuda

        self.model_agg_fn = AllReduceAggregation(world_size=world_size).agg_model()

        self._check_checkpoints()

    def _check_checkpoints(self):
        for epoch in range(self.epochs):
            self.checkpointer.checkpoint_exists(self.ckpt_dir, self.rank, epoch)

    def _load_model(self, epoch):
        # Load epoch-rank model
        model = self.checkpointer.load_model_by_epoch(
            self.ckpt_dir, self.rank, epoch, self.model
        )

        # aggregate models
        self.model_agg_fn(model, op="avg_world")
        return model

    def evaluate_by_epochs(self, dataloader):
        """Evaluate dataset using the averaged models.

        In each epoch each process loads models and averages them. The averaged model is
        used to evaluate train / validation dataset.

        Args:
            dataloader (:obj:`torch.utils.data.DataLoader`): The dataset to be evaluated.

        Returns:
            list: list of stats of models in each epoch.
        """
        stats_list = []
        for epoch in range(self.epochs):
            # Same model for all workers.
            model = self._load_model(epoch)
            model.eval()

            stats = {"epoch": epoch, "count": 0, "total_loss": 0}
            for metric in self.metrics:
                stats["total_" + metric.name] = 0

            data_iter = iterate_dataloader(
                dataloader, self.dtype, self.max_batch_per_epoch, self.use_cuda
            )

            with torch.no_grad():
                for i, (data, target) in enumerate(data_iter):
                    output = model(data)

                    # Compute loss and metrics.
                    count = len(target)
                    stats["count"] += count
                    stats["total_loss"] += self.loss_function(output, target) * count
                    for metric in self.metrics:
                        stats["total_" + metric.name] += metric(output, target) * count

                    logger.info(
                        "E{:4}B{:4}: total loss={:10.3e}".format(
                            epoch, i, stats["total_loss"] / stats["count"]
                        )
                    )

            # Keep globally averaged loss / metrics, etc.
            stats["loss"] = global_average(stats["total_loss"], stats["count"]).item()
            for metric in self.metrics:
                stats[metric.name] = global_average(
                    stats["total_" + metric.name], stats["count"]
                ).item()
                del stats["total_" + metric.name]
            del stats["count"], stats["total_loss"]

            stats_list.append(stats)
        return stats_list
