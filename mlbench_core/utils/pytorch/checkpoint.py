import enum
import json
import os
import shutil

import dill
import torch


class CheckpointFreq(enum.IntEnum):
    # Checkpoint every epoch or best epoch
    ALL = 1
    BEST = 2
    NONE = 3


class Checkpointer(object):
    """ A class for handling checkpoint saving and loading.

    Args:
        ckpt_run_dir (str): The path of the checkpoint directory.
        rank (int): The rank of the eurrent worker.
        freq (int): The frequency of checkpointing. Default: `CheckpointFreq.BEST`
        save_stats (bool): Save stats to additional text files. Default: `True`
    """

    def __init__(self, ckpt_run_dir, rank, freq=CheckpointFreq.BEST, save_stats=True):
        self.dirname = ckpt_run_dir
        self.rank = rank
        self.freq = freq
        self.save_stats = save_stats
        # self.runtime = {'cumu_time_val': []}

    def save(self, tracker, model, optimizer, scheduler, epoch, is_best):
        """ Saves a checkpoint

        Args:
            tracker (:obj:`mlbench_core.utils.pytorch.helpers.Tracker`): The
                metrics tracker object
            model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
            optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
            scheduler (:obj:`mlbench_core.lr_scheduler.pytorch.lr.*`): a scheduler for
                hyperparameters.
            epoch (int): The current epoch
            is_best (bool): Whether the current model is a new best scoring one
        """
        state = {
            "tracker": tracker,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "freq": self.freq,
        }

        filename = "{epoch}_{rank}.pth.tar".format(
            epoch=tracker.current_epoch, rank=self.rank
        )
        checkpoint_path = os.path.join(self.dirname, filename)
        best_model_path = os.path.join(self.dirname, "model_best.pth.tar")

        if self.freq == CheckpointFreq.ALL:
            torch.save(state, checkpoint_path, pickle_module=dill)
            if is_best:
                shutil.copyfile(checkpoint_path, best_model_path)
        elif self.freq == CheckpointFreq.BEST:
            torch.save(state, best_model_path, pickle_module=dill)
        elif self.freq != CheckpointFreq.NONE:
            raise NotImplementedError

        self._maybe_save_stats(tracker.records, tracker.current_epoch, self.rank)

    def _maybe_save_stats(self, records, epoch, rank):
        """Save the records in the tracker."""
        if self.save_stats:
            filename = os.path.join(self.dirname, "{}_{}.json".format(epoch, rank))
            with open(filename, "w") as f:
                json.dump(records, f)

    @staticmethod
    def load(ckpt_run_dir, rank, model, optimizer, scheduler):
        """ Loads a checkpoint

        Args:
            ckpt_run_dir (str): Folder path of checkpoint directory
            rank (int): The rank of the current worker
            model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.
            optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
            scheduler (:obj:`mlbench_core.lr_scheduler.pytorch.lr.*`): a scheduler for
                hyperparameters.

        Returns:
            A tuple of `(Checkpointer, model, optimizer, scheduler)`
        """

        checkpoint_path = determine_restore_ckpt_path(rank, ckpt_run_dir)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "No checkpoint found at '{}' for rank '{}'".format(ckpt_run_dir, rank)
            )

        checkpoint = torch.load(checkpoint_path, pickle_module=dill)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        tracker = checkpoint["tracker"]

        freq = checkpoint["freq"]

        checkpointer = Checkpointer(ckpt_run_dir, rank, freq)
        # checkpointer.runtime['cumu_time_val'] = checkpoint['tracker']['cumu_time_val']

        return checkpointer, model, optimizer, scheduler, tracker

    @staticmethod
    def load_model_by_epoch(ckpt_run_dir, rank, epoch, model):
        """ Loads a checkpoint

        Args:
            ckpt_run_dir (str): Folder path of checkpoint directory
            rank (int): The rank of the current worker
            epoch (int): Epoch of the model to be loaded.
            model (:obj:`torch.nn.Module`): a pytorch model to be trained and validated.

        Returns:
            `model`
        """
        checkpoint_path = os.path.join(
            ckpt_run_dir, "{}_{}.pth.tar".format(epoch, rank)
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "No checkpoint found at '{}' for rank '{}'".format(ckpt_run_dir, rank)
            )

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model"])
        return model

    @staticmethod
    def checkpoint_exists(ckpt_run_dir, rank, epoch):
        """ Check if a checkpoint exists.

        Args:
            ckpt_run_dir (str): Folder path of checkpoint directory
            rank (int): The rank of the current worker
            epoch (int): Epoch of the model to be loaded.

        Returns:
            `model`
        """
        checkpoint_path = os.path.join(
            ckpt_run_dir, "{}_{}.pth.tar".format(epoch, rank)
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "No checkpoint found at '{}' for rank '{}'".format(ckpt_run_dir, rank)
            )


def determine_restore_ckpt_path(rank, checkpoint_root):
    """Determine the checkpoint path to restore.

    Args:
        rank (int): The rank of the current worker
        checkpoint_root (str): Folder path of checkpoint directory

    Returns:
        The path of the newest checkpoint for this worker
    """
    ckpt_ids = os.listdir(checkpoint_root)
    ckpt_ids = list(filter(lambda x: x.endswith(".pth.tar"), ckpt_ids))
    ckpt_ids = list(set(ckpt_ids) - set(["model_best.pth.tar"]))

    ckpt_ids = filter(
        lambda x: x.split("_")[1][: -len(".pth.tar")] == str(rank), ckpt_ids
    )

    latest = sorted(ckpt_ids, reverse=True, key=lambda x: int(x.split("_")[0]))

    path = os.path.join(checkpoint_root, latest[0])
    return path
