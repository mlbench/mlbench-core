import os
import shutil
import torch
import dill

from mlbench_core.utils.pytorch.distributed import elementwise_min


class Checkpointer(object):
    """ A class for handling checkpoint saving and loading

    Args:
        ckpt_run_dir (str): The path of the checkpoint directory.
        rank (int): The rank of the current worker.
        checkpoint_all (bool): Whether to checkpoint on all epochs
            or just when a new best score was achieved. Default: `False`
    """
    def __init__(self, ckpt_run_dir, rank, checkpoint_all=False):
        self.dirname = ckpt_run_dir
        self.rank = rank
        self.checkpoint_all = checkpoint_all
        self.runtime = {'cumu_time_val': []}

    def get_ckpt_id(self, epoch):
        """ Get the name of a checkpoint

        Args:
            epoch (int): The current epoch.

        Returns:
            The name for the current checkpoint
        """
        # {epoch}_{batch} can be sorted
        return "{epoch}_{rank}.pth.tar".format(epoch=epoch, rank=self.rank)

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
            'tracker': tracker,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'checkpoint_all': self.checkpoint_all
        }

        filename = self.get_ckpt_id(tracker.current_epoch)
        checkpoint_path = os.path.join(self.dirname, filename)
        best_model_path = os.path.join(self.dirname, 'model_best.pth.tar')

        if self.checkpoint_all:
            torch.save(state, checkpoint_path)
            if is_best:
                shutil.copyfile(checkpoint_path, best_model_path)
        else:
            torch.save(state, best_model_path, pickle_module=dill)

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
                "No checkpoint found at '{}' for rank '{}'".format(ckpt_run_dir, rank))

        checkpoint = torch.load(checkpoint_path, pickle_module=dill)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        checkpoint_all = checkpoint['checkpoint_all']

        checkpointer = Checkpointer(ckpt_run_dir, rank, checkpoint_all)
        checkpointer.runtime['cumu_time_val'] = checkpoint['tracker']['cumu_time_val']

        return checkpointer, model, optimizer, scheduler


def determine_restore_ckpt_path(rank, checkpoint_root):
    """Determine the checkpoint path to restore.

    Args:
        rank (int): The rank of the current worker
        checkpoint_root (str): Folder path of checkpoint directory

    Returns:
        The path of the newest checkpoint for this worker
    """

    ckpt_ids = os.listdir(checkpoint_root)
    ckpt_ids = list(set(ckpt_ids) - set(['model_best.pth.tar']))

    ckpt_ids = filter(lambda x: x.split("_")[1] == rank, ckpt_ids)

    latest = sorted(ckpt_ids, reverse=True)

    path = os.path.join(checkpoint_root, latest[0])
    return path


def maybe_resume(config, model, optimizer, scheduler):
    """Recover the state of config, model, optimizer and scheduler."""
    if 'resume' in config and config['resume']:
        # reload model from the latest checkpoint.
        config['runtime'] = resume(config, model, optimizer, scheduler)
    else:
        config['runtime'] = {}
    return config
