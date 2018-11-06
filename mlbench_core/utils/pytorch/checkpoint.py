import os
import shutil
import torch

from mlbench_core.utils.pytorch.distributed import elementwise_min


class Checkpointer(object):
    def __init__(self, ckpt_run_dir, rank, checkpoint_all=False):
        self.dirname = ckpt_run_dir
        self.rank = rank
        self.checkpoint_all = checkpoint_all
        self.runtime = {'cumu_time_val': []}

    def get_ckpt_id(self, epoch):
        # {epoch}_{batch} can be sorted
        return "{epoch}_{rank}.pth.tar".format(epoch=epoch, rank=self.rank)

    def save(self, tracker, model, optimizer, scheduler, epoch, is_best):
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
            torch.save(state, best_model_path)

    @staticmethod
    def load(ckpt_run_dir, rank, model, optimizer, scheduler):
        checkpoint_path = determine_restore_ckpt_path(rank, ckpt_run_dir)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "No checkpoint found at '{}' for rank '{}'".format(ckpt_run_dir, rank))

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        checkpoint_all = checkpoint['checkpoint_all']

        checkpointer = Checkpointer(ckpt_run_dir, rank, checkpoint_all)
        checkpointer.runtime['cumu_time_val'] = checkpoint['tracker']['cumu_time_val']

        return checkpointer, model, optimizer, scheduler


def determine_restore_ckpt_path(rank, checkpoint_root):
    """Determine the checkpoint path to restore."""

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
