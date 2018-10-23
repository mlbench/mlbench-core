import os
import shutil
import torch

from mlbench_core.utils.pytorch.distributed import elementwise_min


def get_ckpt_run_dir(checkpoint_root, run_id, dataset_name, model_name, optimizer_name):
    if isinstance(run_id, str):
        assert '_' not in run_id
    run_dir = "{run_id}_{dataset}_{model}_{optimizer}".format(
        run_id=run_id, dataset=dataset_name, model=model_name, optimizer=optimizer_name)
    return os.path.join(checkpoint_root, run_dir)


def get_ckpt_id(epoch, rank):
    # {epoch}_{batch} can be sorted
    return "{epoch}_{rank}.pth.tar".format(epoch=epoch, rank=rank)


def determine_restore_ckpt_path(rank, checkpoint_root, run_id):
    """Determine the checkpoint path to restore."""
    ckpt_run_dirs = os.listdir(checkpoint_root)

    # parse run_ids
    found_ckpts = list(filter(lambda x: x.split("_", 1)[
                       0] == str(run_id), ckpt_run_dirs))

    if len(found_ckpts) == 1:
        found_ckpts = found_ckpts[0]

        ckpt_ids = os.listdir(os.path.join(checkpoint_root, found_ckpts))
        ckpt_ids = list(set(ckpt_ids) - set(['model_best.pth.tar']))

        latest = sorted(map(lambda x: x.split("_")[:2], ckpt_ids))[-1]
        latest = elementwise_min(torch.Tensor([int(latest[0])]))
        epoch = latest[0]

        path = os.path.join(checkpoint_root, found_ckpts,
                            get_ckpt_id(epoch, rank))
        return path
    else:
        raise FileNotFoundError("Found {}".format(found_ckpts))


def save(config, tracker, model, optimizer, scheduler, is_best):
    if config.checkpoint == 'never':
        return

    state = {
        'tracker': tracker,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    dirname = config.ckpt_run_dir
    filename = get_ckpt_id(tracker.current_epoch, config.rank)
    checkpoint_path = os.path.join(dirname, filename)
    best_model_path = os.path.join(dirname, 'model_best.pth.tar')

    if config.checkpoint == 'all':
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_model_path)
    elif config.checkpoint == 'best':
        torch.save(state, best_model_path)
    else:
        raise NotImplementedError


def resume(config, model, optimizer, scheduler):
    # FIXME: using tracker
    checkpoint_path = determine_restore_ckpt_path(
        config.rank, config.checkpoint_root, config.run_id)

    print('Try to load previous model from the path:{}'.format(checkpoint_path))
    if os.path.isfile(checkpoint_path):
        # get checkpoint.
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict):
            # restore models
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise NotImplementedError(type(checkpoint))

            # logging.
        print("Loaded checkpoint '{}' (epoch {})".format(
            checkpoint_path, checkpoint['config_runtime']['current_epoch']))
        checkpoint['config_runtime']['current_epoch'] = checkpoint['config_runtime']['current_epoch']
    else:
        raise FileNotFoundError(
            "No checkpoint found at '{}'".format(config.resume))
    return checkpoint['config_runtime']


def maybe_resume(config, model, optimizer, scheduler):
    """Recover the state of config, model, optimizer and scheduler."""
    if config.resume:
        # reload model from the latest checkpoint.
        config.runtime = resume(config, model, optimizer, scheduler)
    else:
        config.runtime = {}
    return config
