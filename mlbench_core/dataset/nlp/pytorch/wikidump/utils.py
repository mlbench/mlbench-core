import logging
import random

import torch

logger = logging.getLogger("mlbench")


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).
    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    seeds_tensor = torch.LongTensor(seeds).to(device)
    torch.distributed.broadcast(seeds_tensor, 0)
    seeds = seeds_tensor.tolist()
    return seeds


def generate_seeds(rng, size):
    """
    Generate list of random seeds
    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2 ** 32 - 1) for _ in range(size)]
    return seeds


def setup_seeds(master_seed, epochs, device, rank=0, world_size=1):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.
    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        if rank == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logger.info(f"Using random master seed: {master_seed}")
    else:
        # master seed was specified from command line
        logger.info(f"Using master seed from command line: {master_seed}")

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, world_size)

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds
