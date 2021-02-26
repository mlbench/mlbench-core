import os
import random
from copy import deepcopy

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class _WorkerInitObj:
    """Object used for the dataloader's `worker_init_fn`"""

    def __init__(self, seed):
        """
        Args:
            seed (int): worker seed
        """
        self.seed = seed

    def __call__(self, _id):
        """Resets the seed to become `self.seed + _id`

        Args:
            _id (int): Index to add to current seed
        """
        np.random.seed(seed=self.seed + _id)
        random.seed(self.seed + _id)


class PretrainingDataset(Dataset):
    """Represents the Pretraining Dataset used to train BERT on language modelling
    Implementation taken from https://github.com/mlperf/training_results_v0.7/blob/master/NVIDIA/benchmarks/bert/implementations/pytorch/run_pretraining.py.

    The files need to be in HD5 format and pre-processed. Please read the doc for more info on preprocessing
    """

    def __init__(self, root, max_pred_length, worker_seed):
        """

        Args:
            root (str): Root directory where files are contained
            max_pred_length (int): Max prediction length
            worker_seed (int): Worker seed for data loader
        """
        self.root = root
        self.max_pred_length = max_pred_length
        self.raw_files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f)) and "part" in f
        ]
        self.num_files = len(self.raw_files)
        self.input_file, self.inputs = None, None
        self.shuffled_files, self.f_start_id = None, 0
        self.worker_init_fn = _WorkerInitObj(worker_seed)

    def __len__(self):
        """Returns total length of input samples

        Returns:
            (int): Total length of inputs
        """
        if self.inputs is None:
            raise ValueError("Cannot call len if no file has been read")
        return len(self.inputs[0])

    def _load_file(self, input_file):
        """Loads the given file's data

        Args:
            input_file (str): Full path of input file to read
        """

        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __getitem__(self, index):
        if self.inputs is None:
            raise ValueError("Cannot call getitem if no file has been read")
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]

    def reshuffle_files(self, seed):
        """Reshuffles the files using the given seed. Should be called at the beginning of each epoch

        Args:
            seed (int): Shuffling seed
        """
        self.shuffled_files = deepcopy(self.raw_files)
        random.Random(seed).shuffle(self.shuffled_files)
        self.shuffled_files.sort()
        self.f_start_id = 0
        self.num_files = len(self.shuffled_files)

    def select_from_files(self, rank=0, world_size=1):
        """Allows the selection of a file from the list of shuffled files. Files need to be shuffled before calling this
        method. Loads the file and the input data

        Args:
            rank (int): Worker rank (Default 0)
            world_size (int): World size (Default 1)

        """
        if self.shuffled_files is None:
            raise ValueError("Please shuffle files before selecting")
        if world_size > self.num_files:
            remainder = world_size % self.num_files
            data_file = self.shuffled_files[
                (self.f_start_id * world_size + rank + remainder * self.f_start_id)
                % self.num_files
            ]
        else:
            data_file = self.shuffled_files[
                (self.f_start_id * world_size + rank) % self.num_files
            ]

        self.input_file = data_file
        self._load_file(data_file)
