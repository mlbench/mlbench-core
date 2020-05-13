import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from mlbench_core.dataset.nlp.pytorch.wmt17 import Dictionary
from mlbench_core.dataset.nlp.pytorch.wmt17.utils import (
    IndexedDataset,
    collate_tokens,
    data_file_path,
    index_file_path,
)
from mlbench_core.dataset.util.tools import maybe_download_and_extract_tar_gz

logger = logging.getLogger("mlbench")


class WMT17Dataset(Dataset):
    urls = [
        (
            "https://storage.googleapis.com/mlbench-datasets/translation/wmt17_en_de.tar.gz",
            "wmt17_en_de.tar.gz",
        )
    ]
    name = "wmt17"
    dirname = ""

    prefixes = {"train": "train", "validation": "dev", "test": "test"}

    def __init__(
        self,
        root,
        lang=("en", "de"),
        download=False,
        train=False,
        validation=False,
        test=False,
        left_pad=(True, False),
        max_positions=(256, 256),
        seq_len_multiple=1,
        shuffle=True,
    ):
        src_lang, trg_lang = lang
        self.left_pad_source, self.left_pad_target = left_pad
        self.max_source_positions, self.max_target_positions = max_positions

        self.seq_len_multiple = seq_len_multiple
        self.shuffle = shuffle
        if download:
            url, file_name = self.urls[0]
            maybe_download_and_extract_tar_gz(root, file_name, url)

        self.src_dict = Dictionary.load(
            os.path.join(root, "dict.{}.txt".format(src_lang))
        )
        self.trg_dict = Dictionary.load(
            os.path.join(root, "dict.{}.txt".format(trg_lang))
        )

        if train:
            self.prefix = self.prefixes["train"]
        elif validation:
            self.prefix = self.prefixes["validation"]
        elif test:
            self.prefix = self.prefixes["test"]
        else:
            raise NotImplementedError()

        assert self.src_dict.pad() == self.trg_dict.pad()
        assert self.src_dict.eos() == self.trg_dict.eos()

        self.src_path = os.path.join(
            root, "{}.{}-{}.{}".format(self.prefix, src_lang, trg_lang, src_lang)
        )
        self.trg_path = os.path.join(
            root, "{}.{}-{}.{}".format(self.prefix, src_lang, trg_lang, trg_lang)
        )

        assert self._exists()

        self.src_data = IndexedDataset(self.src_path)
        self.trg_data = IndexedDataset(self.trg_path)

        self.src_sizes = np.array(self.src_data.sizes)
        self.trg_sizes = np.array(self.trg_data.sizes)

        print(
            "| Sentences are being padded to multiples of: {}".format(
                self.seq_len_multiple
            )
        )

    def __len__(self):
        return len(self.src_data)

    def _exists(self):

        return (
            os.path.exists(index_file_path(self.src_path))
            and os.path.exists(data_file_path(self.src_path))
            and os.path.exists(index_file_path(self.trg_path))
            and os.path.join(data_file_path(self.trg_path))
        )

    def __getitem__(self, index):
        return {
            "id": index,
            "source": self.src_data[index],
            "target": self.trg_data[index],
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return _collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            bsz_mult=8,
            seq_len_multiple=self.seq_len_multiple,
        )

    def get_dummy_batch(
        self, max_tokens_per_batch, max_positions, src_len=256, tgt_len=256
    ):
        max_source_positions, max_target_positions = self._get_max_positions(
            max_positions
        )
        src_len, tgt_len = (
            min(src_len, max_source_positions),
            min(tgt_len, max_target_positions),
        )
        n_seq_per_batch_based_on_longest_seq = max_tokens_per_batch // max(
            src_len, tgt_len
        )

        return self.collater(
            [
                {
                    "id": i,
                    "source": self.src_dict.dummy_sentence(src_len),
                    "target": self.trg_dict.dummy_sentence(tgt_len)
                    if self.trg_dict is not None
                    else None,
                }
                for i in range(n_seq_per_batch_based_on_longest_seq)
            ]
        )

    def ordered_indices(self, partition_indices=None, seed=None):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.RandomState(seed).permutation(
                len(self) if partition_indices is None else partition_indices
            )
        else:
            indices = (
                np.arange(len(self)) if partition_indices is None else partition_indices
            )

        if self.trg_sizes is not None:
            indices = indices[np.argsort(self.trg_sizes[indices], kind="mergesort")]

        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    def _get_max_positions(self, max_positions=None):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions

        assert len(max_positions) == 2

        max_src_pos, max_tgt_pos = max_positions

        return (
            min(self.max_source_positions, max_src_pos),
            min(self.max_target_positions, max_tgt_pos),
        )


def _collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    bsz_mult=8,
    seq_len_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            bsz_mult,
            seq_len_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge("source", left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target", left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            "target", left_pad=left_pad_target, move_eos_to_beginning=True,
        )
        ntokens = sum(len(s["target"]) for s in samples)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    return {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": target,
    }


def collater_isolated(samples, seq_len_multiple, left_pad_source, left_pad_target):
    """Merge a list of samples to form a mini-batch."""
    return _collate(
        samples,
        pad_idx=1,
        eos_idx=2,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        bsz_mult=8,
        seq_len_multiple=seq_len_multiple,
    )


def get_dummy_batch_isolated(max_tokens_per_batch, max_positions, seq_len_multiple):
    """Creates a dummy batch"""
    max_source_positions, max_target_positions = max_positions[0], max_positions[1]
    src_len, tgt_len = max_source_positions, max_target_positions
    n_seq_per_batch_based_on_longest_seq = max_tokens_per_batch // max(src_len, tgt_len)

    nspecial = 3
    ntok_alloc = 33712
    eos_id = 2
    dummy_seq_src = torch.Tensor(src_len).uniform_(nspecial + 1, ntok_alloc).long()
    dummy_seq_src[-1] = eos_id

    dummy_seq_tgt = torch.Tensor(tgt_len).uniform_(nspecial + 1, ntok_alloc).long()
    dummy_seq_tgt[-1] = eos_id

    return collater_isolated(
        [
            {"id": i, "source": dummy_seq_src, "target": dummy_seq_tgt}
            for i in range(n_seq_per_batch_based_on_longest_seq)
        ],
        seq_len_multiple,
        left_pad_source=True,
        left_pad_target=False,
    )
