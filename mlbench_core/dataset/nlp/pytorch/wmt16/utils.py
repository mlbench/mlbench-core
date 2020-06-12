import numpy as np
import torch

from mlbench_core.dataset.nlp.pytorch.wmt16 import wmt16_config


def build_collate_fn(sort):
    """Builds the collate function that adds the lengths of each datapoint to the batch

    Args:
        sort (bool): Sort within each batch

    Returns:
        func
    """

    def collate_seq(seq):
        """Builds batches for training or inference.
        Batches are returned as pytorch tensors, with padding.

        Args:
            seq (list[`obj`:torch.Tensor]): The batch

        Returns:
            (`obj`:torch.Tensor, `obj`:torch.Tensor): The batch and the lengths of each sequence
        """
        lengths = torch.tensor([len(s) for s in seq], dtype=torch.int64)
        batch_length = max(lengths)

        shape = (len(seq), batch_length)
        seq_tensor = torch.full(shape, wmt16_config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[i, :end_seq].copy_(s[:end_seq])

        seq_tensor = seq_tensor.t()

        return seq_tensor, lengths

    def parallel_collate(seqs):
        """Builds batches from parallel dataset (src, tgt), optionally sorts batch
        by src sequence length.

        Args:
            seqs (tuple): Tuple of (data, target) sequences

        Returns:
            (tuple, tuple): The data and target, along with the lengths
        """
        src_seqs, trg_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(
                *sorted(enumerate(src_seqs), key=lambda x: len(x[1]), reverse=True)
            )
            trg_seqs = [trg_seqs[idx] for idx in indices]

        src_seqs = collate_seq(src_seqs)
        trg_seqs = collate_seq(trg_seqs)
        return src_seqs, trg_seqs

    return parallel_collate


def get_data_dtype(vocab_size):
    """Returns the appropriate dtype for the given vocab size

    Args:
        vocab_size (int):

    Returns:
        (:obj:`np.dtype`, :obj:`torch.dtype`): One of `np.int16`, `np.int32` or `np.int64`
    """
    if vocab_size <= np.iinfo(np.int16).max:
        dtype = np.int16
        torch_dtype = torch.int16
    elif vocab_size <= np.iinfo(np.int32).max:
        dtype = np.int32
        torch_dtype = torch.int32
    elif vocab_size <= np.iinfo(np.int64).max:
        dtype = np.int64
        torch_dtype = torch.int64
    else:
        raise ValueError("Vocabulary size is too large")
    return dtype, torch_dtype


def process_raw_data(file_name, max_size=None):
    """Reads the lines of the given filename without any processing

    Args:
        file_name (str): Full file path
        max_size (int): Maximum size of result

    Returns:
        (list[str]): The read data
    """
    with open(file_name, mode="r", encoding="utf-8") as f:
        data = f.readlines()

    if max_size:
        data = data[:max_size]
    return data


def filter_raw_data(raw_src, raw_trg, min_len=0, max_len=float("inf")):
    """Filters raw data on source and target lengths

    Args:
        raw_src (list[str]): Raw source data
        raw_trg (list[str]): Raw target data
        min_len (int): Minimum length
        max_len (int): Maximum length

    Returns:
        (list[str], list[int],list[str], list[int]): The filtered data and the lengths
    """
    filtered_src = []
    filtered_trg = []
    filtered_src_len = []
    filtered_trg_len = []
    for src, trg in zip(raw_src, raw_trg):
        src_len = src.count(" ") + 1
        trg_len = trg.count(" ") + 1
        if min_len <= src_len <= max_len and min_len <= trg_len <= max_len:
            filtered_src.append(src)
            filtered_trg.append(trg)
            filtered_src_len.append(src_len)
            filtered_trg_len.append(trg_len)
    return filtered_src, filtered_src_len, filtered_trg, filtered_trg_len


def process_data(file_name, tokenizer, max_size=None):
    """Reads the line of a file and segments the lines using the tokenizer

    Args:
        file_name (str): Full file path
        tokenizer (:obj:`WMT16Tokenizer`): The tokenizer to use
        max_size (Optional[int]): The maximum size of result

    Returns:
        (list[int]): Tokenized data
    """
    data = []
    with open(file_name, mode="r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_size and idx == max_size:
                break
            entry = tokenizer.segment(line)
            entry = torch.tensor(entry)
            data.append(entry)
    return data


def filter_data(src, trg, min_len=0, max_len=float("inf")):
    """Filters data on source and target lengths

    Args:
        src (list[int]): Tokenized source data
        trg (list[int]): Tokenized target data
        min_len (int): Minimum length
        max_len (int): Maximum length

    Returns:
        (list[int], list[int]): The filtered data (tokenized)
    """
    filtered_src = []
    filtered_trg = []
    for src, trg in zip(src, trg):
        if min_len <= len(src) <= max_len and min_len <= len(trg) <= max_len:
            filtered_src.append(src)
            filtered_trg.append(trg)
    return filtered_src, filtered_trg
