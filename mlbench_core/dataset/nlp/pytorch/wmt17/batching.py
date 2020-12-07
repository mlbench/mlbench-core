import sys

import numpy as np


def _roundup(x, multiple):
    """ Round up `x` to multiple of `multiple`"""
    return int((x + multiple - 1) // multiple * multiple)


def _rounddown(x, multiple):
    """ Round down `x` to multiple of `multiple`"""
    return int(x // multiple * multiple)


def _is_batch_full(num_tokens, max_tokens, max_sentences, batch_length):
    """Returns true if batch is full

    Args:
        num_tokens (int): number of tokens in batch
        max_tokens (Optional[int]): Max number of tokens in batch
        max_sentences (Optional[int]): Max number of sentences in batch
        batch_length (int): Current batch length

    Returns:
        (bool): Whether the batch is full or not
    """
    if batch_length == 0 or (max_tokens is None and max_sentences is None):
        return False

    elif batch_length == max_sentences or num_tokens > max_tokens:
        return True
    else:
        return False


def _make_batches(
    src_lengths,
    trg_lengths,
    indices,
    max_tokens,
    max_sentences,
    max_len,
    bsz_mult,
    pad_seq,
):
    """Creates the batches for WMT17 Dataset

    Args:
        src_lengths (iterable[int]): The source lengths
        trg_lengths (iterable[int]): The target lengths
        indices (iterable[int]): Indices to consider (must be sorted by lengths)
        max_tokens (Optional[int]): Max tokens per batch
        max_sentences (optional[int]): Max sentences per batch
        max_len (int): Maximum sequence length
        bsz_mult (int): Batch size multiple
        pad_seq (int): Multiple to round up to for sequence lengths

    Returns:
        (list[list[int]]): The list of batches, referenced by index
    """
    batches = []
    nelem = len(indices)
    sample_len = 0

    num_seqs_mult = bsz_mult // pad_seq if (bsz_mult % pad_seq == 0) else bsz_mult

    sample_lens = []
    batch = []
    for i in range(nelem):
        idx = indices[i]
        sample_num_tokens = max(src_lengths[idx], trg_lengths[idx])
        if sample_num_tokens > max_len:
            continue

        sample_len = max(sample_len, sample_num_tokens)
        padded_sample_len = (
            _roundup(sample_len, bsz_mult)
            if len(batch) < num_seqs_mult
            else _roundup(sample_len, pad_seq)
        )
        sample_lens.append(sample_num_tokens)
        num_tokens = (len(batch) + 1) * padded_sample_len

        if _is_batch_full(num_tokens, max_tokens, max_sentences, len(batch)):
            sequences = len(batch)
            if (sequences % num_seqs_mult != 0) and (sequences > num_seqs_mult):
                pad_sequences_opt_seqs = _rounddown(sequences, num_seqs_mult)
                total_tokens_opt_seqs = padded_sample_len * pad_sequences_opt_seqs

                pad_seq_len_opt_seqlen = _roundup(padded_sample_len, bsz_mult)
                pad_sequences_opt_seqlen = max_tokens // pad_seq_len_opt_seqlen
                total_tokens_opt_seqlen = padded_sample_len * pad_sequences_opt_seqlen

                if total_tokens_opt_seqs >= total_tokens_opt_seqlen:
                    sequences = pad_sequences_opt_seqs
                else:
                    sequences = pad_sequences_opt_seqlen

            # Copy overflowing data to next batch if sequences < len(batch)
            new_batch = batch[sequences:]
            batch = batch[:sequences]
            sample_lens = sample_lens[sequences:]  # keep lens for next iter
            sample_len = max(sample_lens)
            batches.append(batch)
            batch = new_batch

        batch.append(idx)

    while len(batch) > 0:
        sequences = max(
            _rounddown(len(batch), num_seqs_mult), len(batch) % num_seqs_mult
        )
        new_batch = batch[sequences:]
        batch = batch[:sequences]
        batches.append(batch)
        batch = new_batch

    return batches


def get_batches(
    dataset, max_tokens=None, max_sentences=None, bsz_mult=8, shuffle=True, seed=0
):
    """Creates the batches for the given dataset

    Args:
        dataset (:obj:`torch.utils.Dataset`): The dataset
        max_tokens (Optional[int]): Maximum number of tokens in one batch.
            Default: `None` (no limit)
        max_sentences (int): Maximum number of sentences per batch.
            Default: `None` (no limit)
        bsz_mult (int): Batch size multiple. Default: 8
        shuffle (bool): Shuffle batches. Default: `True`
        seed (int): Seed to use for shuffling. Default: 0

    Returns:
        (list[list[int]]): The batches
    """
    if hasattr(dataset, "indices"):
        partition_indices = dataset.indices
    else:
        partition_indices = None
    indices = dataset.ordered_indices(partition_indices, seed=seed)
    src_sizes = dataset.src_sizes
    trg_sizes = dataset.trg_sizes

    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_pos_num = min(dataset.max_source_positions, dataset.max_target_positions)

    batches = _make_batches(
        src_sizes,
        trg_sizes,
        indices,
        max_tokens,
        max_sentences,
        max_pos_num,
        bsz_mult,
        dataset.seq_len_multiple,
    )

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(batches)
    return batches
