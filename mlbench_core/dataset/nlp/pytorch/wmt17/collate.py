import torch


def _collate_tokens(
    values,
    pad_idx,
    eos_idx,
    left_pad,
    move_eos_to_beginning=False,
    n_seq_per_batch_multiple=8,
    seq_len_multiple=1,
):
    """ Convert a list of 1d tensors into a padded 2d tensor.

    Args:
        values (list[torch.Tensor]): A list of tensors
        pad_idx (int): Padding symbol index
        eos_idx (int): EOS symbol index
        left_pad (bool): left- or right-padding (true: left, false: right)
        move_eos_to_beginning (bool): Reverse order of sequence of tokens (true: reverse, false: original)
        n_seq_per_batch_multiple (int): The number of sequences per batch to round down to
        seq_len_multiple (int): The number of tokens per sequence to round up to

    Returns:
        (:obj:`torch.Tensor`): The tensor of collated and padded tokens
    """
    size_of_seq_dim = max(v.size(0) for v in values)  # Unpadded size
    n_seq_in_batch = len(values)

    if n_seq_per_batch_multiple % seq_len_multiple == 0:
        n_seq_multiple = n_seq_per_batch_multiple / seq_len_multiple
    else:
        n_seq_multiple = n_seq_per_batch_multiple

    if n_seq_in_batch < n_seq_multiple or n_seq_in_batch % n_seq_multiple > 0:
        seq_len_multiple = n_seq_per_batch_multiple

    size_of_seq_dim = (
        (size_of_seq_dim + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple
    )  # Padded seq len, rounded up to next multiple

    padded_2d_tensor = values[0].new(len(values), size_of_seq_dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()

        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if left_pad:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][size_of_seq_dim - len(val) :])
    else:
        for idx, val in enumerate(values):
            copy_tensor(val, padded_2d_tensor[idx][: len(val)])

    return padded_2d_tensor


def collate_batch(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    bsz_mult=8,
    seq_len_multiple=1,
):
    """Collate a list of samples into a batch

    Args:
        samples (list[dict]): Samples to collate
        pad_idx (int): Padding symbol index
        eos_idx (int): EOS symbol index
        left_pad_source (bool): Pad sources on the left
        left_pad_target (bool): Pad sources on the right
        bsz_mult (int): Batch size multiple
        seq_len_multiple (int): Sequence length multiple

    Returns:
        (dict): `{'id' (list[int]): list of indexes,
                  'ntokens' (int): total number of tokens,
                  'net_input' (dict): input of net, containing 'src_tokens', 'src_lengths'
                                      and 'prev_output_tokens', all tensors,
                  'target' (Tensor): Target Tensor}`

    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return _collate_tokens(
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
