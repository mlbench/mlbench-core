import os
from collections import defaultdict
from functools import partial

import torch

from . import wmt16_config


def _pad_vocabulary(vocab, math):
    """
    Pads vocabulary to a multiple of 'pad' tokens.

    Args:
        vocab (list): list with vocabulary
        math (str): Math precision. either `fp_16`, `manual_fp16` or `fp32`

    Returns:
        list: padded vocabulary
    """
    if math == "fp16" or math == "amp_fp16":
        pad = 8
    elif math == "fp32":
        pad = 1
    else:
        raise NotImplementedError()

    vocab_size = len(vocab)
    padded_vocab_size = (vocab_size + pad - 1) // pad * pad
    for i in range(0, padded_vocab_size - vocab_size):
        token = f"madeupword{i:04d}"
        vocab.append(token)
    assert len(vocab) % pad == 0
    return vocab


class WMT16Tokenizer:
    """Tokenizer Class for WMT16 that uses the whole vocabulary

    Args:
        base_dir (str): Base directory for files
        math_precision (str): Math precision
        separator (str): BPE
    """

    def __init__(
        self, base_dir, math_precision=None, separator="@@",
    ):
        self.separator = separator

        vocab = [
            wmt16_config.PAD_TOKEN,
            wmt16_config.UNK_TOKEN,
            wmt16_config.BOS_TOKEN,
            wmt16_config.EOS_TOKEN,
        ]
        vocab_fname = os.path.join(base_dir, wmt16_config.VOCAB_FNAME)

        with open(vocab_fname, encoding="utf-8") as vfile:
            for line in vfile:
                vocab.append(line.strip())

        vocab = _pad_vocabulary(vocab, math_precision)
        self.vocab_size = len(vocab)

        self.tok2idx = defaultdict(partial(int, wmt16_config.UNK))
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx

        self.idx2tok = {}
        for key, value in self.tok2idx.items():
            self.idx2tok[value] = key

    def segment(self, line):
        """
        Tokenizes single sentence and adds special BOS and EOS tokens.

        :param line: sentence

        returns: list representing tokenized sentence
        """
        line = line.strip().split()
        entry = [self.tok2idx[i] for i in line]
        entry = [wmt16_config.BOS] + entry + [wmt16_config.EOS]
        return entry

    def detokenize(self, inputs, delim=" "):
        """
        Detokenizes single sentence and removes token separator characters.

        :param inputs: sequence of tokens
        :param delim: tokenization delimiter

        returns: string representing detokenized sentence
        """
        detok = delim.join([self.idx2tok[idx] for idx in inputs])
        detok = detok.replace(self.separator + " ", "")
        detok = detok.replace(self.separator, "")

        detok = detok.replace(wmt16_config.BOS_TOKEN, "")
        detok = detok.replace(wmt16_config.EOS_TOKEN, "")
        detok = detok.replace(wmt16_config.PAD_TOKEN, "")
        detok = detok.strip()
        return detok
