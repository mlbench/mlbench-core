import os
from collections import defaultdict
from functools import partial

import torch

from . import wmt14_config

try:
    import sacremoses
    import subword_nmt.apply_bpe
except ImportError as e:
    pass


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


def _parse_vocab(vocab_fname, math_precision):
    """
    Parses the vocabulary file
    Args:
        vocab_fname (str): Vocab file name
        math_precision (str): Math mode. either `fp_16`, `manual_fp16` or `fp32`

    Returns:
        list, dict, dict: List of words, id-to-token and token-to-id dictionaries
    """
    vocab = [
        wmt14_config.PAD_TOKEN,
        wmt14_config.UNK_TOKEN,
        wmt14_config.BOS_TOKEN,
        wmt14_config.EOS_TOKEN,
    ]
    with open(vocab_fname, encoding="utf-8") as vfile:
        for line in vfile:
            vocab.append(line.strip())

    vocab = _pad_vocabulary(vocab, math_precision)

    tok2idx = defaultdict(partial(int, wmt14_config.UNK))
    for idx, token in enumerate(vocab):
        tok2idx[token] = idx

    idx2tok = {}
    for key, value in tok2idx.items():
        idx2tok[value] = key

    return vocab, idx2tok, tok2idx


class WMT14Tokenizer:
    """Tokenizer Class for WMT14 that uses the whole vocabulary

    Args:
        base_dir (str): Base directory for files
        lang (dict): With keys `src` and `trg` designating source and target language
        math_precision (str): Math precision
        separator:
    """

    def __init__(
        self, base_dir, lang=None, math_precision=None, separator="@@",
    ):
        self.separator = separator
        self.lang = lang

        # base_dir = os.path.join(base_dir, "wmt14")
        bpe_fname = os.path.join(base_dir, wmt14_config.BPE_CODES)
        vocab_fname = os.path.join(base_dir, wmt14_config.VOCAB_FNAME)

        if bpe_fname:
            with open(bpe_fname, "r", encoding="utf-8") as bpe_codes:
                self.bpe = subword_nmt.apply_bpe.BPE(bpe_codes)

        if vocab_fname:
            tmp = _parse_vocab(vocab_fname, math_precision)

            vocab, self.idx2tok, self.tok2idx = tmp
            self.vocab_size = len(vocab)

        if lang:
            self.moses_tokenizer = sacremoses.MosesTokenizer(lang["src"])
            self.moses_detokenizer = sacremoses.MosesDetokenizer(lang["trg"])

    def parse_line(self, line):
        return torch.tensor(self.segment(self.preprocess(line)))

    def segment(self, line):
        """
        Tokenizes single sentence and adds special BOS and EOS tokens.

        Args:
            line (str): sentence

        Returns:
            list representing tokenized sentence
        """
        entry = [self.tok2idx[i] for i in line]
        entry = [wmt14_config.BOS] + entry + [wmt14_config.EOS]
        return entry

    def detokenize_bpe(self, inp, delim=" "):
        """
        Detokenizes single sentence and removes token separator characters.

        Args:
            inp (list): sequence of tokens
            delim (str): tokenization delimiter

        Returns:
            string representing detokenized sentence
        """
        detok = delim.join([self.idx2tok[idx] for idx in inp])
        detok = detok.replace(self.separator + " ", "")
        detok = detok.replace(self.separator, "")

        detok = detok.replace(wmt14_config.BOS_TOKEN, "")
        detok = detok.replace(wmt14_config.EOS_TOKEN, "")
        detok = detok.replace(wmt14_config.PAD_TOKEN, "")
        detok = detok.strip()
        return detok

    def detokenize_moses(self, inp):
        output = self.moses_detokenizer.detokenize(inp.split())
        return output

    def detokenize(self, inp):
        detok_bpe = self.detokenize_bpe(inp)
        output = self.detokenize_moses(detok_bpe)
        return output

    def preprocess(self, x):
        return x.strip().split()
