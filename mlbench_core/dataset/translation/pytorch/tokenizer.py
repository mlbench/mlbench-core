import os
from collections import defaultdict
from functools import partial

import sacremoses
import subword_nmt.apply_bpe
import torch
from mlbench_core.dataset.translation.pytorch import config


def _pad_vocabulary(vocab, math):
    """
    Pads vocabulary to a multiple of 'pad' tokens.

    Args:
        vocab (list): list with vocabulary
        math (str): Math precision. either `fp_16`, `manual_fp16` or `fp32`

    Returns:
        list: padded vocabulary
    """
    if math == "fp16" or math == "manual_fp16":
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
    vocab = [config.PAD_TOKEN, config.UNK_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN]
    with open(vocab_fname, encoding="utf-8") as vfile:
        for line in vfile:
            vocab.append(line.strip())

    vocab = _pad_vocabulary(vocab, math_precision)

    tok2idx = defaultdict(partial(int, config.UNK))
    for idx, token in enumerate(vocab):
        tok2idx[token] = idx

    idx2tok = {}
    for key, value in tok2idx.items():
        idx2tok[value] = key

    return vocab, idx2tok, tok2idx


def _pad_batch(segmented_batch):
    """
    Given a batch of segmented tokens, adds a PAD token
    so that they all have the same length

    Args:
        segmented_batch (list):

    Returns:
        list
    """
    max_len = max(len(x) for x in segmented_batch)

    for line in segmented_batch:
        line += [config.PAD] * (max_len - len(line))

    return segmented_batch


class WMT14Tokenizer:
    def __init__(
        self,
        base_dir,
        batch_first=False,
        is_target=False,
        include_lengths=False,
        lang=None,
        math_precision=None,
        separator="@@",
    ):
        """
        Tokenizer Class for WMT14 that uses the whole vocabulary
        Args:
            base_dir (str): Base directory for files
            batch_first (bool): Batch as first dimension
            include_lengths (bool): Include sentence length
            lang (dict): With keys `src` and `trg` designating source and target language
            math_precision (str): Math precision
            separator:
        """
        self.separator = separator
        self.lang = lang
        self.batch_first = batch_first
        self.include_lengths = include_lengths

        # base_dir = os.path.join(base_dir, "wmt14")
        bpe_fname = os.path.join(base_dir, config.BPE_CODES)
        vocab_fname = os.path.join(base_dir, config.VOCAB_FNAME)

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

        self.is_target = is_target

    def process(self, batch, device=None):
        """
        Processes a batch of inputs by segmenting and putting in
        appropriate format
        Args:
            batch (list): The list of lines to process
            device (str): The device to use

        Returns:
            tensor
        """
        segmented = [self.segment(line) for line in batch]
        lengths = torch.tensor([len(x) for x in segmented], device=device)
        segmented = _pad_batch(segmented)
        tensor = torch.tensor(segmented, device=device)

        if not self.batch_first:
            tensor = torch.t(tensor)
        if self.include_lengths:
            return tensor, lengths

        return tensor

    def segment(self, line):
        """
        Tokenizes single sentence and adds special BOS and EOS tokens.

        Args:
            line (str): sentence

        Returns:
            list representing tokenized sentence
        """
        entry = [self.tok2idx[i] for i in line]
        entry = [config.BOS] + entry + [config.EOS]
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

        detok = detok.replace(config.BOS_TOKEN, "")
        detok = detok.replace(config.EOS_TOKEN, "")
        detok = detok.replace(config.PAD_TOKEN, "")
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
