import os
from collections import Counter

import numpy as np
import torch

from mlbench_core.dataset.util.tools import maybe_download_and_extract_zip


class Dictionary(object):
    """Simple dictionary class to count word occurences and build vocabulary"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        """Adds `word` to counter

        Args:
            word (str): Word

        Returns:
            (int): Word index
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Wikitext2Dataset(object):
    """Class representing Wikitext2 dataset"""

    URL = (
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
        "wikitext-2-v1.zip",
    )

    def __init__(self, path, bptt, train=False, valid=False, test=False, min_seq_len=5):
        """
        Args:
            path (str): Root directory
            bptt (int): BPTT Length
            train (bool): Load train data
            valid (bool):  Load validation data
            test (bool): Load test data
            min_seq_len (int): Minimum sequence length. Default 5
        """

        url, filename = self.URL
        maybe_download_and_extract_zip(path, filename, url)

        path = os.path.join(path, "wikitext-2")
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "wiki.train.tokens"))
        self.valid = self.tokenize(os.path.join(path, "wiki.valid.tokens"))
        self.test = self.tokenize(os.path.join(path, "wiki.test.tokens"))

        if train:
            self.raw_data = self.train
        elif valid:
            self.raw_data = self.valid
        elif test:
            self.raw_data = self.test
        else:
            raise ValueError("Please specify data type")

        self.data = None
        self.bptt = bptt
        self.min_seq_len = min_seq_len

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def generate_batches(self, global_bsz, worker_bsz=None, rank=1):
        """Generates the batches given the batch size.
        Stores the matrix in `self.data`.

        Args:
            global_bsz (int): Sum of all worker batch sizes. Used to generate the batches
            worker_bsz (int): Worker batch size
            rank (int): Current worker rank
        """
        if not worker_bsz:
            worker_bsz = global_bsz
        assert (
            global_bsz % worker_bsz == 0
        ), "Worker batch size should be divisible by global"
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = self.raw_data.size(0) // global_bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.raw_data.narrow(0, 0, nbatch * global_bsz)
        # Evenly divide the data across the bsz batches.
        self.data = data.reshape(global_bsz, -1).t().contiguous()

        # Slice the batches for current worker
        start, end = (rank - 1) * worker_bsz, rank * worker_bsz
        self.data = self.data[:, start:end].contiguous()

    def _get_batch(self, i, random_length=False, cuda=False):
        """Gets a batch form the generated batches.
        If `random_length`, the sequence length will be sampled from:

        Normal(x, 5) where `x = self.bptt if np.random.random() < 0.95 else self.bptt / 2`

        Args:
            i (int): Batch index
            random_length (bool): Random sequence length. Default `False`
            cuda (bool): Use cuda acceleration

        Returns:
            (:obj:`torch.Tensor`, :obj:`torch.Tensor`, int): Data and target tensors, with sequence length
        """
        seq_len = None
        if random_length:
            bptt_len = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            # Prevent excessively small or negative sequence lengths
            seq_len = max(self.min_seq_len, int(np.random.normal(bptt_len, 5)))

        seq_len = min(seq_len if seq_len else self.bptt, len(self.data) - 1 - i)
        data = self.data[i : i + seq_len]
        target = self.data[i + 1 : i + 1 + seq_len].view(-1)
        if cuda:
            return data.cuda(), target.cuda(), seq_len
        return data, target, seq_len

    def get_loader(self, random_length=False, cuda=False):
        """Returns an iterator over the batches

        Args:
            random_length (bool): Random sequence length. Default `False`
            cuda (bool): Use cuda acceleration

        Returns:
            iterator over the batches
        """

        def _iterator():
            i = 0
            while i < self.data.size(0) - 2:
                data, target, seq_len = self._get_batch(
                    i, random_length=random_length, cuda=cuda
                )
                i += seq_len
                yield data, target

        return _iterator()
