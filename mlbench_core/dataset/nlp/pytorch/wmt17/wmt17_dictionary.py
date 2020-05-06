# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from collections import Counter

import torch

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif bpe_symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif bpe_symbol is not None:
        sentence = (sentence + " ").replace(bpe_symbol, "").rstrip()
    return sentence


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, pad="<pad>_", eos="<EOS>_"):
        self.pad_word, self.eos_word = pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        # Commented out and hard-coded since pad and eos are in the dictionary files already
        self.add_symbol("<lua_index_compat>")
        self.pad_index = 1
        self.eos_index = 2
        self.nspecial = 3

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        else:
            assert idx < len(self.symbols)

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        if sym in self.indices:
            return self.indices[sym]
        else:
            assert sym in self.indices

    def string(self, tensor, bpe_symbol=None):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t) for t in tensor)

        def token_string(i):
            return self[i]

        sent = " ".join(token_string(i) for i in tensor if i != self.eos())
        if bpe_symbol is not None:
            sent = (sent + " ").replace(bpe_symbol, "").rstrip()

        return sent

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0>
        <symbol1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, "r", encoding="utf-8") as fd:
                        return cls.load(fd)
                else:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fd:
                        return cls.load(fd)

            except FileNotFoundError as fnfe:
                raise fnfe

            except Exception:
                raise Exception(
                    "Incorrect encoding detected in {}, please rebuild the dataset".format(
                        f
                    )
                )

        d = cls()
        for line in f.readlines():
            word = line.strip()[1:-1]  ## Remove the single quotes
            count = 1
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)

        n_pad_tokens_on_end = 33712 - len(d.symbols)
        # assert n_pad_tokens_on_end == 3  ## DEBUG: remove later, sanity check

        for i in range(n_pad_tokens_on_end):
            pad_str = "<pad000" + str(i) + ">"
            d.indices[pad_str] = len(d.symbols)
            d.symbols.append(pad_str)
            d.count.append(1)

        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)

            with open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)

        for symbol, count in zip(
            self.symbols[self.nspecial :], self.count[self.nspecial :]
        ):
            print("{} {}".format(symbol, count), file=f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t
