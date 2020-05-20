import torch


class Dictionary(object):
    """Dictionary Class for WMT17 Dataset.
    Essentially a mapping from symbols to consecutive integers

    Args:
        pad (str): Padding symbol to use
        eos (str): End of String symbol to use
    """

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

        Args:
            f (str): Dictionary file name
            ignore_utf_errors (bool): Ignore UTF-8 related errors
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
            word = line.strip()[1:-1]
            count = 1
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)

        n_pad_tokens_on_end = 33712 - len(d.symbols)

        for i in range(n_pad_tokens_on_end):
            pad_str = "<pad000" + str(i) + ">"
            d.indices[pad_str] = len(d.symbols)
            d.symbols.append(pad_str)
            d.count.append(1)

        return d
