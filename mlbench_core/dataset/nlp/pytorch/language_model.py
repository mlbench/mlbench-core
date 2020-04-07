import math

from torch.utils.data import Dataset
from torchtext.experimental.datasets import WikiText2


class BPTTWikiText2(Dataset):
    """WikiText2 dataset with backpropagation through time.

    Args:
        bptt_len (int): Length of BPTT segments
        train (bool): Whether to get the train or validation set (default=True)
        tokenizer (:obj:`torchtext.data.utils.tokenizer`): Tokenizer to use
        root (str): Root folder for the dataset
        """

    def __init__(self, bptt_len, train=True, **kwargs):
        super(BPTTWikiText2, self).__init__()

        self.bptt_len = bptt_len

        train_set, _, val_set = WikiText2(**kwargs)
        self.vocab = train_set.get_vocab()

        if train:
            self.data = train_set
        else:
            if "vocab" in kwargs:
                del kwargs["vocab"]

            (val_set,) = WikiText2(vocab=self.vocab, data_select="valid", **kwargs)
            self.data = val_set

    def __getitem__(self, i):
        i = i * self.bptt_len
        seq_len = min(self.bptt_len, len(self.data) - i - 1)
        return self.data[i : i + seq_len], self.data[i + 1 : i + 1 + seq_len]

    def __len__(self):
        return math.ceil((len(self.data) - 1) / self.bptt_len)

    def __iter__(self):
        for i in range(0, len(self) * self.bptt_len, self.bptt_len):
            seq_len = min(self.bptt_len, len(self.data) - i - 1)
            text = self.data[i : i + seq_len]
            target = self.data[i + 1 : i + 1 + seq_len]
            yield text, target

    def get_vocab(self):
        return self.vocab
