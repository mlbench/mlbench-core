r""" Create dataset and dataloader in PyTorch. """
import logging
import os

import spacy
from spacy.symbols import ORTH

import torchtext
import torchtext.datasets as nlp_datasets

_logger = logging.getLogger('mlbench')


def _get_text():
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok, batch_first=False)
    return TEXT


class Wikitext2(nlp_datasets.Wikitext2):
    """Wikitext2 Dataset.

    Loads Wikitext2 dataset.
    Based on `torchtext.datasets.Wikitext2`

    Args:
        root (str): Root folder of Imagenet dataset (without `train/` or `val/`)
        train (bool): Whether to get the train or validation set (default=True)
    """

    def __init__(self, root, train=True):
        self.train = train
        self.text_field = _get_text()
        self.root = root

        path = root

        if not os.path.exists(root) or not os.listdir(root):
            path = self.download(root)

        if train:
            path = os.path.join(path, 'wiki.train.tokens')
        else:
            path = os.path.join(path, 'wiki.valid.tokens')

        super(Wikitext2, self).__init__(path, self.text_field)
