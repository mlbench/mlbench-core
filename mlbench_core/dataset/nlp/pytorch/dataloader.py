r""" Create dataset and dataloader in PyTorch. """
import logging
import os

import spacy
from spacy.symbols import ORTH

import torchtext
import torchtext.datasets as nlp_datasets


def _get_text():
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok, batch_first=False)
    return TEXT


class Wikitext2(nlp_datasets.WikiText2):
    """Wikitext2 Dataset.

    Loads Wikitext2 dataset.
    Based on `torchtext.datasets.Wikitext2`

    Args:
        root (str): Root folder of Imagenet dataset (without `train/` or `val/`)
        train (bool): Whether to get the train or validation set (default=True)
    """

    def __init__(self, root, text_field=None, download=True, train=True):
        self.train = train

        self.text_field = text_field
        if not self.text_field:
            self.text_field = _get_text()

        self.root = root

        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wikitext-2/wikitext-2")

        if train:
            path = os.path.join(path, "wiki.train.tokens")
        else:
            path = os.path.join(path, "wiki.valid.tokens")

        super(Wikitext2, self).__init__(path, self.text_field)
