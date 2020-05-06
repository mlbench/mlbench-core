import os

import torch
from torch.utils.data import Dataset

from mlbench_core.dataset.util.tools import maybe_download_and_extract_tar_gz

from .wmt14 import WMT14Tokenizer, wmt14_config


def _construct_filter_pred(min_len, max_len=None):
    """
    Constructs a filter predicate
    Args:
        min_len (int): Min sentence length
        max_len (int): Max sentence length

    Returns:
        func
    """
    filter_pred = lambda x: not (x[0] < min_len or x[1] < min_len)
    if max_len is not None:
        filter_pred = lambda x: not (
            x[0] < min_len or x[0] > max_len or x[1] < min_len or x[1] > max_len
        )

    return filter_pred


def build_collate_fn(sort):
    """Builds the collate function that adds the lengths of each datapoint to the batch

    Args:
        sort (bool): Sort within each batch

    Returns:
        func
    """

    def collate_seq(seq):
        """Builds batches for training or inference.
        Batches are returned as pytorch tensors, with padding.

        Args:
            seq (list[`obj`:torch.Tensor]): The batch

        Returns:
            (`obj`:torch.Tensor, `obj`:torch.Tensor): The batch and the lengths of each sequence
        """
        lengths = torch.tensor([len(s) for s in seq], dtype=torch.int64)
        batch_length = max(lengths)

        shape = (len(seq), batch_length)
        seq_tensor = torch.full(shape, wmt14_config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[i, :end_seq].copy_(s[:end_seq])

        seq_tensor = seq_tensor.t()

        return seq_tensor, lengths

    def parallel_collate(seqs):
        """Builds batches from parallel dataset (src, tgt), optionally sorts batch
        by src sequence length.

        Args:
            seqs (tuple): Tuple of (data, target) sequences

        Returns:
            (tuple, tuple): The data and target, along with the lengths
        """
        src_seqs, trg_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(
                *sorted(enumerate(src_seqs), key=lambda x: len(x[1]), reverse=True)
            )
            trg_seqs = [trg_seqs[idx] for idx in indices]

        src_seqs = collate_seq(src_seqs)
        trg_seqs = collate_seq(trg_seqs)
        return src_seqs, trg_seqs

    return parallel_collate


class WMT14Dataset(Dataset):
    """Dataset for WMT14 en to de translation
    Based on `torchtext.datasets.WMT14`

    Args:
        root (str): Root folder where to download files
        lang (dict): Language translation pair
        math_precision (str): One of `fp16` or `fp32`. The precision used during training
        download (bool): Download the dataset from source
        train (bool): Load train set
        validation (bool): Load validation set
        lazy (bool): Load the dataset in a lazy format
        min_len (int): Minimum sentence length
        max_len (int | None): Maximum sentence length
        max_size (int | None): Maximum dataset size
    """

    urls = [
        (
            "https://storage.googleapis.com/mlbench-datasets/translation/wmt16_en_de.tar.gz",
            "wmt16_en_de.tar.gz",
        )
    ]
    name = "wmt14"
    dirname = ""

    def __init__(
        self,
        root,
        lang=None,
        math_precision=None,
        download=True,
        train=False,
        validation=False,
        lazy=False,
        sort=False,
        min_len=0,
        max_len=None,
        max_size=None,
    ):
        self.lazy = lazy

        super(WMT14Dataset, self).__init__()
        if download:
            url, file_name = self.urls[0]
            maybe_download_and_extract_tar_gz(root, file_name, url)

        src_tokenizer = WMT14Tokenizer(root, lang=lang, math_precision=math_precision,)
        trg_tokenizer = WMT14Tokenizer(root, lang=lang, math_precision=math_precision,)

        self.vocab_size = src_tokenizer.vocab_size
        self.fields = {"src": src_tokenizer, "trg": trg_tokenizer}

        self.max_len = max_len
        self.min_len = min_len

        if train:
            path = os.path.join(root, wmt14_config.TRAIN_FNAME)
        elif validation:
            path = os.path.join(root, wmt14_config.VAL_FNAME)
        else:
            raise NotImplementedError()

        self.examples, self.indices = self._process_data(
            path,
            filter_pred=_construct_filter_pred(min_len, max_len),
            lazy=lazy,
            sort=sort,
            max_size=max_size,
        )

    def _process_data(self, path, filter_pred, sort=False, lazy=False, max_size=None):
        """Loads data from given path and processes the lines

        Args:
            path (str): Dataset directory path
            filter_pred (func): Filter predicate function (to filter inputs)
            lazy (bool): Whether to load the dataset in lazy mode
            max_size (int | None): Maximum size of dataset

        Returns:
            List: The list of examples
        """
        src_path, trg_path = tuple(
            os.path.expanduser(path + x) for x in wmt14_config.EXTS
        )
        examples = []
        src_lengths = []
        with open(src_path, mode="r", encoding="utf-8") as src_file, open(
            trg_path, mode="r", encoding="utf-8"
        ) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()

                should_consider = filter_pred(
                    (src_line.count(" ") + 1, trg_line.count(" ") + 1)
                )
                if src_line != "" and trg_line != "" and should_consider:
                    src_lengths.append(src_line.count(" ") + 1)
                    if lazy:
                        examples.append((src_line, trg_line))
                    else:
                        examples.append(self._parse_example((src_line, trg_line)))

                if max_size and len(examples) >= max_size:
                    break

        indices = list(range(len(examples)))
        if sort:
            indices, _ = zip(*sorted(enumerate(src_lengths), key=lambda x: x[1]))
        return examples, indices

    def __len__(self):
        return len(self.examples)

    def _parse_example(self, example):
        src_line, trg_line = example
        return (
            self.fields["src"].parse_line(src_line),
            self.fields["trg"].parse_line(trg_line),
        )

    def __getitem__(self, item):
        if self.lazy:
            return self._parse_example(self.examples[self.indices[item]])
        else:
            return self.examples[self.indices[item]]
