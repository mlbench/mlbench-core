import os

from mlbench_core.dataset.translation.pytorch import config
from mlbench_core.dataset.translation.pytorch.tokenizer import WMT14Tokenizer
from torchtext.data import Example, Dataset
from mlbench_core.dataset.util.tools import maybe_download_and_extract_tar_gz


def _construct_filter_pred(min_len, max_len):
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


def process_data(path, filter_pred, fields, lazy=False, max_size=None):
    """
    Loads data from the input file.
    """

    src_path, trg_path = tuple(os.path.expanduser(path + x) for x in config.EXTS)
    examples = []
    with open(src_path, mode="r", encoding="utf-8") as src_file, open(
        trg_path, mode="r", encoding="utf-8"
    ) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line, trg_line = src_line.strip(), trg_line.strip()

            should_consider = filter_pred(
                (src_line.count(" ") + 1, trg_line.count(" ") + 1)
            )
            if src_line != "" and trg_line != "" and should_consider:
                if lazy:
                    examples.append((src_line, trg_line))
                else:
                    examples.append(Example.fromlist([src_line, trg_line], fields))

            if max_size and len(examples) >= max_size:
                break
    return examples


class WMT14Dataset(Dataset):
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
        batch_first=False,
        include_lengths=False,
        lang=None,
        math_precision=None,
        download=True,
        train=False,
        validation=False,
        lazy=False,
        min_len=0,
        max_len=None,
        max_size=None,
    ):
        """WMT14 Dataset.

        Loads WMT14 dataset.
        Based on `torchtext.datasets.WMT14`

        Args:
            root (str): Root folder of WMT14 dataset
            download (bool): Download dataset
            train (bool): Whether to get the train or validation set.
                Default=True
            batch_first (bool): if True the model uses (batch,seq,feature)
                tensors, if false the model uses (seq, batch, feature)
        """

        self.lazy = lazy

        super(WMT14Dataset, self).__init__(examples=[], fields={})
        if download:
            url, file_name = self.urls[0]
            maybe_download_and_extract_tar_gz(root, file_name, url)

        src_tokenizer = WMT14Tokenizer(
            root,
            batch_first=batch_first,
            include_lengths=include_lengths,
            lang=lang,
            math_precision=math_precision,
        )
        trg_tokenizer = WMT14Tokenizer(
            root,
            batch_first=batch_first,
            include_lengths=include_lengths,
            lang=lang,
            math_precision=math_precision,
            is_target=True,
        )

        self.vocab_size = src_tokenizer.vocab_size
        self.list_fields = [("src", src_tokenizer), ("trg", trg_tokenizer)]

        self.fields = dict(self.list_fields)
        self.max_len = max_len
        self.min_len = min_len

        if train:
            path = os.path.join(root, config.TRAIN_FNAME)
        elif validation:
            path = os.path.join(root, config.VAL_FNAME)
        else:
            raise NotImplementedError()

        self.examples = process_data(
            path,
            filter_pred=_construct_filter_pred(min_len, max_len),
            fields=self.list_fields,
            lazy=lazy,
            max_size=max_size,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.lazy:
            src_line, trg_line = self.examples[item]
            return Example.fromlist([src_line, trg_line], self.list_fields)
        else:
            return self.examples[item]

    def __iter__(self):
        for x in self.examples:
            if self.lazy:
                src_line, trg_line = x
                yield Example.fromlist([src_line, trg_line], self.list_fields)
            else:
                yield x
