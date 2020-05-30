import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mlbench_core.dataset.util.tools import maybe_download_and_extract_tar_gz

from .wmt16 import (
    WMT16Tokenizer,
    filter_data,
    filter_raw_data,
    get_data_dtype,
    process_data,
    process_raw_data,
    wmt16_config,
)


class WMT16Dataset(Dataset):
    """Dataset for WMT16 en to de translation

    Args:
        root (str): Root folder where to download files
        lang (tuple): Language translation pair
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

    def __init__(
        self,
        root,
        lang=("en", "de"),
        math_precision=None,
        download=True,
        train=False,
        validation=False,
        lazy=False,
        preprocessed=False,
        sort=False,
        min_len=0,
        max_len=None,
        max_size=None,
    ):
        super(WMT16Dataset, self).__init__()
        if download:
            url, file_name = self.urls[0]
            maybe_download_and_extract_tar_gz(root, file_name, url)

        if train:
            self.path = os.path.join(root, wmt16_config.TRAIN_FNAME)
        elif validation:
            self.path = os.path.join(root, wmt16_config.VAL_FNAME)
        else:
            raise NotImplementedError()

        self.min_len, self.max_len = min_len, max_len
        self.lazy = lazy
        self.preprocessed = preprocessed
        self.tokenizer = WMT16Tokenizer(root, math_precision=math_precision)

        # Data is not tokenized already
        if not preprocessed:
            src_path, trg_path = tuple(
                os.path.expanduser(self.path + "." + x) for x in lang
            )
            if lazy:
                raw_src = process_raw_data(src_path, max_size)
                raw_trg = process_raw_data(trg_path, max_size)
                assert len(raw_src) == len(
                    raw_trg
                ), "Source and target have different lengths"

                raw_src, src_lens, raw_trg, trg_lens = filter_raw_data(
                    raw_src, raw_trg, min_len - 2, max_len - 2
                )

                assert len(raw_src) == len(
                    raw_trg
                ), "Source and target have different lengths"
                # Adding 2 because EOS and BOS are added later during tokenization
                src_lengths = [i + 2 for i in src_lens]
                trg_lengths = [i + 2 for i in trg_lens]
                self.src = raw_src
                self.trg = raw_trg
            else:
                src = process_data(src_path, self.tokenizer, max_size)
                trg = process_data(trg_path, self.tokenizer, max_size)
                assert len(src) == len(trg), "Source and target have different lengths"

                src, trg = filter_data(src, trg, min_len, max_len)

                assert len(src) == len(trg), "Source and target have different lengths"

                src_lengths = [len(s) for s in src]
                trg_lengths = [len(t) for t in trg]
                self.src, self.trg = src, trg
        else:
            self.path = "{}.bin".format(self.path)
            self.file = None
            (
                file_max_len,
                file_min_len,
                file_vocab_size,
                src_lengths,
                trg_lengths,
            ) = self._extract_preprocessed_data()

            assert file_max_len == self.max_len
            assert file_min_len == self.min_len
            assert file_vocab_size == self.vocab_size

            self.dtype, _ = get_data_dtype(self.vocab_size)
            itemsize = np.iinfo(self.dtype).dtype.itemsize
            self.item_stride = itemsize * self.max_len * 2

        self.src_lengths = torch.tensor(src_lengths)
        self.trg_lengths = torch.tensor(trg_lengths)
        # self.lengths = self.src_lengths + self.trg_lengths

        self.sorted = False
        if sort:
            self._sort_by_src_length()
            self.sorted = True

    def _extract_preprocessed_data(self):
        if not (os.path.exists(self.path) and os.path.isfile(self.path)):
            raise ValueError("File {} not found".format(self.path))
        with open(self.path, "rb") as f:
            length = int(np.fromfile(f, np.int64, 1))
            file_vocab_size = int(np.fromfile(f, np.int64, 1))
            file_min_len = int(np.fromfile(f, np.int64, 1))
            file_max_len = int(np.fromfile(f, np.int64, 1))

            src_lengths = np.fromfile(f, np.int64, length)
            trg_lengths = np.fromfile(f, np.int64, length)
            self.offset = int(np.fromfile(f, np.int64, 1))
        return file_max_len, file_min_len, file_vocab_size, src_lengths, trg_lengths

    def _sort_by_src_length(self):
        self.src_lengths, indices = self.src_lengths.sort(descending=True)
        self.src = [self.src[i] for i in indices]
        self.trg = [self.trg[i] for i in indices]
        self.trg_lengths = [self.trg_lengths[i] for i in indices]

    def prepare(self):
        if self.preprocessed:
            self.file = open(self.path, "rb")

    def __getitem__(self, idx):
        if self.preprocessed:
            offset = self.offset + self.item_stride * idx
            self.file.seek(offset, os.SEEK_SET)
            data = np.fromfile(self.file, self.dtype, self.max_len * 2)
            data = data.astype(np.int64)
            src_len = self.src_lengths[idx]
            tgt_len = self.trg_lengths[idx]
            src = torch.tensor(data[0:src_len])
            tgt = torch.tensor(data[self.max_len : self.max_len + tgt_len])
            return src, tgt
        else:
            if self.lazy:
                src = torch.tensor(self.tokenizer.segment(self.src[idx]))
                trg = torch.tensor(self.tokenizer.segment(self.trg[idx]))
            else:
                src, trg = self.src[idx], self.trg[idx]
            return src, trg

    def write_as_preprocessed(
        self, collate_fn, min_len=0, max_len=50, num_workers=2, batch_size=1024
    ):
        loader = DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )

        srcs = []
        tgts = []
        src_lengths = []
        tgt_lengths = []
        dtype, torch_dtype = get_data_dtype(self.vocab_size)

        for (src, src_len), (tgt, tgt_len) in tqdm(loader):
            src_lengths.append(deepcopy(src_len.to(torch_dtype)))
            tgt_lengths.append(deepcopy(tgt_len.to(torch_dtype)))
            srcs.append(deepcopy(src))
            tgts.append(deepcopy(tgt))
            del src
            del src_len
            del tgt
            del tgt_len

        del loader
        import gc

        gc.collect()

        srcs = torch.cat(srcs)
        tgts = torch.cat(tgts)
        src_lengths = torch.cat(src_lengths)
        tgt_lengths = torch.cat(tgt_lengths)

        assert len(srcs) == len(tgts) == len(src_lengths) == len(tgt_lengths)
        length = len(srcs)

        offset = 0
        dest_fname = "{}.bin".format(self.path)
        with open(dest_fname, "wb") as f:
            offset += f.write((np.array(length, dtype=np.int64)))
            offset += f.write((np.array(self.vocab_size, dtype=np.int64)))
            offset += f.write((np.array(min_len, dtype=np.int64)))
            offset += f.write((np.array(max_len, dtype=np.int64)))
            offset += f.write((np.array(src_lengths, dtype=np.int64)))
            offset += f.write((np.array(tgt_lengths, dtype=np.int64)))

            offset += np.iinfo(np.int64).dtype.itemsize
            f.write((np.array(offset, dtype=np.int64)))

            del src_lengths
            del tgt_lengths
            gc.collect()
            data = torch.cat((srcs, tgts), dim=1).numpy()
            f.write((np.array(data, dtype=dtype)))

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.src_lengths)
