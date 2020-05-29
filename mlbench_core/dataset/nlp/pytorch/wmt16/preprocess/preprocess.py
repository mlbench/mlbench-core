import argparse
import logging

import torch

from mlbench_core.dataset.nlp.pytorch.wmt16 import wmt16_config
from mlbench_core.dataset.nlp.pytorch.wmt16_dataset import WMT16Dataset

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNMT prepare data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-dir",
        default="data/wmt16_de_en",
        help="path to the directory with training/test data",
    )
    parser.add_argument(
        "--max-size",
        default=None,
        type=int,
        help="use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset",
    )

    parser.add_argument(
        "--math",
        default="amp_fp16",
        choices=["fp32", "fp16", "amp_fp16"],
        help="arithmetic type",
    )

    parser.add_argument(
        "--max-length-train",
        default=75,
        type=int,
        help="maximum sequence length for training \
                        (including special BOS and EOS tokens)",
    )
    parser.add_argument(
        "--min-length-train",
        default=0,
        type=int,
        help="minimum sequence length for training \
                        (including special BOS and EOS tokens)",
    )

    parser.add_argument(
        "--num-workers", default=2, type=int, help="Number of workers for loader"
    )
    parser.add_argument(
        "--batch-size", default=1024, type=int, help="Batch size for loader"
    )
    args = parser.parse_args()
    return args


def build_collate_fn(max_seq_len):
    def collate_seq(seq):
        lengths = torch.tensor([len(s) for s in seq])
        batch_length = max_seq_len

        shape = (len(seq), batch_length)
        seq_tensor = torch.full(shape, wmt16_config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[i, :end_seq].copy_(s[:end_seq])

        return seq_tensor, lengths

    def parallel_collate(seqs):
        src_seqs, tgt_seqs = zip(*seqs)
        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]])

    return parallel_collate


def main():
    args = parse_args()

    logger.info(f"Run arguments: {args}")

    train_data = WMT16Dataset(
        args.dataset_dir,
        lang=("en", "de"),
        math_precision=args.math,
        download=False,
        train=True,
        lazy=True,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size,
    )

    collate_fn = build_collate_fn(max_seq_len=args.max_length_train)

    train_data.write_as_preprocessed(
        collate_fn,
        args.min_length_train,
        args.max_length_train,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
