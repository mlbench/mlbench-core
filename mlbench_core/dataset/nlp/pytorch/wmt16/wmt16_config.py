"""Configuration for WMT16 dataset"""
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "<\s>"

PAD, UNK, BOS, EOS = 0, 1, 2, 3
BPE_CODES = "bpe.32000"
VOCAB_FNAME = "vocab.bpe.32000"

TRAIN_FNAME = "train.tok.clean.bpe.32000"
VAL_FNAME = "newstest2014.tok.bpe.32000"

EXTS = (".en", ".de")
