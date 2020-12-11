import logging
import os
import unicodedata
from collections import OrderedDict


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    return OrderedDict((l.strip(), i) for i, l in enumerate(open(vocab_file, 'r', encoding='utf-8')))


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    return text.strip().split()


class FullTokenizer(object):
    """Runs end-to-end tokenization, applying punctuation splitting & WordPiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'")

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict((ids, tok) for tok, ids in self.vocab.items())
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab=self.vocab)
        self.max_len = max_len

    def __call__(self, text):
        """End-to-end tokenization using subtokenizers."""
        tokens = []
        for token in self.basic_tokenizer(text):
            for subtoken in self.wordpiece_tokenizer(token):
                tokens.append(subtoken)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = [self.vocab[token] for token in tokens]
        if self.max_len and len(ids) > self.max_len:
            raise ValueError(
                f"Token indices sequence length is longer than the specified maximum "
                f"sequence length for this BERT model ({len(ids)} > {self.max_len}). Running this "
                f"sequence through BERT will result in indexing errors"
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        return [self.ids_to_tokens[i] for i in ids]


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""


    def __init__(self, do_lower_case=True,
                 never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split


    def __call__(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        return whitespace_tokenize(' '.join(split_tokens))


    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = [char for char in text if unicodedata.category(char) != 'Mn']
        return ''.join(output)


    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)

        start_new_word = True
        output = []
        for char in chars:
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)

        return [''.join(x) for x in output]


    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            if self._is_chinese_char(ord(char)):
                output += [' ', char, ' ']
            else:
                output.append(char)
        return ''.join(output)


    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        return (0x4E00  <= cp <= 0x9FFF  or
                0x3400  <= cp <= 0x4DBF  or
                0x20000 <= cp <= 0x2A6DF or
                0x2A700 <= cp <= 0x2B73F or
                0x2B740 <= cp <= 0x2B81F or
                0x2B820 <= cp <= 0x2CEAF or
                0xF900  <= cp <= 0xFAFF  or
                0x2F800 <= cp <= 0x2FA1F)


    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordPieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def __call__(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of WordPiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    return char in (' ', '\t', '\n', '\r') or unicodedata.category(char) == 'Zs'


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ('\t', '\n', '\r'):
        return False
    return unicodedata.category(char).startswith('C')


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(char).startswith('P')
