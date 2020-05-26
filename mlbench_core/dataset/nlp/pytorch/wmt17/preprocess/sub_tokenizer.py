# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines Subtokenizer class to encode and decode strings."""

import os
import re
import sys
import unicodedata

import numpy as np
import six
from six.moves import xrange

LUA = "<lua_index_compat>"
PAD = "<pad>_"
PAD_ID = 1
EOS = "<EOS>_"
EOS_ID = 2
UNK = "<bypass_unk>"
RESERVED_TOKENS = [LUA, PAD, EOS, UNK]

# Set of characters that will be used in the function _escape_token() (see func
# docstring for more details).
# This set is added to the alphabet list to ensure that all escaped tokens can
# be encoded.
_ESCAPE_CHARS = set(u"\\_u;0123456789")
# Regex for the function _unescape_token(), the inverse of _escape_token().
# This is used to find "\u", "\\", and "\###;" substrings in the token.
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
#
_UNDEFINED_UNICODE = u"\u3013"
#
# # Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i)
    for i in xrange(sys.maxunicode)
    if (
        unicodedata.category(six.unichr(i)).startswith("L")
        or unicodedata.category(six.unichr(i)).startswith("N")
    )
)


class Subtokenizer(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self, vocab_file, reserved_tokens=None):
        """Initializes class, creating a vocab file if data_files is provided."""
        print("Initializing Subtokenizer from file %s." % vocab_file)

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        elif reserved_tokens is "assumed_in_file":
            reserved_tokens = []

        self.subtoken_list = _load_vocab_file(vocab_file, reserved_tokens)
        self.alphabet = _generate_alphabet_dict(self.subtoken_list, reserved_tokens)
        self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)

        self.max_subtoken_length = 0
        for subtoken in self.subtoken_list:
            self.max_subtoken_length = max(self.max_subtoken_length, len(subtoken))

        # Create cache to speed up subtokenization
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size

    @staticmethod
    def init_from_existing_vocab_file(vocab_file):
        """Create subtoken vocabulary based on files, and save vocab to file.

        Args:
          vocab_file: String name of vocab file to store subtoken vocabulary.

        Returns:
          Subtokenizer object
        """

        if os.path.exists(vocab_file):
            print("Vocab file exists (%s)" % vocab_file)
        else:
            print("Vocab file does not exist (%s)" % vocab_file)

        return Subtokenizer(vocab_file, reserved_tokens="assumed_in_file")

    def encode(self, raw_string, add_eos=False):
        """Encodes a string into a list of int subtoken ids."""
        ret = []
        tokens = _split_string_to_tokens(raw_string)
        for token in tokens:
            ret.extend(self._token_to_subtoken_ids(token))
        if add_eos:
            ret.append(EOS_ID)
        return ret

    def _token_to_subtoken_ids(self, token):
        """Encode a single token into a list of subtoken ids."""
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value

        ret = _split_token_to_subtokens(
            _escape_token(token, self.alphabet),
            self.subtoken_to_id_dict,
            self.max_subtoken_length,
        )
        ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]

        self._cache[cache_location] = (token, ret)
        return ret

    def decode(self, subtokens):
        """Converts list of int subtokens ids into a string."""
        if isinstance(subtokens, np.ndarray):
            # Note that list(subtokens) converts subtokens to a python list, but the
            # items remain as np.int32. This converts both the array and its items.
            subtokens = subtokens.tolist()

        if not subtokens:
            return ""

        assert isinstance(subtokens, list) and isinstance(
            subtokens[0], int
        ), "Subtokens argument passed into decode() must be a list of integers."

        return _join_tokens_to_string(self._subtoken_ids_to_tokens(subtokens))

    def _subtoken_ids_to_tokens(self, subtokens):
        """Convert list of int subtoken ids to a list of string tokens."""
        escaped_tokens = "".join(
            [self.subtoken_list[s] for s in subtokens if s < len(self.subtoken_list)]
        )
        escaped_tokens = escaped_tokens.split("_")

        # All tokens in the vocabulary list have been escaped (see _escape_token())
        # so each token must be unescaped when decoding.
        ret = []
        for token in escaped_tokens:
            if token:
                ret.append(_unescape_token(token))
        return ret


def _load_vocab_file(vocab_file, reserved_tokens=None):
    """Load vocabulary while ensuring reserved tokens are at the top."""
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS

    subtoken_list = []
    with open(vocab_file, mode="r", newline="\n") as f:
        for line in f:
            subtoken = line.strip()
            subtoken = subtoken[1:-1]  # Remove surrounding single-quotes
            if subtoken in reserved_tokens:
                continue
            subtoken_list.append(subtoken)
    return reserved_tokens + subtoken_list


def _split_string_to_tokens(text):
    """Splits text to a list of string tokens."""
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in xrange(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != u" " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def _join_tokens_to_string(tokens):
    """Join a list of string tokens into a single string."""
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(u" ")
        ret.append(token)
    return "".join(ret)


def _escape_token(token, alphabet):
    r"""Replace characters that aren't in the alphabet and append "_" to token.

    Apply three transformations to the token:
      1. Replace underline character "_" with "\u", and backslash "\" with "\\".
      2. Replace characters outside of the alphabet with "\###;", where ### is the
         character's Unicode code point.
      3. Appends "_" to mark the end of a token.

    Args:
      token: unicode string to be escaped
      alphabet: list of all known characters

    Returns:
      escaped string
    """
    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


def _unescape_token(token):
    r"""Replaces escaped characters in the token with their unescaped versions.

    Applies inverse transformations as _escape_token():
      1. Replace "\u" with "_", and "\\" with "\".
      2. Replace "\###;" with the unicode character the ### refers to.

    Args:
      token: escaped string

    Returns:
      unescaped string
    """

    def match(m):
        r"""Returns replacement string for matched object.

        Matched objects contain one of the strings that matches the regex pattern:
          r"\\u|\\\\|\\([0-9]+);"
        The strings can be '\u', '\\', or '\###;' (### is any digit number).

        m.group(0) refers to the entire matched string ('\u', '\\', or '\###;').
        m.group(1) refers to the first parenthesized subgroup ('###').

        m.group(0) exists for all match objects, while m.group(1) exists only for
        the string '\###;'.

        This function looks to see if m.group(1) exists. If it doesn't, then the
        matched string must be '\u' or '\\' . In this case, the corresponding
        replacement ('_' and '\') are returned. Note that in python, a single
        backslash is written as '\\', and double backslash as '\\\\'.

        If m.goup(1) exists, then use the integer in m.group(1) to return a
        unicode character.

        Args:
          m: match object

        Returns:
          String to replace matched object with.
        """
        # Check if the matched strings are '\u' or '\\'.
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        # If m.group(1) exists, try and return unicode character.
        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return _UNDEFINED_UNICODE

    # Use match function to replace escaped substrings in the token.
    return _UNESCAPE_REGEX.sub(match, token)


def _list_to_index_dict(lst):
    """Create dictionary mapping list items to their indices in the list."""
    return {item: n for n, item in enumerate(lst)}


#
#
def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
    """Splits a token into subtokens defined in the subtoken dict."""
    ret = []
    start = 0
    token_len = len(token)
    while start < token_len:
        # Find the longest subtoken, so iterate backwards.
        for end in xrange(min(token_len, start + max_subtoken_length), start, -1):
            subtoken = token[start:end]
            if subtoken in subtoken_dict:
                ret.append(subtoken)
                start = end
                break
        else:  # Did not break
            # If there is no possible encoding of the escaped token then one of the
            # characters in the token is not in the alphabet. This should be
            # impossible and would be indicative of a bug.
            raise ValueError('Was unable to split token "%s" into subtokens.' % token)
    return ret


def _generate_alphabet_dict(iterable, reserved_tokens=None):
    """Create set of characters that appear in any element in the iterable."""
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS
    elif reserved_tokens is "assumed_in_file":
        reserved_tokens = []

    alphabet = {c for token in iterable for c in token}
    alphabet |= {c for token in reserved_tokens for c in token}
    alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.

    return alphabet
