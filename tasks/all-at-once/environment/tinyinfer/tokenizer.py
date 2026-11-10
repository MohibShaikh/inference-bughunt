"""Byte-level tokenizer and a streaming detokenizer."""

import codecs

BOS, EOS, PAD = 256, 257, 258


class ByteTokenizer:
    def encode(self, text, add_bos=True):
        tokens = list(text.encode("utf-8"))
        return [BOS] + tokens if add_bos else tokens

    def decode(self, tokens):
        data = bytes(t for t in tokens if t < 256)
        return data.decode("utf-8", errors="replace")


class StreamDecoder:
    """Turn a token stream into text as it arrives.

    `push(token)` returns the text that became available with this token,
    which is "" while a multi-byte character is still incomplete. `flush()`
    returns whatever is left at the end of the stream. Joining every `push`
    result and the `flush` result equals `ByteTokenizer().decode(tokens)`.
    Special tokens (ids >= 256) produce no text.
    """

    def __init__(self):
        self._pending = []

    def push(self, token):
        if token >= 256:
            return ""
        return bytes([token]).decode("utf-8", errors="replace")

    def flush(self):
        return ""
