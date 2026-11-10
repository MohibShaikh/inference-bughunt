import random

import pytest
from common import app, ref

SAMPLES = [
    "héllo wörld",
    "日本語のテキスト",
    "emoji \U0001f642\U0001f680 mixed ñ text",
    "Ελληνικά",
    "á combining mark",
    "\U0001f468‍\U0001f469‍\U0001f467 family",
    "",
]


def random_text(seed, n=60):
    rng = random.Random(seed)
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x400, 0x4FF), (0x4E00, 0x4FFF), (0x1F600, 0x1F64F)]
    return "".join(chr(rng.randint(*rng.choice(ranges))) for _ in range(n))


def stream(decoder, tokens):
    return [decoder.push(t) for t in tokens]


@pytest.mark.parametrize("text", SAMPLES + [random_text(s) for s in range(6)])
def test_valid_text_streams_character_by_character(text):
    tokens = app.ByteTokenizer().encode(text)
    decoder = app.StreamDecoder()
    pieces = stream(decoder, tokens)
    tail = decoder.flush()
    assert "".join(pieces) + tail == text
    # Each character comes out of the push that completes it, never later.
    assert pieces == stream(ref.StreamDecoder(), tokens)
    assert tail == ""


def test_special_tokens_produce_no_text():
    tokens = [app.BOS, 0xE6, 0x97, app.PAD, 0xA5, app.EOS]
    decoder = app.StreamDecoder()
    assert stream(decoder, tokens) == ["", "", "", "", "日", ""]
    assert decoder.flush() == ""


def test_stream_cut_off_mid_character():
    tokens = list(b"ab") + list("\U0001f642".encode())[:2]
    decoder = app.StreamDecoder()
    pieces = stream(decoder, tokens)
    assert pieces == ["a", "b", "", ""]
    assert decoder.flush() == "�"


@pytest.mark.parametrize("seed", range(8))
def test_arbitrary_bytes_match_one_shot_decode(seed):
    rng = random.Random(seed)
    tokens = [rng.randint(0, 258) for _ in range(200)]
    decoder = app.StreamDecoder()
    text = "".join(stream(decoder, tokens)) + decoder.flush()
    assert text == ref.ByteTokenizer().decode(tokens)
    assert text == app.ByteTokenizer().decode(tokens)


def test_decoders_are_independent():
    first, second = app.StreamDecoder(), app.StreamDecoder()
    emoji = list("\U0001f680".encode())
    assert stream(first, emoji[:2]) == ["", ""]
    assert stream(second, list(b"ok")) == ["o", "k"]
    assert stream(first, emoji[2:]) == ["", "\U0001f680"]


def test_decoder_can_be_reused_after_flush():
    decoder = app.StreamDecoder()
    stream(decoder, list("é".encode())[:1])
    assert decoder.flush() == "�"
    assert stream(decoder, list("é".encode())) == ["", "é"]
