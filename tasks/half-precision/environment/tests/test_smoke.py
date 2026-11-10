"""Smoke test shipped to the agent. It passes before the fix and must pass after it."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer import BOS, ByteTokenizer, Config, StreamDecoder, TinyLM, generate, make_sampler  # noqa: E402


def make_model():
    torch.manual_seed(0)
    return TinyLM(Config()).double().eval()


def test_forward_shapes():
    model = make_model()
    logits, cache = model(torch.randint(0, 256, (2, 7)))
    assert logits.shape == (2, 7, 259)
    assert len(cache) == model.cfg.n_layer


def test_cache_matches_no_cache_for_one_short_prompt():
    model = make_model()
    prompt = [BOS, 104, 105]
    assert generate(model, [prompt], 10, eos_id=-1) == generate(model, [prompt], 10, eos_id=-1, use_cache=False)


def test_sampler_runs():
    model = make_model()
    sampler = make_sampler(temperature=0.8, top_k=20, generator=torch.Generator().manual_seed(0))
    out = generate(model, [[BOS, 1], [BOS, 2]], 5, eos_id=-1, sampler=sampler)
    assert [len(o) for o in out] == [5, 5]


def test_ascii_streaming():
    decoder = StreamDecoder()
    text = "".join(decoder.push(t) for t in ByteTokenizer().encode("hello")) + decoder.flush()
    assert text == "hello"
