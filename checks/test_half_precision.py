import pytest
import torch
from common import NO_EOS, app, left_pad, make_prompts, model_pair

LENGTHS = [[3, 11, 6, 15], [1, 20], [7, 2, 2, 19, 12], [30, 4]]
# float16 forward passes land within about 2e-3 of float64 on this model.
HALF_TOLERANCE = 0.03


def padded(seed):
    return left_pad(make_prompts(seed, LENGTHS[seed]), app.PAD, "cpu")


@pytest.mark.parametrize("seed", range(4))
def test_half_padded_batch(seed):
    a, r = model_pair(seed)
    a = a.half()
    tokens, mask = padded(seed)
    got, _ = a(tokens, attention_mask=mask)
    want, _ = r(tokens, attention_mask=mask)
    assert got.dtype == torch.float16
    assert torch.isfinite(got[mask]).all()
    assert (got[mask].double() - want[mask]).abs().max() < HALF_TOLERANCE


@pytest.mark.parametrize("seed", range(4))
def test_half_with_cache(seed):
    a, r = model_pair(seed)
    a = a.half()
    tokens, mask = padded(seed)
    _, past = a(tokens[:, :-1], attention_mask=mask[:, :-1])
    got, _ = a(tokens[:, -1:], attention_mask=mask, past=past)
    want, _ = r(tokens, attention_mask=mask)
    assert got.dtype == torch.float16
    assert torch.isfinite(got).all()
    assert (got[:, -1].double() - want[:, -1]).abs().max() < HALF_TOLERANCE


@pytest.mark.parametrize("seed", range(4))
def test_half_unpadded_batch(seed):
    a, r = model_pair(seed)
    a = a.half()
    tokens = torch.tensor(make_prompts(seed, [12, 12, 12]))
    got, _ = a(tokens)
    want, _ = r(tokens)
    assert (got.double() - want).abs().max() < HALF_TOLERANCE


def test_half_generate_runs():
    a, _ = model_pair(0)
    a = a.half()
    out = app.generate(a, make_prompts(0, [3, 11, 6, 15]), 8, eos_id=NO_EOS)
    assert [len(o) for o in out] == [8, 8, 8, 8]


@pytest.mark.parametrize("seed", range(2))
def test_float32_unchanged(seed):
    a, r = model_pair(seed)
    a = a.float()
    tokens, mask = padded(seed)
    got, _ = a(tokens, attention_mask=mask)
    want, _ = r(tokens, attention_mask=mask)
    assert (got[mask].double() - want[mask]).abs().max() < 1e-4
