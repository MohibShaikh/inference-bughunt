import pytest
import torch
from common import NO_EOS, Recorder, app, expected, left_pad, make_prompts, model_pair

LENGTHS = [[3, 11, 6, 15], [1, 20], [7, 2, 2, 19, 12], [14, 13]]


@pytest.mark.parametrize("seed", range(4))
def test_padded_batch_matches_single_prompt(seed):
    a, r = model_pair(seed)
    prompts = make_prompts(seed, LENGTHS[seed])
    assert app.generate(a, prompts, 12, eos_id=NO_EOS, use_cache=True) == expected(r, prompts, 12)
    assert app.generate(a, prompts, 12, eos_id=NO_EOS, use_cache=False) == expected(r, prompts, 12)


@pytest.mark.parametrize("seed", range(4))
def test_cached_forward_matches_reference(seed):
    a, r = model_pair(seed)
    tokens, mask = left_pad(make_prompts(seed, LENGTHS[seed]), app.PAD, "cpu")
    _, past = a(tokens[:, :-2], attention_mask=mask[:, :-2])
    got, _ = a(tokens[:, -2:], attention_mask=mask, past=past)
    want, _ = r(tokens, attention_mask=mask)
    # Logits at padding positions are meaningless, so compare real tokens only.
    real = mask[:, -2:]
    assert torch.allclose(got[real], want[:, -2:][real], atol=1e-9)


def test_cache_and_batching_still_used():
    a, _ = model_pair(0)
    prompts = make_prompts(0, [3, 11, 6, 15])
    recorder = Recorder(a)
    app.generate(a, prompts, 10, eos_id=NO_EOS, use_cache=True)
    assert recorder.calls[0] == (4, 15)
    assert recorder.calls[1:] == [(4, 1)] * 9
