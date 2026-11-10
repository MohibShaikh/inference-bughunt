import pytest
from common import NO_EOS, Recorder, app, expected, make_prompts, model_pair

WINDOW = 48
LENGTHS = [[40], [47, 10], [48, 30, 44], [70, 5], [20, 60, 46, 3]]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("use_cache", [True, False])
def test_long_generation_matches_reference(seed, use_cache):
    a, r = model_pair(seed)
    assert a.cfg.max_seq_len == WINDOW
    prompts = make_prompts(seed, LENGTHS[seed])
    got = app.generate(a, prompts, 24, eos_id=NO_EOS, use_cache=use_cache)
    assert got == expected(r, prompts, 24)


def test_small_window():
    a, r = model_pair(7, max_seq_len=8)
    prompts = make_prompts(7, [3, 8, 12])
    assert app.generate(a, prompts, 20, eos_id=NO_EOS, use_cache=True) == expected(r, prompts, 20)


def test_cache_used_while_rows_fit():
    a, _ = model_pair(0)
    recorder = Recorder(a)
    app.generate(a, make_prompts(0, [10, 20]), 10, eos_id=NO_EOS, use_cache=True)
    assert recorder.calls[0] == (2, 20)
    assert recorder.calls[1:] == [(2, 1)] * 9


def test_cache_used_until_the_window_is_full():
    a, _ = model_pair(0)
    recorder = Recorder(a)
    app.generate(a, make_prompts(0, [44, 10]), 10, eos_id=NO_EOS, use_cache=True)
    # The longest row is 44, 45, 46, 47, 48 tokens long on the first five steps: all fit.
    assert recorder.calls[0] == (2, 44)
    assert recorder.calls[1:5] == [(2, 1)] * 4
    assert len(recorder.calls) == 10
