import pytest
from common import Recorder, app, expected, make_prompts, model_pair

MAX_NEW = 14
# (seed, prompt lengths, row that finishes first, step at which it emits EOS)
CASES = [
    (0, [3, 11, 6, 15], 0, 2),
    (1, [3, 11, 6, 15], 2, 4),
    (2, [9, 2, 17], 1, 1),
    (3, [5, 12, 8, 20, 4], 0, 3),
    (4, [16, 4, 10], 1, 6),
]


def eos_case(ref_model, prompts, row, at):
    """Pick an EOS id that makes `row` finish at step `at`, and the outputs that follow."""
    streams = expected(ref_model, prompts, MAX_NEW)
    eos = streams[row][at]
    want = [s[: s.index(eos)] if eos in s else s for s in streams]
    return eos, want


@pytest.mark.parametrize("seed,lengths,row,at", CASES)
def test_early_finish_leaves_other_rows_alone(seed, lengths, row, at):
    a, r = model_pair(seed)
    prompts = make_prompts(seed, lengths)
    eos, want = eos_case(r, prompts, row, at)
    assert len(want[row]) <= at
    assert app.generate(a, prompts, MAX_NEW, eos_id=eos, use_cache=True) == want
    assert app.generate(a, prompts, MAX_NEW, eos_id=eos, use_cache=False) == want


@pytest.mark.parametrize("seed", range(3))
def test_longest_row_finishing_first_in_a_full_window(seed):
    # The long row fills the window and ends at once; the short row must carry on cached.
    a, r = model_pair(seed, max_seq_len=8)
    prompts = make_prompts(seed, [8, 2])
    streams = expected(r, prompts, 5)
    eos = streams[0][0]
    want = [s[: s.index(eos)] if eos in s else s for s in streams]
    assert app.generate(a, prompts, 5, eos_id=eos, use_cache=True) == want


@pytest.mark.parametrize("seed,lengths,row,at", CASES)
def test_finished_rows_leave_the_batch(seed, lengths, row, at):
    a, r = model_pair(seed)
    prompts = make_prompts(seed, lengths)
    eos, want = eos_case(r, prompts, row, at)
    # A row takes part in one forward call per token it emits, plus the call that emits its EOS.
    steps = [len(w) + 1 if len(w) < MAX_NEW else MAX_NEW for w in want]
    batch_sizes = [sum(s > t for s in steps) for t in range(max(steps))]

    recorder = Recorder(a)
    app.generate(a, prompts, MAX_NEW, eos_id=eos, use_cache=True)
    assert [b for b, _ in recorder.calls] == batch_sizes
    assert all(t == 1 for _, t in recorder.calls[1:])
