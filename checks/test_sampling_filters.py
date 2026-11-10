import pytest
import torch
from common import app, ref

V = 259
NEG_INF = float("-inf")


def random_logits(seed, rows=8, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, V, generator=g, dtype=torch.float64) * scale


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_keeps_the_nucleus(seed, top_p):
    logits = random_logits(seed)
    got = app.top_p_filter(logits.clone(), top_p)
    kept = got != NEG_INF
    probs = logits.softmax(dim=-1)

    # Kept logits are untouched and the most likely token always survives.
    assert torch.equal(got[kept], logits[kept])
    assert kept.gather(-1, logits.argmax(dim=-1, keepdim=True)).all()

    for row in range(logits.shape[0]):
        inside, outside = probs[row][kept[row]], probs[row][~kept[row]]
        # The kept set is the top of the distribution...
        assert outside.numel() == 0 or inside.min() >= outside.max()
        # ...it holds at least top_p of the mass...
        assert inside.sum() >= top_p - 1e-12
        # ...and it is minimal: dropping its least likely member falls below top_p.
        assert inside.sum() - inside.min() < top_p

    assert torch.equal(kept, ref.top_p_filter(logits.clone(), top_p) != NEG_INF)


@pytest.mark.parametrize("top_p", [0.0, 1e-9])
def test_top_p_tiny_keeps_only_the_best_token(top_p):
    logits = random_logits(11)
    kept = app.top_p_filter(logits.clone(), top_p) != NEG_INF
    assert (kept.sum(dim=-1) == 1).all()
    assert kept.gather(-1, logits.argmax(dim=-1, keepdim=True)).all()


def test_top_p_one_is_a_no_op():
    logits = random_logits(12)
    assert torch.equal(app.top_p_filter(logits.clone(), 1.0), logits)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("penalty", [1.1, 1.5, 3.0])
def test_repetition_penalty(seed, penalty):
    logits = random_logits(seed, rows=4)
    g = torch.Generator().manual_seed(seed + 100)
    histories = [torch.randint(0, V, (n,), generator=g).tolist() for n in (0, 1, 12, 40)]
    got = app.repetition_penalty(logits.clone(), histories, penalty)
    assert torch.allclose(got, ref.repetition_penalty(logits.clone(), histories, penalty))
    for row, history in enumerate(histories):
        seen = sorted(set(history))
        untouched = [t for t in range(V) if t not in set(seen)]
        assert torch.equal(got[row, untouched], logits[row, untouched])
        # A seen token's logit never goes up, whatever its sign.
        assert (got[row, seen] <= logits[row, seen]).all()
        assert (got[row, seen] < logits[row, seen]).all() or not seen


def test_repetition_penalty_does_not_modify_its_input():
    logits = random_logits(5, rows=2)
    before = logits.clone()
    app.repetition_penalty(logits, [[1, 2, 3], [4]], 1.3)
    assert torch.equal(logits, before)


def test_sampler_draws_stay_inside_the_nucleus():
    logits = random_logits(21, rows=1).float()
    allowed = ref.top_p_filter(logits.clone(), 0.5) != NEG_INF
    sampler = app.make_sampler(top_p=0.5, generator=torch.Generator().manual_seed(0))
    drawn = {int(sampler(logits, [[]])) for _ in range(300)}
    assert all(allowed[0, t] for t in drawn)
    assert len(drawn) > 1 or int(allowed.sum()) == 1


def test_sampler_penalty_never_favours_a_seen_token():
    # Token 7 has a negative logit and is in the history, so the penalty must push it down.
    rows = 2000
    logits = torch.full((rows, V), -3.0)
    logits[:, 7] = -2.0
    sampler = app.make_sampler(penalty=4.0, generator=torch.Generator().manual_seed(0))
    hits = int((sampler(logits, [[7]] * rows) == 7).sum())
    # Correct penalty: logit -8, about 0.05 expected hits. Dividing instead: logit -0.5,
    # about 90 expected hits.
    assert hits <= 5
