"""Logit processors and a sampler for `generate`.

Pipeline order: repetition penalty -> temperature -> top-k -> top-p -> sample.
"""

import torch


def repetition_penalty(logits, histories, penalty):
    """Make every token already present in a row's history less likely.

    logits: (B, V). histories: one token list per row. penalty >= 1.
    A penalised token's logit is divided by `penalty` when positive and
    multiplied by it when negative, so its probability never goes up.
    """
    if penalty == 1.0:
        return logits
    out = logits.clone()
    for b, history in enumerate(histories):
        if not history:
            continue
        seen = torch.tensor(sorted(set(history)), dtype=torch.long, device=logits.device)
        picked = out[b, seen]
        out[b, seen] = picked / penalty
    return out


def top_k_filter(logits, k):
    """Keep the k highest logits per row; set the rest to -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits, top_p):
    """Nucleus filtering.

    Per row, keep the smallest set of highest-probability tokens whose total
    probability is at least `top_p`; set every other logit to -inf. The most
    likely token is always kept.
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_index = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    remove_sorted = mass_before >= top_p
    remove_sorted[..., 0] = False  # the most likely token survives even when top_p is 0
    return logits.masked_fill(remove_sorted, float("-inf"))


def make_sampler(*, temperature=1.0, top_k=0, top_p=1.0, penalty=1.0, generator=None):
    """Build a sampler callable for `generate`."""

    def sampler(logits, histories):
        logits = repetition_penalty(logits.float(), histories, penalty)
        logits = logits / temperature
        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    return sampler
