"""Batched autoregressive generation with a KV cache."""

import torch

from .tokenizer import EOS, PAD


def _left_pad(rows, pad_id, device):
    width = max(len(r) for r in rows)
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for b, row in enumerate(rows):
        tokens[b, width - len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[b, width - len(row):] = True
    return tokens, mask


@torch.no_grad()
def generate(model, prompts, max_new_tokens, *, eos_id=EOS, pad_id=PAD, use_cache=True, sampler=None):
    """Generate a continuation for every prompt in the batch.

    prompts: list of non-empty token id lists; lengths may differ.
    sampler: callable (logits (B, V), histories: list[list[int]]) -> LongTensor (B,).
        Defaults to greedy argmax.

    Returns one list of new tokens per prompt. A row stops when it produces
    `eos_id`; the EOS token itself is not included in the output.

    Behaviour the callers rely on:
      * A row's output does not depend on which other rows share its batch.
      * `use_cache=True` and `use_cache=False` return identical tokens.
      * The model always sees at most the last `max_seq_len` tokens of a row,
        positioned from 0 (a sliding window).
      * With the cache on, each step feeds only the newest token per row for
        as long as every row still fits in the window, and rows that have
        finished are dropped from the batch.
    """
    device = next(model.parameters()).device
    window = model.cfg.max_seq_len
    seqs = [list(p) for p in prompts]
    outputs = [[] for _ in prompts]
    active = list(range(len(prompts)))
    past, mask = None, None

    for _ in range(max_new_tokens):
        if not active:
            break
        overflow = max(len(seqs[i]) for i in active) > window
        if use_cache and past is not None and not overflow:
            tokens = torch.tensor([[seqs[i][-1]] for i in active], dtype=torch.long, device=device)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            logits, past = model(tokens, attention_mask=mask, past=past)
        else:
            # Cached keys embed absolute positions, so they cannot survive the
            # window sliding: once any row overflows, recompute from scratch.
            tokens, mask = _left_pad([seqs[i][-window:] for i in active], pad_id, device)
            logits, present = model(tokens, attention_mask=mask)
            past = present if use_cache and not overflow else None

        last = logits[:, -1]
        if sampler is None:
            next_tokens = last.argmax(dim=-1)
        else:
            next_tokens = sampler(last, [seqs[i] for i in active])

        keep = []
        for b, i in enumerate(active):
            token = int(next_tokens[b])
            if token == eos_id:
                continue
            outputs[i].append(token)
            seqs[i].append(token)
            keep.append(b)

        if len(keep) < len(active):
            index = torch.tensor(keep, dtype=torch.long, device=device)
            active = [active[b] for b in keep]
            if past is not None:
                past = [(k[index], v[index]) for k, v in past]
                mask = mask[index]
                # Columns that are padding for every surviving row are dead weight;
                # drop them so the cache never grows wider than the longest row.
                dead = int((~mask.any(dim=0)).long().cumprod(dim=0).sum())
                if dead:
                    past = [(k[:, :, dead:], v[:, :, dead:]) for k, v in past]
                    mask = mask[:, dead:]

    return outputs
