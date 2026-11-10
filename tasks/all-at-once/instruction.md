# Six open bug reports

Six separate reports came in against `tinyinfer`. They are reproduced below. Each has its own cause. Fix all of them.

## Report 1: Batched generation disagrees with single-prompt generation

Users of `tinyinfer.generate` report that when they send several prompts of
different lengths in one batch, some rows come back with different tokens than
the same prompt produces when it is sent alone. Greedy decoding is
deterministic, so this should never happen.

What we know:

- It only happens with `use_cache=True`.
- `use_cache=False` returns the right tokens for the same batch.
- Batches where every prompt has the same length are fine.

Find and fix the bug.

Batching and the KV cache must both stay in use. With the cache on, every step
after the first must still feed the model only the newest token of each row,
and all prompts must still go through the model as one batch.

## Report 2: One sequence ending changes what the others generate

When one sequence in a batch reaches EOS before the others, some of the
remaining sequences sometimes continue with different tokens than they produce
when run alone. Greedy decoding is deterministic, so a row's output must not
depend on its batch-mates.

What we know:

- It needs prompts of different lengths in the same batch.
- It needs `use_cache=True`.
- Until the first row finishes, every row is correct.

Find and fix the bug.

The existing optimisations must stay. With the cache on, every step after the
first must still feed the model only the newest token of each row, and a row
that has produced EOS must be dropped from the batch before the next step.

## Report 3: Long generations go wrong with the cache on

For long generations, `generate(..., use_cache=True)` and
`generate(..., use_cache=False)` return different tokens. They agree for a
while and then diverge. The divergence starts once a row's total length
(prompt plus generated tokens) passes `model.cfg.max_seq_len`.

The `use_cache=False` output is the correct one. It defines the sliding-window
behaviour: the model sees the last `max_seq_len` tokens of a row, positioned
from 0.

Find and fix the bug so both modes return identical tokens for any length.

The cache must stay useful. While every row in the batch still fits inside the
window, every step after the first must feed the model only the newest token
of each row.

## Report 4: NaN logits in half precision

We serve `TinyLM` in float16 to save memory. Some requests come back as NaN
logits, which the sampler then turns into garbage tokens. The same requests
are fine in float32.

What we know:

- It was reported from GPU serving, and it reproduces on CPU with `model.half()`.
- Not every batch is affected. Plenty of float16 requests are fine.

Find and fix the bug.

The model must keep running in float16: `model.half()` must return float16
logits that match the float32 logits to within normal half-precision error,
for every valid input, with and without the KV cache.

## Report 5: Two complaints about sampling quality

Two separate reports came in about `tinyinfer.sampling`:

1. With `top_p` set (for example 0.9), generations contain tokens the model
   considers very unlikely, and the most likely token is sometimes never
   sampled at all.
2. With a repetition penalty above 1, some tokens that were already used
   become more likely to be repeated, not less.

Find and fix whatever is causing these. The docstrings in
`tinyinfer/sampling.py` define the intended behaviour of each function.

## Report 6: Streaming breaks accents, CJK and emoji

Our streaming endpoint uses `tinyinfer.StreamDecoder` to turn tokens into text
as they arrive. Users see the replacement character (U+FFFD) wherever the text
has an accent, a CJK character or an emoji. Decoding the same tokens in one go
with `ByteTokenizer().decode(tokens)` gives the right text.

Find and fix the bug.

Streaming must stay streaming: each character has to be returned by the `push`
call that completes it. Holding text back until `flush` is not acceptable. For
any token stream, valid or not, joining every `push` result and the `flush`
result must equal `ByteTokenizer().decode(tokens)`.

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
