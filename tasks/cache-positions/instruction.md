# Batched generation disagrees with single-prompt generation

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

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
