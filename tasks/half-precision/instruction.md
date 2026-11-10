# NaN logits in half precision

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

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
