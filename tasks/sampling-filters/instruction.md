# Two complaints about sampling quality

Two separate reports came in about `tinyinfer.sampling`:

1. With `top_p` set (for example 0.9), generations contain tokens the model
   considers very unlikely, and the most likely token is sometimes never
   sampled at all.
2. With a repetition penalty above 1, some tokens that were already used
   become more likely to be repeated, not less.

Find and fix whatever is causing these. The docstrings in
`tinyinfer/sampling.py` define the intended behaviour of each function.

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
