# Long generations go wrong with the cache on

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

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
