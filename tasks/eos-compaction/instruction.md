# One sequence ending changes what the others generate

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

## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
