# Streaming breaks accents, CJK and emoji

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
