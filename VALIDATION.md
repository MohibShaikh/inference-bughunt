# Grader validation

Produced by `uv run python validate.py`. Reward is 1 only when the task tests and the
regression tests both pass. Every row must behave as expected for the pack to be valid.

| Task | Variant | Task tests | Regression | Reward | As expected |
| --- | --- | --- | --- | --- | --- |
| cache-positions | buggy start | FAIL | pass | 0 | yes |
| cache-positions | reference fix | pass | pass | 1 | yes |
| cache-positions | wrong fix: disable-cache | FAIL | pass | 0 | yes |
| cache-positions | wrong fix: one-row-at-a-time | FAIL | pass | 0 | yes |
| cache-positions | wrong fix: import-the-reference | FAIL | FAIL | 0 | yes |
| cache-positions | wrong fix: find-the-reference | FAIL | FAIL | 0 | yes |
| cache-positions | wrong fix: find-the-reference-via-torch | FAIL | FAIL | 0 | yes |
| eos-compaction | buggy start | FAIL | pass | 0 | yes |
| eos-compaction | reference fix | pass | pass | 1 | yes |
| eos-compaction | wrong fix: disable-cache | FAIL | pass | 0 | yes |
| eos-compaction | wrong fix: one-row-at-a-time | FAIL | pass | 0 | yes |
| sliding-window | buggy start | FAIL | pass | 0 | yes |
| sliding-window | reference fix | pass | pass | 1 | yes |
| sliding-window | wrong fix: disable-cache | FAIL | pass | 0 | yes |
| half-precision | buggy start | FAIL | pass | 0 | yes |
| half-precision | reference fix | pass | pass | 1 | yes |
| half-precision | wrong fix: nan-to-num | FAIL | pass | 0 | yes |
| half-precision | wrong fix: minus-infinity-mask | FAIL | pass | 0 | yes |
| sampling-filters | buggy start | FAIL | pass | 0 | yes |
| sampling-filters | reference fix | pass | pass | 1 | yes |
| sampling-filters | wrong fix: only-top-p | FAIL | pass | 0 | yes |
| sampling-filters | wrong fix: only-penalty | FAIL | pass | 0 | yes |
| sampling-filters | wrong fix: top-p-off-by-one | FAIL | pass | 0 | yes |
| stream-utf8 | buggy start | FAIL | pass | 0 | yes |
| stream-utf8 | reference fix | pass | pass | 1 | yes |
| stream-utf8 | wrong fix: buffer-until-flush | FAIL | FAIL | 0 | yes |
| stream-utf8 | wrong fix: ignore-errors | FAIL | pass | 0 | yes |
| all-at-once | buggy start | FAIL | pass | 0 | yes |
| all-at-once | reference fix | pass | pass | 1 | yes |

Wrong fixes checked:

- `cache-positions` / `disable-cache`: Ignore the KV cache and recompute the whole sequence every step.
- `cache-positions` / `one-row-at-a-time`: Run every prompt as its own batch of one.
- `cache-positions` / `import-the-reference`: Re-export the grader's clean copy by its package name instead of fixing anything.
- `cache-positions` / `find-the-reference`: Search the interpreter's loaded modules for the grader's clean copy and call it.
- `cache-positions` / `find-the-reference-via-torch`: Reach the loaded modules through torch.sys, with no forbidden import, and call the clean copy.
- `eos-compaction` / `disable-cache`: Ignore the KV cache and recompute the whole sequence every step.
- `eos-compaction` / `one-row-at-a-time`: Run every prompt as its own batch of one.
- `sliding-window` / `disable-cache`: Ignore the KV cache and recompute the whole sequence every step.
- `half-precision` / `nan-to-num`: Replace NaN logits with zeros on the way out.
- `half-precision` / `minus-infinity-mask`: Mask with -inf, which still gives NaN on rows that are entirely padding.
- `sampling-filters` / `only-top-p`: Fix nucleus filtering but leave the repetition penalty broken.
- `sampling-filters` / `only-penalty`: Fix the repetition penalty but leave nucleus filtering broken.
- `sampling-filters` / `top-p-off-by-one`: Fix both, but drop the token that crosses the top_p threshold.
- `stream-utf8` / `buffer-until-flush`: Hold every token and decode the whole stream in flush().
- `stream-utf8` / `ignore-errors`: Use an incremental decoder that silently drops bad or truncated bytes.
