# inference-bughunt

Seven debugging tasks for coding agents, set in a small LLM inference stack.
Each task plants realistic serving bugs and ships a grader that rejects the
shortcut fixes an agent is tempted to make, such as switching off the feature
that was broken.

The tasks are in [Harbor](https://docs.harborframework.com) format and run on
CPU. No GPU or model weights are needed.

## The tasks

The stack under test is `tinyinfer`: a byte-level transformer with a KV cache,
batched generation with left padding, a sampler, and a streaming detokenizer.
It is about 300 lines of PyTorch.

| Task | What the user reports | Shortcut the grader rejects |
| --- | --- | --- |
| `cache-positions` | Shorter prompts in a batch get different tokens than when sent alone, only with the cache on. | Disabling the cache. Running one prompt at a time. |
| `eos-compaction` | When one sequence hits EOS, the others start producing different tokens. | Disabling the cache. Running one prompt at a time. |
| `sliding-window` | Cached and uncached generation diverge once a row outgrows the context window. | Disabling the cache. |
| `half-precision` | float16 inference returns NaN for some batches; float32 is fine. | `nan_to_num` on the output. Masking with `-inf`. |
| `sampling-filters` | top-p samples unlikely tokens; the repetition penalty rewards repetition. | Fixing one of the two bugs. An off-by-one nucleus. |
| `stream-utf8` | Streaming shows U+FFFD for accents, CJK and emoji. | Holding all text until the stream ends. Dropping bad bytes. |
| `all-at-once` | All six reports above, with every bug planted in the same copy of the stack. | Graded by all six test suites. |

Instructions describe the symptom the way a bug report would. They do not name
the file or the cause.

## How the graders resist shortcuts

- **Hidden tests with a hidden reference.** The tests and a clean copy of the
  stack are added only after the agent finishes. Expected outputs come from the
  clean copy, run one prompt at a time with the cache off.
- **A separate grading container.** Harbor copies only `/app/tinyinfer` out of
  the agent's container into a fresh one. Anything else the agent changed
  (installed packages, test files, the environment) never reaches the grader.
- **The reference cannot be imported.** The clean copy is loaded from a file
  path under a random module name and is never on the import path. Submissions
  may import only `torch` and a short list of standard library modules, and may
  not use `eval`, `exec`, `open` or `__import__`, so they cannot go looking for
  it either. The instructions state this rule.
- **The optimisation must survive.** The cache and batching tasks record the
  shape of every forward call. A fix that recomputes the whole sequence each
  step, or stops batching, returns correct tokens and still scores 0. Each
  instruction states this requirement, so it is not a surprise.
- **Exact comparisons in float64.** Token-level checks run the model in double
  precision, so a correct fix cannot fail on rounding noise and results are the
  same on any machine.
- **Regression tests on every task.** Behaviour unrelated to the bug is checked
  too, so a fix cannot break something else to pass.

## Validation

`validate.py` grades 29 variants without Docker: for each task, the buggy
starting state, the reference fix, and every known wrong fix. The full table is
in [VALIDATION.md](VALIDATION.md).

- Every buggy start fails its task tests and passes the regression tests.
- Every reference fix scores 1.
- All 15 wrong fixes score 0, including three that try to steal the grader's
  reference copy.

In Harbor with Docker, the oracle agent scores 7/7 and the do-nothing agent
scores 0/7.

## Results

Four models were run once each on `all-at-once`, the hardest task. Every model
fixed all six bugs.

| Model | Harness | Effort | Reports fixed | Regression |
| --- | --- | --- | --- | --- |
| GPT-6.1 Sol | Codex CLI | low | 6/6 | pass |
| GPT-6 Astra | Codex CLI | medium | 6/6 | pass |
| DeepSeek V4.1 Flash | OpenCode | high | 6/6 | pass |
| DeepSeek V4 Pro | OpenCode | high | 6/6 | pass |

**What this means: the pack is saturated.** It does not separate these models,
so in its current form it is a grader-design sample and not a benchmark.
Harder versions would need bugs that interact, a larger codebase, or reports
with less diagnostic detail.

How the runs were done:

- Each model worked in its own fresh checkout containing only the buggy stack,
  the smoke test and a short README. The checkout was outside this repository,
  so solutions, hidden tests and bug definitions were not reachable from it.
- The prompt was the task instruction and nothing else, with `/app` paths
  changed to the checkout and the path of a Python interpreter added.
- Runs were on a Windows host, not inside the Harbor container, and each model
  was run once. Treat the numbers as a first reading.
- Each run's log was searched afterwards for paths outside its own checkout.
  None were found.
- The import rule was added to the instructions after these runs. All four
  submissions pass it.

## Running it

```bash
uv sync                      # CPU PyTorch and pytest
uv run python build.py       # regenerate tasks/ from stack/, bugs.py and checks/
uv run python validate.py    # grade all 29 variants, rewrite VALIDATION.md

uv tool install harbor
harbor run -p tasks -a oracle                                    # expect 7/7
harbor run -p tasks -a nop                                       # expect 0/7
harbor run -p tasks -a claude-code -m anthropic/claude-sonnet-5-5   # a real agent
```

On Windows, set `PYTHONUTF8=1` before running `harbor`.

## Layout

- `stack/tinyinfer/`: the clean stack. It is the reference solution for every task.
- `bugs.py`: each task's planted bug, instruction and known wrong fixes, as text edits.
- `checks/`: the hidden tests, plus the smoke test the agent is given.
- `build.py`, `validate.py`: generate the tasks and check the graders.
- `tasks/`: the generated Harbor tasks.

## Limits

- The model has random weights. The bugs are about serving logic, not model quality.
- One bug site per single task (two for `sampling-filters`). Real incidents are messier.
- The wrong-fix list is what I thought of. A shortcut outside it may still pass.
- The reference runs in the same Python process as the submission. The import
  rule blocks the obvious ways to reach it, but a static check is not a sandbox.
