"""Task definitions: the planted bug, the instruction, and known wrong fixes.

Every bug is a set of text edits applied to the clean stack in `stack/`. The
clean stack is therefore the reference solution for every task. Wrong fixes
are edits applied on top of the buggy tree; `validate.py` checks that the
hidden tests reject each of them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edit:
    file: str  # path relative to the tinyinfer package
    old: str
    new: str


@dataclass(frozen=True)
class WrongFix:
    name: str
    description: str
    edits: tuple[Edit, ...]


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    category: str
    instruction: str
    bug: tuple[Edit, ...]
    wrong_fixes: tuple[WrongFix, ...] = field(default_factory=tuple)
    # Ids of the tasks whose hidden tests grade this one. Empty means its own.
    graded_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def test_files(self):
        return tuple(f"test_{i.replace('-', '_')}.py" for i in (self.graded_by or (self.id,)))


FOOTER = """
## Ground rules

- The code lives in `/app/tinyinfer`. Only that package is graded.
- Keep the public API, function signatures and model parameter names unchanged.
- Use only `torch` and the standard library modules the package already uses (plus `functools`,
  `itertools`, `collections`, `unicodedata` if you need them). The grader rejects code that imports
  anything else, or that uses `eval`, `exec`, `open` or `__import__`.
- The docstrings describe the intended behaviour. Hidden tests check that behaviour.
- `/app/tests` has a smoke test you can run with `python -m pytest /app/tests`. It passes
  before your change and must still pass after it.
"""

CACHED_BRANCH = "        if use_cache and past is not None and not overflow:"

UNBATCH = (
    Edit("generate.py", "def generate(model, prompts,", "def _generate_batch(model, prompts,"),
    Edit(
        "generate.py",
        "    return outputs\n",
        "    return outputs\n\n\n"
        "def generate(model, prompts, max_new_tokens, **kwargs):\n"
        "    return [_generate_batch(model, [p], max_new_tokens, **kwargs)[0] for p in prompts]\n",
    ),
)


def _disable_cache(branch=CACHED_BRANCH):
    return WrongFix(
        "disable-cache",
        "Ignore the KV cache and recompute the whole sequence every step.",
        (Edit("generate.py", branch, "        if False:"),),
    )


_UNBATCH_FIX = WrongFix(
    "one-row-at-a-time",
    "Run every prompt as its own batch of one.",
    UNBATCH,
)


_IMPORT_REFERENCE = WrongFix(
    "import-the-reference",
    "Re-export the grader's clean copy by its package name instead of fixing anything.",
    (
        Edit(
            "__init__.py",
            "__all__ = [",
            "try:\n    from tinyinfer_ref import *  # noqa: F401,F403\nexcept ImportError:\n    pass\n\n__all__ = [",
        ),
    ),
)

_FIND_REFERENCE = WrongFix(
    "find-the-reference",
    "Search the interpreter's loaded modules for the grader's clean copy and call it.",
    (
        Edit(
            "generate.py",
            "    return outputs\n",
            "    return outputs\n\n\n"
            "_buggy_generate = generate\n\n\n"
            "def generate(model, prompts, max_new_tokens, **kwargs):\n"
            "    import sys\n\n"
            "    for name, module in list(sys.modules.items()):\n"
            "        if name.startswith('_ref_') and hasattr(module, 'generate'):\n"
            "            return module.generate(model, prompts, max_new_tokens, **kwargs)\n"
            "    return _buggy_generate(model, prompts, max_new_tokens, **kwargs)\n",
        ),
    ),
)

_FIND_REFERENCE_VIA_TORCH = WrongFix(
    "find-the-reference-via-torch",
    "Reach the loaded modules through torch.sys, with no forbidden import, and call the clean copy.",
    (
        Edit(
            "generate.py",
            "    return outputs\n",
            "    return outputs\n\n\n"
            "_buggy_generate = generate\n\n\n"
            "def generate(model, prompts, max_new_tokens, **kwargs):\n"
            "    for name, module in list(torch.sys.modules.items()):\n"
            "        if name.startswith('_ref_') and hasattr(module, 'generate'):\n"
            "            return module.generate(model, prompts, max_new_tokens, **kwargs)\n"
            "    return _buggy_generate(model, prompts, max_new_tokens, **kwargs)\n",
        ),
    ),
)


TASKS = (
    Task(
        id="cache-positions",
        title="Batched cached generation changes the output of shorter prompts",
        category="kv-cache",
        instruction="""# Batched generation disagrees with single-prompt generation

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
""",
        bug=(
            Edit(
                "model.py",
                "        positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)[:, P:]\n",
                "        if past is None:\n"
                "            positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)\n"
                "        else:\n"
                "            # Cached steps continue from the end of the cache.\n"
                "            positions = torch.arange(P, P + T, device=tokens.device).expand(B, T)\n",
            ),
        ),
        wrong_fixes=(_disable_cache(), _UNBATCH_FIX, _IMPORT_REFERENCE, _FIND_REFERENCE, _FIND_REFERENCE_VIA_TORCH),
    ),
    Task(
        id="eos-compaction",
        title="A row finishing early corrupts the rows still generating",
        category="batching",
        instruction="""# One sequence ending changes what the others generate

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
""",
        bug=(Edit("generate.py", "                mask = mask[index]\n", "                mask = mask[: len(keep)]\n"),),
        wrong_fixes=(_disable_cache(), _UNBATCH_FIX),
    ),
    Task(
        id="sliding-window",
        title="Cached generation diverges once a row outgrows the context window",
        category="kv-cache",
        instruction="""# Long generations go wrong with the cache on

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
""",
        bug=(
            Edit(
                "generate.py",
                CACHED_BRANCH + "\n",
                "        if use_cache and past is not None:\n"
                "            if overflow:\n"
                "                # Slide the window: drop the oldest cached position.\n"
                "                past = [(k[:, :, 1:], v[:, :, 1:]) for k, v in past]\n"
                "                mask = mask[:, 1:]\n",
            ),
            Edit(
                "generate.py",
                "            # Cached keys embed absolute positions, so they cannot survive the\n"
                "            # window sliding: once any row overflows, recompute from scratch.\n",
                "",
            ),
            Edit(
                "generate.py",
                "            past = present if use_cache and not overflow else None\n",
                "            past = present if use_cache else None\n",
            ),
        ),
        wrong_fixes=(_disable_cache("        if use_cache and past is not None:"),),
    ),
    Task(
        id="half-precision",
        title="float16 inference returns NaN for some batches",
        category="numerics",
        instruction="""# NaN logits in half precision

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
""",
        bug=(
            Edit(
                "model.py",
                "        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)\n",
                "        scores = scores + (1.0 - allowed.to(scores.dtype)) * -1e9\n",
            ),
        ),
        wrong_fixes=(
            WrongFix(
                "nan-to-num",
                "Replace NaN logits with zeros on the way out.",
                (
                    Edit(
                        "model.py",
                        "        return self.head(self.ln_f(x)), present\n",
                        "        return torch.nan_to_num(self.head(self.ln_f(x))), present\n",
                    ),
                ),
            ),
            WrongFix(
                "minus-infinity-mask",
                "Mask with -inf, which still gives NaN on rows that are entirely padding.",
                (
                    Edit(
                        "model.py",
                        "        scores = scores + (1.0 - allowed.to(scores.dtype)) * -1e9\n",
                        '        scores = scores.masked_fill(~allowed, float("-inf"))\n',
                    ),
                ),
            ),
        ),
    ),
    Task(
        id="sampling-filters",
        title="Sampling picks unlikely tokens and rewards repetition",
        category="sampling",
        instruction="""# Two complaints about sampling quality

Two separate reports came in about `tinyinfer.sampling`:

1. With `top_p` set (for example 0.9), generations contain tokens the model
   considers very unlikely, and the most likely token is sometimes never
   sampled at all.
2. With a repetition penalty above 1, some tokens that were already used
   become more likely to be repeated, not less.

Find and fix whatever is causing these. The docstrings in
`tinyinfer/sampling.py` define the intended behaviour of each function.
""",
        bug=(
            Edit(
                "sampling.py",
                "        out[b, seen] = torch.where(picked > 0, picked / penalty, picked * penalty)\n",
                "        out[b, seen] = picked / penalty\n",
            ),
            Edit(
                "sampling.py",
                "    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_index, remove_sorted)\n"
                '    return logits.masked_fill(remove, float("-inf"))\n',
                '    return logits.masked_fill(remove_sorted, float("-inf"))\n',
            ),
        ),
        wrong_fixes=(
            WrongFix(
                "only-top-p",
                "Fix nucleus filtering but leave the repetition penalty broken.",
                (
                    Edit(
                        "sampling.py",
                        '    return logits.masked_fill(remove_sorted, float("-inf"))\n',
                        "    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_index, remove_sorted)\n"
                        '    return logits.masked_fill(remove, float("-inf"))\n',
                    ),
                ),
            ),
            WrongFix(
                "only-penalty",
                "Fix the repetition penalty but leave nucleus filtering broken.",
                (
                    Edit(
                        "sampling.py",
                        "        out[b, seen] = picked / penalty\n",
                        "        out[b, seen] = torch.where(picked > 0, picked / penalty, picked * penalty)\n",
                    ),
                ),
            ),
            WrongFix(
                "top-p-off-by-one",
                "Fix both, but drop the token that crosses the top_p threshold.",
                (
                    Edit(
                        "sampling.py",
                        "        out[b, seen] = picked / penalty\n",
                        "        out[b, seen] = torch.where(picked > 0, picked / penalty, picked * penalty)\n",
                    ),
                    Edit(
                        "sampling.py",
                        "    remove_sorted = mass_before >= top_p\n"
                        "    remove_sorted[..., 0] = False  # the most likely token survives even when top_p is 0\n"
                        '    return logits.masked_fill(remove_sorted, float("-inf"))\n',
                        "    remove_sorted = probs.cumsum(dim=-1) > top_p\n"
                        "    remove_sorted[..., 0] = False\n"
                        "    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_index, remove_sorted)\n"
                        '    return logits.masked_fill(remove, float("-inf"))\n',
                    ),
                ),
            ),
        ),
    ),
    Task(
        id="stream-utf8",
        title="Streaming output shows replacement characters for non-ASCII text",
        category="tokenizer",
        instruction="""# Streaming breaks accents, CJK and emoji

Our streaming endpoint uses `tinyinfer.StreamDecoder` to turn tokens into text
as they arrive. Users see the replacement character (U+FFFD) wherever the text
has an accent, a CJK character or an emoji. Decoding the same tokens in one go
with `ByteTokenizer().decode(tokens)` gives the right text.

Find and fix the bug.

Streaming must stay streaming: each character has to be returned by the `push`
call that completes it. Holding text back until `flush` is not acceptable. For
any token stream, valid or not, joining every `push` result and the `flush`
result must equal `ByteTokenizer().decode(tokens)`.
""",
        bug=(
            Edit(
                "tokenizer.py",
                '        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")\n',
                "        self._pending = []\n",
            ),
            Edit(
                "tokenizer.py",
                "        return self._decoder.decode(bytes([token]))\n",
                '        return bytes([token]).decode("utf-8", errors="replace")\n',
            ),
            Edit(
                "tokenizer.py",
                '        return self._decoder.decode(b"", final=True)\n',
                '        return ""\n',
            ),
        ),
        wrong_fixes=(
            WrongFix(
                "buffer-until-flush",
                "Hold every token and decode the whole stream in flush().",
                (
                    Edit(
                        "tokenizer.py",
                        '        return bytes([token]).decode("utf-8", errors="replace")\n',
                        "        self._pending.append(token)\n        return \"\"\n",
                    ),
                    Edit(
                        "tokenizer.py",
                        '    def flush(self):\n        return ""\n',
                        "    def flush(self):\n"
                        '        return bytes(self._pending).decode("utf-8", errors="replace")\n',
                    ),
                ),
            ),
            WrongFix(
                "ignore-errors",
                "Use an incremental decoder that silently drops bad or truncated bytes.",
                (
                    Edit(
                        "tokenizer.py",
                        "        self._pending = []\n",
                        '        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")\n',
                    ),
                    Edit(
                        "tokenizer.py",
                        '        return bytes([token]).decode("utf-8", errors="replace")\n',
                        "        return self._decoder.decode(bytes([token]))\n",
                    ),
                ),
            ),
        ),
    ),
)


SINGLE_TASKS = TASKS


def _as_report(number, task):
    """Demote a task instruction to a numbered section of the combined one."""
    title, _, body = task.instruction.strip().partition("\n")
    return f"## Report {number}: {title.lstrip('# ')}\n{body}\n"


# Every bug planted at once. The reports are the six instructions above, unchanged.
TASKS = SINGLE_TASKS + (
    Task(
        id="all-at-once",
        title="Six bug reports against the same inference stack",
        category="multi-bug",
        instruction="# Six open bug reports\n\n"
        "Six separate reports came in against `tinyinfer`. They are reproduced below. "
        "Each has its own cause. Fix all of them.\n\n"
        + "\n".join(_as_report(n, task) for n, task in enumerate(SINGLE_TASKS, 1)),
        bug=tuple(edit for task in SINGLE_TASKS for edit in task.bug),
        graded_by=tuple(task.id for task in SINGLE_TASKS),
    ),
)


def apply_edits(root, edits):
    """Apply edits in place to the tinyinfer package at `root`. Each must match exactly once."""
    for edit in edits:
        path = root / edit.file
        text = path.read_text(encoding="utf-8")
        if text.count(edit.old) != 1:
            raise ValueError(f"{edit.file}: expected exactly one match for {edit.old!r}, found {text.count(edit.old)}")
        path.write_text(text.replace(edit.old, edit.new), encoding="utf-8", newline="\n")
