"""Generate one Harbor task directory per bug under tasks/.

Run with `uv run python build.py`. The output is committed, so this only needs
re-running after changing the stack, a bug definition or a hidden test.
"""

import shutil
from pathlib import Path

from bugs import FOOTER, TASKS, apply_edits

ROOT = Path(__file__).resolve().parent
STACK = ROOT / "stack" / "tinyinfer"
CHECKS = ROOT / "checks"
OUT = ROOT / "tasks"

TORCH = "torch==2.15.0"
PIP = (
    f"RUN pip install --no-cache-dir {TORCH} --index-url https://download.pytorch.org/whl/cpu \\\n"
    " && pip install --no-cache-dir numpy==2.3.5 pytest==9.2.0\n"
)

ENV_DOCKERFILE = f"""FROM python:3.12-slim

{PIP}
WORKDIR /app
COPY tinyinfer /app/tinyinfer
COPY tests /app/tests
COPY README.md /app/README.md
"""

# The verifier runs in its own container. Harbor copies /app/tinyinfer over from
# the agent's container, so nothing else the agent touched can reach the grader.
TESTS_DOCKERFILE = f"""FROM python:3.12-slim

{PIP}
WORKDIR /app
COPY . /tests/
"""

TEST_SH = """#!/bin/bash
# Writes 1 to the reward file only if every hidden test passes.
mkdir -p /logs/verifier
if APP_DIR=/app python -m pytest /tests -q -rA -p no:cacheprovider; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

SOLVE_SH = """#!/bin/bash
set -euo pipefail
cp -r /solution/files/. /app/tinyinfer/
"""

APP_README = """# tinyinfer

A small byte-level language model inference stack in PyTorch.

- `tinyinfer/model.py`: decoder-only transformer with a key/value cache.
- `tinyinfer/generate.py`: batched generation (left padding, KV cache, EOS handling, sliding window).
- `tinyinfer/sampling.py`: repetition penalty, top-k, top-p and the sampler.
- `tinyinfer/tokenizer.py`: byte tokenizer and streaming detokenizer.

Run the smoke test with `python -m pytest /app/tests`.
"""

TASK_TOML = """schema_version = "1.4"
artifacts = ["/app/tinyinfer"]

[task]
name = "inference-bughunt/{id}"
version = "1.0.0"
description = "{title}"
authors = []
keywords = ["inference", "debugging", "pytorch", "{category}"]

[metadata]
category = "{category}"

[verifier]
timeout_sec = 600.0
environment_mode = "separate"

[agent]
timeout_sec = 1800.0

[environment]
network_mode = "public"
build_timeout_sec = 1200.0
os = "linux"
cpus = 2
memory_mb = 4096
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_package(src, dst):
    """Copy the package with LF line endings, skipping caches."""
    for file in sorted(src.glob("*.py")):
        write(dst / file.name, file.read_text(encoding="utf-8"))


def build_task(task):
    out = OUT / task.id
    write(out / "instruction.md", task.instruction.rstrip() + "\n" + FOOTER)
    write(out / "task.toml", TASK_TOML.format(id=task.id, title=task.title, category=task.category))

    env = out / "environment"
    copy_package(STACK, env / "tinyinfer")
    apply_edits(env / "tinyinfer", task.bug)
    write(env / "tests" / "test_smoke.py", (CHECKS / "public" / "test_smoke.py").read_text(encoding="utf-8"))
    write(env / "README.md", APP_README)
    write(env / "Dockerfile", ENV_DOCKERFILE)

    # The solution is every file the bug touched, restored from the clean stack.
    for name in sorted({edit.file for edit in task.bug}):
        write(out / "solution" / "files" / name, (STACK / name).read_text(encoding="utf-8"))
    write(out / "solution" / "solve.sh", SOLVE_SH)

    tests = out / "tests"
    for name in ("common.py", "test_regression.py", *task.test_files):
        write(tests / name, (CHECKS / name).read_text(encoding="utf-8"))
    copy_package(STACK, tests / "reference-copy" / "tinyinfer_ref")
    write(tests / "test.sh", TEST_SH)
    write(tests / "Dockerfile", TESTS_DOCKERFILE)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for task in TASKS:
        build_task(task)
        print(f"built tasks/{task.id}")


if __name__ == "__main__":
    main()
