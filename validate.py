"""Check that every task's grader behaves as claimed, without Docker.

For each task this runs the shipped hidden tests against:
  * the buggy starting state     -> task tests must FAIL, regression tests must PASS
  * the reference solution       -> everything must PASS
  * each known wrong fix         -> the reward must be 0

It also runs the public smoke test against the buggy state and the reference.
Run with `uv run python validate.py`. Exits non-zero if any expectation fails
and rewrites VALIDATION.md with the table.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bugs import TASKS, apply_edits

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"


FAILURES = {}  # last output of every pytest run, for diagnosing surprises


def pytest_passes(targets, app_dir):
    env = dict(os.environ, APP_DIR=str(app_dir), PYTHONDONTWRITEBYTECODE="1")
    # Pin rootdir and cwd to the directory holding the tests. Otherwise pytest walks up to a
    # shared parent and can trip over another variant's temp dir as it is being deleted.
    base = Path(targets[0]) if Path(targets[0]).is_dir() else Path(targets[0]).parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *map(str, targets), "-q", "-x", "-p", "no:cacheprovider", f"--rootdir={base}"],
        env=env,
        cwd=base,
        capture_output=True,
        text=True,
    )
    FAILURES[(tuple(map(str, targets)), str(app_dir))] = result.stdout[-3000:] + result.stderr[-1500:]
    return result.returncode == 0


def run_variant(task, name, prepare):
    """Build the variant in a temp dir and grade it. Returns (task tests, regression, smoke)."""
    task_dir = TASKS_DIR / task.id
    test_files = [task_dir / "tests" / name for name in task.test_files]
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp)
        shutil.copytree(task_dir / "environment" / "tinyinfer", app / "tinyinfer")
        shutil.copytree(task_dir / "environment" / "tests", app / "tests")
        prepare(app / "tinyinfer")
        outcome = (
            pytest_passes(test_files, app),
            pytest_passes([task_dir / "tests" / "test_regression.py"], app),
            pytest_passes([app / "tests"], app),
        )
        logs = [text for (_, where), text in FAILURES.items() if where == str(app)]
        return outcome, logs


def variants(task):
    solution = TASKS_DIR / task.id / "solution" / "files"
    yield "buggy start", "The task as the agent receives it.", lambda pkg: None
    yield (
        "reference fix",
        "The shipped solution.",
        lambda pkg: shutil.copytree(solution, pkg, dirs_exist_ok=True),
    )
    for fix in task.wrong_fixes:
        yield f"wrong fix: {fix.name}", fix.description, lambda pkg, fix=fix: apply_edits(pkg, fix.edits)


def expectation(name, task_ok, regression_ok, smoke_ok):
    reward = int(task_ok and regression_ok)
    if name == "buggy start":
        return reward, (not task_ok) and regression_ok and smoke_ok
    if name == "reference fix":
        return reward, task_ok and regression_ok and smoke_ok
    return reward, reward == 0


def main():
    jobs = [(task, name, note, prepare) for task in TASKS for name, note, prepare in variants(task)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda job: run_variant(job[0], job[1], job[3]), jobs))

    mark = {True: "pass", False: "FAIL"}
    lines = [
        "# Grader validation",
        "",
        "Produced by `uv run python validate.py`. Reward is 1 only when the task tests and the",
        "regression tests both pass. Every row must behave as expected for the pack to be valid.",
        "",
        "| Task | Variant | Task tests | Regression | Reward | As expected |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    all_ok = True
    for (task, name, note, _), ((task_ok, regression_ok, smoke_ok), logs) in zip(jobs, results):
        reward, ok = expectation(name, task_ok, regression_ok, smoke_ok)
        all_ok &= ok
        if not ok:
            print(f"UNEXPECTED: {task.id} / {name} (smoke test passed: {smoke_ok})", file=sys.stderr)
            for log in logs:
                print(log, file=sys.stderr)
        lines.append(
            f"| {task.id} | {name} | {mark[task_ok]} | {mark[regression_ok]} | {reward} | {'yes' if ok else 'NO'} |"
        )
    lines += ["", "Wrong fixes checked:", ""]
    for task in TASKS:
        for fix in task.wrong_fixes:
            lines.append(f"- `{task.id}` / `{fix.name}`: {fix.description}")
    text = "\n".join(lines) + "\n"
    (ROOT / "VALIDATION.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print("ALL EXPECTATIONS MET" if all_ok else "SOME EXPECTATIONS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
