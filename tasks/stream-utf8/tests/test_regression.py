"""Behaviour that every task must leave intact. Passes on every buggy starting state."""

import ast
from pathlib import Path

import pytest
import torch
from common import APP_DIR, NO_EOS, app, expected, make_prompts, model_pair, ref

# The stack needs torch and a little of the standard library. Anything that can reach
# into the interpreter or the filesystem is out, so a submission cannot go looking for
# the grader's reference copy.
ALLOWED_IMPORTS = {
    "torch", "math", "codecs", "dataclasses", "typing", "functools", "itertools",
    "collections", "unicodedata", "warnings", "numbers", "operator", "enum", "abc",
    "__future__",
}
BANNED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "globals", "locals", "vars",
    "__builtins__", "__loader__", "__spec__", "breakpoint", "input",
    "getattr", "setattr", "delattr",
}
# Attribute names that lead from an allowed module back into the interpreter,
# for example torch.sys.modules.
BANNED_ATTRIBUTES = {
    "sys", "os", "modules", "importlib", "inspect", "gc", "builtins", "subprocess",
    "ctypes", "pickle", "load", "hub", "jit", "serialization", "_getframe",
    "f_globals", "f_locals", "f_back",
}


def test_submission_uses_only_allowed_imports():
    files = sorted((Path(APP_DIR) / "tinyinfer").rglob("*.py"))
    assert files
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [] if node.level else [node.module or ""]
            else:
                modules = []
            for module in modules:
                assert module.split(".")[0] in ALLOWED_IMPORTS, f"{file.name} imports {module}"
            if isinstance(node, ast.Name):
                assert node.id not in BANNED_NAMES, f"{file.name} uses {node.id}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in BANNED_ATTRIBUTES, f"{file.name} uses .{node.attr}"
                assert not node.attr.startswith("__") or node.attr in {"__init__", "__name__", "__class__"}, (
                    f"{file.name} uses {node.attr}"
                )


def test_parameter_names_unchanged():
    a, r = model_pair(0)
    assert list(a.state_dict()) == list(r.state_dict())


@pytest.mark.parametrize("seed", range(3))
def test_logits_match_reference(seed):
    a, r = model_pair(seed)
    tokens = torch.tensor(make_prompts(seed, [20, 20, 20]))
    got, _ = a(tokens)
    want, _ = r(tokens)
    assert torch.allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("use_cache", [True, False])
def test_single_prompt(seed, use_cache):
    a, r = model_pair(seed)
    for prompt in make_prompts(seed, [1, 5, 17, 30]):
        got = app.generate(a, [prompt], 12, eos_id=NO_EOS, use_cache=use_cache)
        assert got == expected(r, [prompt], 12)


@pytest.mark.parametrize("seed", range(3))
def test_equal_length_batch(seed):
    a, r = model_pair(seed)
    prompts = make_prompts(seed, [9, 9, 9, 9])
    assert app.generate(a, prompts, 12, eos_id=NO_EOS) == expected(r, prompts, 12)


def test_single_prompt_stops_at_eos():
    a, r = model_pair(1)
    (prompt,) = make_prompts(1, [8])
    stream = expected(r, [prompt], 12)[0]
    eos = stream[5]
    want = stream[: stream.index(eos)]
    assert app.generate(a, [prompt], 12, eos_id=eos) == [want]


def test_max_new_tokens_zero():
    a, _ = model_pair(0)
    assert app.generate(a, [[1, 2, 3]], 0) == [[]]


def test_top_k_filter():
    logits = torch.randn(4, 259, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert torch.equal(app.top_k_filter(logits, 10), ref.top_k_filter(logits, 10))


def test_sampler_returns_valid_tokens():
    logits = torch.randn(4, 259, generator=torch.Generator().manual_seed(0))
    sampler = app.make_sampler(temperature=0.7, top_k=40, generator=torch.Generator().manual_seed(0))
    tokens = sampler(logits, [[1], [2], [3], [4]])
    assert tokens.shape == (4,)
    assert ((tokens >= 0) & (tokens < 259)).all()


def test_tokenizer_round_trip():
    tok = app.ByteTokenizer()
    text = "plain ascii and héllo 日本 \U0001f642"
    tokens = tok.encode(text)
    assert tokens[0] == app.BOS
    assert tok.decode(tokens) == text


def test_stream_decoder_ascii():
    decoder = app.StreamDecoder()
    tokens = app.ByteTokenizer().encode("hello, world")
    pieces = [decoder.push(t) for t in tokens]
    assert pieces == [""] + list("hello, world")
    assert decoder.flush() == ""
