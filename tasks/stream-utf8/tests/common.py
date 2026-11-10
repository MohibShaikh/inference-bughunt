"""Shared helpers for the hidden tests.

`app` is the package under test (from APP_DIR, default /app). `ref` is a clean
copy of the same stack that ships only with the tests. Expected outputs always
come from `ref`, run one prompt at a time with the cache off.
"""

import importlib
import importlib.util
import os
import secrets
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
APP_DIR = os.environ.get("APP_DIR", "/app")

if APP_DIR not in sys.path:
    # Appended, so nothing placed in APP_DIR can shadow torch or pytest.
    sys.path.append(APP_DIR)

import tinyinfer as app  # noqa: E402


def _load_reference():
    """Load the clean stack from its file path under a random module name.

    It is never on sys.path and its name changes every run, so the package
    under test cannot import it. test_regression.py also rejects submissions
    that import the modules needed to go looking for it.
    """
    package = HERE / "reference-copy" / "tinyinfer_ref"  # not importable by name
    name = "_ref_" + secrets.token_hex(8)
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ref = _load_reference()
left_pad = importlib.import_module(ref.__name__ + ".generate")._left_pad

torch.set_num_threads(1)

NO_EOS = -1


def model_pair(seed, dtype=torch.float64, **cfg):
    """Return (app model, reference model) with identical weights."""
    torch.manual_seed(seed)
    ref_model = ref.TinyLM(ref.Config(**cfg)).to(dtype).eval()
    app_model = app.TinyLM(app.Config(**cfg)).to(dtype).eval()
    app_model.load_state_dict(ref_model.state_dict())
    return app_model, ref_model


def make_prompts(seed, lengths):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (n,), generator=g).tolist() for n in lengths]


def expected(ref_model, prompts, max_new_tokens, eos_id=NO_EOS):
    return [
        ref.generate(ref_model, [p], max_new_tokens, eos_id=eos_id, use_cache=False)[0]
        for p in prompts
    ]


class Recorder:
    """Record the (batch, new_tokens) shape of every forward call on a model."""

    def __init__(self, model):
        self.calls = []
        inner = model.forward

        def forward(tokens, *args, **kwargs):
            self.calls.append(tuple(tokens.shape))
            return inner(tokens, *args, **kwargs)

        model.forward = forward
