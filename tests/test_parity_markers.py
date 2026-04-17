"""Guardrail checks for parity tests to stay opt-in and documented."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path

PARITY_PATH = Path(__file__).resolve().parent / "test_parity.py"
_spec = importlib.util.spec_from_file_location("parity_mod", PARITY_PATH)
assert _spec is not None and _spec.loader is not None
parity_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity_mod)


def test_parity_module_has_clear_docstring() -> None:
    doc = inspect.getdoc(parity_mod)
    assert doc is not None
    assert "Numerical parity" in doc


def test_parity_tests_are_marked_slow() -> None:
    slow_tests = []
    for name in dir(parity_mod):
        if name.startswith("test_"):
            fn = getattr(parity_mod, name)
            marks = getattr(fn, "pytestmark", [])
            if not isinstance(marks, list):
                marks = [marks]
            if any(getattr(m, "name", "") == "slow" for m in marks):
                slow_tests.append(name)
    assert slow_tests, "Expected at least one slow-marked parity test"
