"""Guardrail checks for parity tests to stay opt-in and documented."""

from __future__ import annotations

import ast
from pathlib import Path

PARITY_PATH = Path(__file__).resolve().parent / "test_parity.py"
_PARITY_SOURCE = PARITY_PATH.read_text(encoding="utf-8")
_PARITY_TREE = ast.parse(_PARITY_SOURCE, filename=str(PARITY_PATH))


def test_parity_module_has_clear_docstring() -> None:
    doc = ast.get_docstring(_PARITY_TREE)
    assert doc is not None
    assert "Numerical parity" in doc


def _decorator_name(node: ast.expr) -> str:
    """Return the dotted attribute path of a decorator expression (best-effort)."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        parts: list[str] = [node.attr]
        cur: ast.expr = node.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Name):
        return node.id
    return ""


def test_parity_tests_are_marked_slow() -> None:
    slow_tests: list[str] = []
    for stmt in _PARITY_TREE.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not stmt.name.startswith("test_"):
            continue
        for deco in stmt.decorator_list:
            name = _decorator_name(deco)
            if name.endswith("pytest.mark.slow") or name == "slow":
                slow_tests.append(stmt.name)
                break
    assert slow_tests, "Expected at least one slow-marked parity test"
