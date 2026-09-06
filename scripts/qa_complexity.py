#!/usr/bin/env python3
"""Small dependency-free complexity report for the Python compiler core."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "core", ROOT / "cli", ROOT / "runtime")


def complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    score = 1
    for child in ast.walk(node):
        if isinstance(
            child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With)
        ):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(0, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            score += 1 + len(child.ifs)
    return score


def main() -> int:
    rows = []
    for root in TARGETS:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as error:
                print(f"syntax-error {path}:{error.lineno}: {error.msg}")
                return 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows.append((complexity(node), path.relative_to(ROOT), node.lineno, node.name))
    rows.sort(reverse=True)
    print("complexity file:line function")
    for score, path, line, name in rows[:20]:
        print(f"{score:10d} {path}:{line} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
