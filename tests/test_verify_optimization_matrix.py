import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "verify_optimization_matrix", ROOT / "scripts" / "verify_optimization_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_case_rejects_quality_gate_regression(monkeypatch):
    module = _module()

    class Result:
        returncode = 0
        stdout = json.dumps({"ok": True, "quality_gate": {"ok": False}, "variants": {}})
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    result = module.run_case("reduction", "small", 1, 1, 0.2, {})
    assert result["ok"] is False


def test_matrix_case_accepts_complete_quality_gate(monkeypatch):
    module = _module()

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "ok": True,
                "quality_gate": {"ok": True},
                "variants": {"baseline": {"returncode": 0, "benchmark": {"ok": True}}},
            }
        )
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    result = module.run_case("stream", "medium", 1, 1, 0.2, {})
    assert result["ok"] is True
