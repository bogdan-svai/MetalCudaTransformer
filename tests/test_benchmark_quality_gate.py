import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_optimization_suite.py"
    spec = importlib.util.spec_from_file_location("benchmark_optimization_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_gate_accepts_correct_fast_variant():
    module = _module()
    report = {
        "variants": {"baseline": {"benchmark": {"ok": True}}, "opt": {"benchmark": {"ok": True}}},
        "comparisons": {"opt": {"gpu.median": {"regression": -0.1}}},
    }
    assert module.evaluate_quality_gate(report, 0.05)["ok"] is True


def test_quality_gate_rejects_regression_and_failed_variant():
    module = _module()
    report = {
        "variants": {"baseline": {"benchmark": {"ok": True}}, "opt": {"error": "failed"}},
        "comparisons": {"opt": {"gpu.median": {"regression": 0.2}}},
    }
    gate = module.evaluate_quality_gate(report, 0.05)
    assert gate["ok"] is False
    assert len(gate["violations"]) == 2
