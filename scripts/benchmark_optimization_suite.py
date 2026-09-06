#!/usr/bin/env python3
"""Run the product benchmark suite for each optimizer configuration."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "benchmark_suite.py"


def _median_metrics(value, prefix=()):
    metrics = {}
    if isinstance(value, dict):
        if "median" in value and isinstance(value["median"], (int, float)):
            metrics[".".join(prefix + ("median",))] = value["median"]
        for key, child in value.items():
            metrics.update(_median_metrics(child, prefix + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            metrics.update(_median_metrics(child, prefix + (str(index),)))
    return metrics


def evaluate_quality_gate(report, max_regression=0.05):
    violations = []
    for name, payload in report.get("variants", {}).items():
        if not payload.get("benchmark"):
            violations.append({"variant": name, "reason": "benchmark_failed"})
    for name, metrics in report.get("comparisons", {}).items():
        for path, comparison in metrics.items():
            regression = comparison.get("regression")
            if regression is not None and regression > max_regression:
                violations.append(
                    {
                        "variant": name,
                        "metric": path,
                        "regression": regression,
                        "limit": max_regression,
                    }
                )
    return {"ok": not violations, "max_regression": max_regression, "violations": violations}


def _throughput_metric(path):
    return any(token in path.lower() for token in ("per_second", "gb_per_second", "gflops"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", choices=("all", "stream", "hotspot", "reduction", "histogram"), default="all"
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Fail when a median metric regresses by more than this fraction.",
    )
    parser.add_argument("--output", type=Path)
    args, extra = parser.parse_known_args()
    variants = {
        "baseline": {
            "MCU_METAL_CONSTANT_PROMOTION": "0",
            "MCU_METAL_UNROLL_MAX": "0",
            "MCU_METAL_SCALARIZE": "0",
        },
        "constant_promotion": {
            "MCU_METAL_CONSTANT_PROMOTION": "1",
            "MCU_METAL_UNROLL_MAX": "0",
            "MCU_METAL_SCALARIZE": "0",
        },
        "unroll_16": {
            "MCU_METAL_CONSTANT_PROMOTION": "0",
            "MCU_METAL_UNROLL_MAX": "16",
            "MCU_METAL_SCALARIZE": "0",
        },
        "all_safe": {"MCU_METAL_CONSTANT_PROMOTION": "1", "MCU_METAL_UNROLL_MAX": "16"},
    }
    report = {"schema_version": 1, "suite": "mcu.optimizer_matrix", "variants": {}}
    with tempfile.TemporaryDirectory(prefix="mcu-suite-matrix-") as directory:
        for name, overrides in variants.items():
            env = os.environ.copy()
            env.update(overrides)
            env["MCU_CACHE_DIR"] = str(Path(directory) / name / "cache")
            command = [
                sys.executable,
                str(SUITE),
                "--only",
                args.only,
                "--samples",
                str(args.samples),
                "--warmups",
                str(args.warmups),
                *extra,
            ]
            result = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
            )
            payload = {"environment": overrides, "returncode": result.returncode}
            if result.returncode == 0:
                payload["benchmark"] = json.loads(result.stdout)
            else:
                payload["error"] = result.stderr or result.stdout
            report["variants"][name] = payload
    baseline = report["variants"].get("baseline", {}).get("benchmark")
    if baseline:
        baseline_metrics = _median_metrics(baseline)
        comparisons = {}
        for name, payload in report["variants"].items():
            benchmark = payload.get("benchmark")
            if not benchmark:
                continue
            metrics = _median_metrics(benchmark)
            comparisons[name] = {
                path: {
                    "baseline": baseline_value,
                    "value": metrics[path],
                    "speedup": (
                        (
                            metrics[path] / baseline_value
                            if _throughput_metric(path)
                            else baseline_value / metrics[path]
                        )
                        if metrics[path]
                        else None
                    ),
                    "regression": (
                        (
                            baseline_value / metrics[path] - 1
                            if _throughput_metric(path)
                            else metrics[path] / baseline_value - 1
                        )
                        if baseline_value
                        else None
                    ),
                }
                for path, baseline_value in baseline_metrics.items()
                if path in metrics and baseline_value
            }
        report["comparisons"] = comparisons
    report["quality_gate"] = evaluate_quality_gate(report, args.max_regression)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["quality_gate"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
