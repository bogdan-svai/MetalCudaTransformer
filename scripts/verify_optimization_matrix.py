#!/usr/bin/env python3
"""Run the optimizer quality gate across workload and data-size tiers."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "benchmark_optimization_suite.py"


def run_case(workload, tier, samples, warmups, max_regression, environment):
    command = [
        sys.executable,
        str(SUITE),
        "--only",
        workload,
        "--tier",
        tier,
        "--samples",
        str(samples),
        "--warmups",
        str(warmups),
        "--max-regression",
        str(max_regression),
    ]
    process = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return {
            "workload": workload,
            "tier": tier,
            "ok": False,
            "returncode": process.returncode,
            "error": f"invalid benchmark JSON: {exc}",
        }
    gate = payload.get("quality_gate", {})
    return {
        "workload": workload,
        "tier": tier,
        "ok": bool(gate.get("ok", False))
        and process.returncode == 0
        and all(
            item.get("returncode") == 0 and bool(item.get("benchmark"))
            for item in payload.get("variants", {}).values()
        ),
        "quality_gate": gate,
        "variants": {
            name: item.get("returncode") == 0 and bool(item.get("benchmark"))
            for name, item in payload.get("variants", {}).items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "stream", "hotspot", "reduction", "histogram"),
        default="all",
    )
    parser.add_argument(
        "--tiers", nargs="+", choices=("small", "medium", "large"), default=["medium"]
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--max-regression", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="mcu-optimization-matrix-") as directory:
        for tier in args.tiers:
            environment = os.environ.copy()
            environment["MCU_CACHE_DIR"] = str(Path(directory) / tier / "cache")
            workloads = (
                ("stream", "hotspot", "reduction", "histogram")
                if args.only == "all"
                else (args.only,)
            )
            for workload in workloads:
                results.append(
                    run_case(
                        workload,
                        tier,
                        max(1, args.samples),
                        max(1, args.warmups),
                        args.max_regression,
                        environment,
                    )
                )
    report = {
        "schema_version": 1,
        "suite": "mcu.optimization_matrix",
        "ok": all(item["ok"] for item in results),
        "configuration": {
            "workloads": list(dict.fromkeys(item["workload"] for item in results)),
            "tiers": args.tiers,
            "samples": max(1, args.samples),
            "warmups": max(1, args.warmups),
            "max_regression": args.max_regression,
        },
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
