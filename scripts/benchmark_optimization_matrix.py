#!/usr/bin/env python3
"""Reproducible before/after benchmark for one CUDA executable."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def measure(executable, args, environment, samples):
    values = []
    for _ in range(samples):
        started = perf_counter()
        result = subprocess.run(
            [str(executable), *args],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or str(result.returncode))
        values.append(elapsed)
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--run-args", nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    source = options.source.resolve()
    matrix = {
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
        "unroll": {
            "MCU_METAL_CONSTANT_PROMOTION": "0",
            "MCU_METAL_UNROLL_MAX": "16",
            "MCU_METAL_SCALARIZE": "0",
        },
        "all_safe": {"MCU_METAL_CONSTANT_PROMOTION": "1", "MCU_METAL_UNROLL_MAX": "16"},
    }
    report = {"source": str(source), "samples": options.samples, "variants": {}}
    with tempfile.TemporaryDirectory(prefix="mcu-optimization-matrix-") as directory:
        directory = Path(directory)
        for name, overrides in matrix.items():
            environment = os.environ.copy()
            environment.update(overrides)
            environment["MCU_CACHE_DIR"] = str(directory / name / "cache")
            executable = directory / name / "program"
            executable.parent.mkdir(parents=True, exist_ok=True)
            build = subprocess.run(
                [
                    sys.executable,
                    str(MCU),
                    "compile",
                    str(source),
                    "-O3",
                    "-o",
                    str(executable),
                    "--json",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = {"environment": overrides, "build_returncode": build.returncode}
            if build.returncode == 0:
                try:
                    build_json = json.loads(build.stdout)
                    sidecar = Path(build_json["build"]["sidecar"])
                    routing = json.loads(
                        (sidecar / "routing_table.txt").read_text(encoding="utf-8")
                    )
                    passes = routing.get("metal_optimization", {}).get("passes", [])
                    payload["compiler"] = {
                        "passes": passes,
                        "ir_optimization": routing.get("ir_optimization", {}),
                        "applied_passes": [
                            item["name"] for item in passes if item.get("replacements", 0) > 0
                        ],
                        "skipped_passes": [
                            item["name"] for item in passes if item.get("replacements", 0) == 0
                        ],
                    }
                except (KeyError, OSError, json.JSONDecodeError):
                    payload["compiler"] = {"available": False}
                payload["runtime"] = measure(
                    executable, options.run_args, environment, options.samples
                )
                payload["correct"] = True
            else:
                payload["correct"] = False
                payload["build_output"] = build.stderr or build.stdout
            report["variants"][name] = payload
    baseline = report["variants"].get("baseline", {}).get("runtime", {}).get("median")
    if baseline:
        for payload in report["variants"].values():
            runtime = payload.get("runtime", {})
            if runtime:
                runtime["speedup_vs_baseline"] = baseline / runtime["median"]
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if options.output:
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
