#!/usr/bin/env python3
import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def summary(values):
    if not values:
        return {"samples": [], "minimum": None, "median": None, "p95": None}
    return {
        "samples": [round(value, 6) for value in values],
        "minimum": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(percentile(values, 0.95), 6),
    }


def run(command, cwd=ROOT):
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return process


def run_report(command, output=None):
    full_command = [*command]
    if output:
        full_command.extend(["--output", str(output)])
    process = run(full_command)
    if process.returncode != 0:
        raise RuntimeError(process.stderr or process.stdout)
    if output and output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    return json.loads(process.stdout)


def command_stdout(command):
    process = run(command)
    return process.stdout.strip() if process.returncode == 0 else None


def machine_metadata():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "sdk": command_stdout(["xcrun", "-sdk", "macosx", "--show-sdk-path"]),
        "git_revision": command_stdout(["git", "rev-parse", "--short", "HEAD"]),
    }


def summarize_sha256(report):
    observed = report.get("budgets", {}).get("observed", {})
    build = report.get("build", {})
    correctness_inputs = report.get("correctness", {}).get("inputs", {})
    kernel_metrics = {}
    for input_name, runtime in report.get("runtime", {}).get("inputs", {}).items():
        per_input = {}
        for metrics in runtime.get("metrics", []):
            for phase, value in metrics.get("phases", {}).items():
                prefix = "gpu_execution.kernel."
                if phase.startswith(prefix):
                    kernel = phase[len(prefix) :]
                    per_input.setdefault(kernel, []).append(float(value.get("seconds", 0.0)))
        kernel_metrics[input_name] = {
            kernel: summary(values) for kernel, values in sorted(per_input.items())
        }
    return {
        "ok": bool(report.get("ok")) and all(observed.values()),
        "device": report.get("machine", {}).get("device"),
        "clean_seconds": build.get("clean_seconds"),
        "incremental_seconds": build.get("incremental_seconds"),
        "no_op": build.get("no_op", {}),
        "nonce_results": {
            input_name: {
                "nonce": values.get("nonce"),
                "block_hash": values.get("block_hash"),
            }
            for input_name, values in sorted(correctness_inputs.items())
            if "nonce" in values
        },
        "kernel_gpu_seconds": kernel_metrics,
        "budget_observed": observed,
    }


def summarize_streams(report):
    tiers = report.get("tiers", {})
    return {
        "ok": bool(report.get("ok")),
        "budget_observed": report.get("budgets", {}),
        "tiers": {
            name: {
                "launches": tier.get("launches"),
                "element_count": tier.get("element_count"),
                "median_speedup": tier.get("median_speedup"),
                "minimum_speedup": tier.get("minimum_speedup"),
            }
            for name, tier in sorted(tiers.items())
        },
    }


def summarize_cublas(report):
    cases = report.get("cases", [])
    return {
        "ok": bool(report.get("ok")),
        "cases": [
            {
                "size": item.get("size"),
                "returncode": item.get("returncode"),
                "process_seconds": item.get("metrics", {}).get("process_seconds"),
                "backend_mps": item.get("metrics", {})
                .get("counters", {})
                .get("cublas_sgemm_backend_mps", 0),
            }
            for item in cases
        ],
    }


def summarize_checker(report):
    cases = report.get("cases", [])
    by_tier = {}
    for item in cases:
        tier = item.get("tier", "unknown")
        by_tier.setdefault(tier, {"cases": 0, "passed": 0, "run_seconds": [], "build_seconds": []})
        by_tier[tier]["cases"] += 1
        by_tier[tier]["passed"] += 1 if item.get("passed") else 0
        by_tier[tier]["run_seconds"].append(float(item.get("run_seconds", 0.0)))
        by_tier[tier]["build_seconds"].append(float(item.get("build_seconds", 0.0)))
    return {
        "ok": bool(report.get("ok")),
        "passed": report.get("passed"),
        "total": report.get("total"),
        "tiers": {
            tier: {
                "cases": values["cases"],
                "passed": values["passed"],
                "run_seconds": summary(values["run_seconds"]),
                "build_seconds": summary(values["build_seconds"]),
            }
            for tier, values in sorted(by_tier.items())
        },
        "slowest_cases": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "tier": item.get("tier"),
                "run_seconds": item.get("run_seconds"),
            }
            for item in sorted(cases, key=lambda case: case.get("run_seconds", 0.0), reverse=True)[
                :5
            ]
        ],
    }


def build_report(samples, include_cublas=True, include_checker=True):
    with tempfile.TemporaryDirectory(prefix="mcu-performance-") as temporary:
        temp = Path(temporary)
        sha256 = run_report(
            [sys.executable, str(ROOT / "scripts" / "verify_sha256.py"), "--samples", str(samples)],
            temp / "sha256.json",
        )
        streams = run_report(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_streams.py"),
                "--samples",
                str(samples),
            ],
            temp / "streams.json",
        )
        cublas = None
        if include_cublas:
            cublas = run_report([sys.executable, str(ROOT / "scripts" / "verify_cublas.py")])
        checker = None
        if include_checker:
            checker = run_report(
                [sys.executable, str(ROOT / "scripts" / "cuda_checker.py"), "--tier", "all"],
                temp / "cuda_checker.json",
            )

    sha256_summary = summarize_sha256(sha256)
    streams_summary = summarize_streams(streams)
    cublas_summary = summarize_cublas(cublas) if cublas else None
    checker_summary = summarize_checker(checker) if checker else None
    ok = sha256_summary["ok"] and streams_summary["ok"]
    if cublas_summary:
        ok = ok and cublas_summary["ok"]
    if checker_summary:
        ok = ok and checker_summary["ok"]
    return {
        "schema_version": 2,
        "suite": "performance",
        "ok": ok,
        "samples": samples,
        "machine": machine_metadata(),
        "reports": {
            "sha256": sha256_summary,
            "streams": streams_summary,
            "cublas": cublas_summary,
            "cuda_checker": checker_summary,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Collect MCU performance baselines")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-cublas", action="store_true")
    parser.add_argument("--skip-checker", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("performance verification requires macOS Metal")
    report = build_report(
        max(1, args.samples),
        include_cublas=not args.skip_cublas,
        include_checker=not args.skip_checker,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
