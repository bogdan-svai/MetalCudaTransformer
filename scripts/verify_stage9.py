#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "stage9_project"
MCU = ROOT / "mcu"


def run(command, cwd=None, env=None):
    started = perf_counter()
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, check=False,
    )
    return result, perf_counter() - started


def fields(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def summary(values):
    return {
        "samples": [round(value, 6) for value in values],
        "minimum": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(percentile(values, 0.95), 6),
    }


def json_command(arguments, cwd=None):
    result, elapsed = run([sys.executable, str(MCU), *arguments, "--json"], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout), elapsed


def verify(samples):
    with tempfile.TemporaryDirectory(prefix="mcu-stage9-") as temporary:
        project = Path(temporary) / "project"
        shutil.copytree(FIXTURE, project)
        expected_by_input = {}
        for input_path in sorted((project / "inputs").glob("*.in")):
            relative = str(input_path.relative_to(project))
            oracle_result, _ = run([sys.executable, str(project / "oracle.py"), relative], cwd=project)
            if oracle_result.returncode != 0:
                raise RuntimeError(oracle_result.stderr)
            expected_by_input[relative] = fields(oracle_result.stdout)

        clean_payload, clean_seconds = json_command(["make", "-C", str(project), "clean", "compile"])
        executable = project / "miner"
        if not executable.is_file():
            raise RuntimeError("Stage 9 Make build did not produce miner")

        no_op = []
        no_op_payloads = []
        for _ in range(samples):
            payload, elapsed = json_command(["make", "-C", str(project), "compile"])
            no_op.append(elapsed)
            no_op_payloads.append(payload)

        kernels = project / "src" / "kernels.cu"
        kernels.write_text(kernels.read_text(encoding="ascii") + "\n", encoding="ascii")
        incremental_payload, incremental_seconds = json_command(["make", "-C", str(project), "compile"])

        runtime_by_input = {}
        all_metrics = []
        deterministic = True
        for input_name, expected in expected_by_input.items():
            runtime_seconds = []
            runtime_metrics = []
            observed = []
            for index in range(samples):
                metric_path = project / f"runtime-{Path(input_name).stem}-{index}.json"
                environment = os.environ.copy()
                environment["MCU_METRICS_FILE"] = str(metric_path)
                result, elapsed = run([str(executable), input_name], cwd=project, env=environment)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr or result.stdout)
                actual = fields(result.stdout)
                if actual != expected:
                    raise RuntimeError(f"oracle mismatch for {input_name}: expected {expected}, observed {actual}")
                observed.append(actual)
                runtime_seconds.append(elapsed)
                runtime_metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
            input_deterministic = all(item == observed[0] for item in observed)
            deterministic = deterministic and input_deterministic
            all_metrics.extend(runtime_metrics)
            runtime_by_input[input_name] = {
                "wall": summary(runtime_seconds),
                "metrics": runtime_metrics,
                "deterministic": input_deterministic,
            }

        cold_startups = [
            metric["phases"].get("device_initialization", {}).get("seconds", 0.0)
            + metric["phases"].get("artifact_load", {}).get("seconds", 0.0)
            + metric["phases"].get("pipeline_creation", {}).get("seconds", 0.0)
            for metric in all_metrics
        ]
        dispatch_overheads = [
            metric["phases"].get("command_submission", {}).get("seconds", 0.0)
            for metric in all_metrics
        ]
        return {
            "schema_version": 1,
            "ok": deterministic,
            "machine": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "device": all_metrics[-1]["device"],
            },
            "correctness": {
                "inputs": expected_by_input,
                "runs_per_input": samples,
                "deterministic": deterministic,
            },
            "build": {
                "clean_seconds": round(clean_seconds, 6),
                "clean_driver_invocations": clean_payload["make"]["driver_invocations"],
                "no_op": summary(no_op),
                "no_op_driver_invocation_counts": [
                    len(item["make"]["driver_invocations"]) for item in no_op_payloads
                ],
                "incremental_seconds": round(incremental_seconds, 6),
                "incremental_driver_invocations": incremental_payload["make"]["driver_invocations"],
            },
            "runtime": {"inputs": runtime_by_input},
            "budgets": {
                "make_no_op_p95_seconds": 0.20,
                "incremental_seconds": 1.50,
                "clean_seconds": 10.00,
                "observed": {
                    "make_no_op_p95_pass": percentile(no_op, 0.95) <= 0.20,
                    "incremental_pass": incremental_seconds <= 1.50,
                    "clean_pass": clean_seconds <= 10.00,
                    "cold_startup_p95_pass": percentile(cold_startups, 0.95) <= 0.25,
                    "dispatch_overhead_p95_pass": percentile(dispatch_overheads, 0.95) <= 0.05,
                },
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Verify the MCU Stage 9 project contract")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Stage 9 native verification requires macOS Metal")
    report = verify(max(1, args.samples))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
