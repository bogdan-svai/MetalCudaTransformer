#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "sha256_miner"
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


def json_command(arguments, cwd=None, env=None):
    result, elapsed = run([sys.executable, str(MCU), *arguments, "--json"], cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout), elapsed


def configure_find_nonce_geometry(project, grid, block):
    main_source = project / "app" / "main.cpp"
    source = main_source.read_text(encoding="utf-8")
    replacement = f"find_nonce<<<{grid}, {block}>>>"
    pattern = re.compile(r"find_nonce\s*<<<\s*\d+\s*,\s*\d+\s*>>>")
    if len(pattern.findall(source)) != 1:
        raise RuntimeError("find_nonce launch shape was not found exactly once")
    main_source.write_text(pattern.sub(replacement, source), encoding="utf-8")


def verify(samples, find_nonce_grid=4096, find_nonce_block=128):
    with tempfile.TemporaryDirectory(prefix="mcu-sha256-") as temporary:
        project = Path(temporary) / "project"
        shutil.copytree(FIXTURE, project)
        configure_find_nonce_geometry(project, find_nonce_grid, find_nonce_block)
        environment = os.environ.copy()
        environment.setdefault("MCU_CACHE_DIR", str(Path(temporary) / "cache"))
        expected_by_input = {}
        for input_path in sorted((project / "inputs").glob("*.in")):
            relative = str(input_path.relative_to(project))
            oracle_result, _ = run([sys.executable, str(project / "oracle.py"), relative], cwd=project)
            if oracle_result.returncode != 0:
                raise RuntimeError(oracle_result.stderr)
            expected_by_input[relative] = fields(oracle_result.stdout)

        clean_payload, clean_seconds = json_command(
            ["make", "-C", str(project), "clean", "compile"],
            env=environment,
        )
        executable = project / "miner"
        if not executable.is_file():
            raise RuntimeError("Make build did not produce miner")

        no_op = []
        no_op_payloads = []
        for _ in range(samples):
            payload, elapsed = json_command(["make", "-C", str(project), "compile"], env=environment)
            no_op.append(elapsed)
            no_op_payloads.append(payload)

        kernels = project / "src" / "kernels.cu"
        kernels.write_text(kernels.read_text(encoding="ascii") + "\n", encoding="ascii")
        incremental_payload, incremental_seconds = json_command(
            ["make", "-C", str(project), "compile"],
            env=environment,
        )

        runtime_by_input = {}
        all_metrics = []
        deterministic = True
        for input_name, expected in expected_by_input.items():
            runtime_seconds = []
            runtime_metrics = []
            observed = []
            for index in range(samples):
                metric_path = project / f"runtime-{Path(input_name).stem}-{index}.json"
                run_environment = environment.copy()
                run_environment["MCU_METRICS_FILE"] = str(metric_path)
                result, elapsed = run([str(executable), input_name], cwd=project, env=run_environment)
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
            "benchmark_parameters": {
                "find_nonce_grid": find_nonce_grid,
                "find_nonce_block": find_nonce_block,
                "find_nonce_threads": find_nonce_grid * find_nonce_block,
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
    parser = argparse.ArgumentParser(description="Verify the MCU project contract")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--find-nonce-grid", type=int, default=4096)
    parser.add_argument("--find-nonce-block", type=int, default=128)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("native verification requires macOS Metal")
    if args.find_nonce_grid < 1 or args.find_nonce_block < 1 or args.find_nonce_block > 1024:
        parser.error("find_nonce geometry requires grid >= 1 and block in [1, 1024]")
    report = verify(max(1, args.samples), args.find_nonce_grid, args.find_nonce_block)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
