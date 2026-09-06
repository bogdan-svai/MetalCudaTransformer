#!/usr/bin/env python3
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def run(command, *, cwd=ROOT, env=None):
    started = perf_counter()
    process = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return process, perf_counter() - started


def fields(output):
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def summary(values):
    ordered = sorted(values)
    p95_index = max(0, -(-95 * len(ordered) // 100) - 1)
    return {
        "samples": [round(value, 9) for value in values],
        "minimum": round(min(values), 9),
        "median": round(statistics.median(values), 9),
        "p95": round(ordered[p95_index], 9),
    }


def compiler_provenance(build):
    """Extract optimization pass results from the compiler JSON payload."""
    try:
        payload = json.loads(build.stdout)
    except (TypeError, json.JSONDecodeError):
        return {"available": False}
    build_payload = payload.get("build", {})
    optimization = build_payload.get("optimization_report", {})
    native_payload = build_payload.get("native", {})
    if not optimization:
        sidecar = build_payload.get("sidecar") or native_payload.get("sidecar")
        if sidecar:
            routing_path = Path(sidecar) / "routing_table.txt"
            try:
                routing = json.loads(routing_path.read_text(encoding="utf-8"))
                optimization = routing.get("metal_optimization", {})
            except (OSError, json.JSONDecodeError):
                optimization = {}
    passes = optimization.get("passes", [])
    return {
        "available": bool(optimization),
        "passes": passes,
        "applied_passes": [item["name"] for item in passes if item.get("replacements", 0) > 0],
        "skipped_passes": [item["name"] for item in passes if item.get("replacements", 0) == 0],
    }


def benchmark_stream(samples, elements, iterations, warmups=3):
    directory = ROOT / "benchmarks" / "babelstream_core"
    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="mcu-benchmark-stream-") as temporary:
        temporary = Path(temporary)
        executable = temporary / "stream"
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(temporary / "cache")
        build, build_seconds = run(
            [
                sys.executable,
                str(MCU),
                "compile",
                str(directory / manifest["source"]),
                "-O3",
                "-o",
                str(executable),
                "--json",
            ],
            env=environment,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)
        runs = []
        for index in range(samples + warmups):
            metrics_path = temporary / f"metrics-{index}.json"
            run_environment = environment.copy()
            run_environment["MCU_METRICS_FILE"] = str(metrics_path)
            process, wall = run(
                [str(executable), str(elements), str(iterations)],
                env=run_environment,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    process.stderr or process.stdout or f"benchmark exited {process.returncode}"
                )
            output = fields(process.stdout)
            if output.get("ok") != "1":
                raise RuntimeError(f"benchmark correctness failure: {output}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if index >= warmups:
                runs.append({"wall_seconds": wall, "metrics": metrics})

    kernels = {}
    for kernel, streams in manifest["operations"].items():
        phase = f"gpu_execution.kernel.{kernel}"
        seconds = [item["metrics"]["phases"][phase]["seconds"] for item in runs]
        transferred = streams * elements * 4 * iterations
        kernels[kernel] = {
            "gpu_seconds": summary(seconds),
            "effective_gb_per_second": summary([transferred / value / 1.0e9 for value in seconds]),
            "bytes_per_iteration": streams * elements * 4,
        }
    return {
        **manifest,
        "ok": True,
        "parameters": {
            "elements": elements,
            "iterations": iterations,
            "samples": samples,
            "warmup_runs": warmups,
        },
        "build_seconds": round(build_seconds, 6),
        "compiler": compiler_provenance(build),
        "wall_seconds": summary([item["wall_seconds"] for item in runs]),
        "process_seconds": summary([item["metrics"]["process_seconds"] for item in runs]),
        "kernels": kernels,
    }


def benchmark_hotspot(samples, width, height, iterations, warmups=3):
    directory = ROOT / "benchmarks" / "hotspot_core"
    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="mcu-benchmark-hotspot-") as temporary:
        temporary = Path(temporary)
        executable = temporary / "hotspot"
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(temporary / "cache")
        build, build_seconds = run(
            [
                sys.executable,
                str(MCU),
                "compile",
                str(directory / manifest["source"]),
                "-O3",
                "-o",
                str(executable),
                "--json",
            ],
            env=environment,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)
        runs = []
        for index in range(samples + warmups):
            metrics_path = temporary / f"metrics-{index}.json"
            run_environment = environment.copy()
            run_environment["MCU_METRICS_FILE"] = str(metrics_path)
            process, wall = run(
                [str(executable), str(width), str(height), str(iterations)],
                env=run_environment,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    process.stderr or process.stdout or f"benchmark exited {process.returncode}"
                )
            output = fields(process.stdout)
            if output.get("ok") != "1":
                raise RuntimeError(f"benchmark correctness failure: {output}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if index >= warmups:
                runs.append({"wall_seconds": wall, "metrics": metrics, "output": output})
    phase = "gpu_execution.kernel.hotspot_step"
    seconds = [item["metrics"]["phases"][phase]["seconds"] for item in runs]
    updates = width * height * iterations
    transferred = updates * manifest["bytes_per_cell_update"]
    return {
        **manifest,
        "ok": True,
        "parameters": {
            "width": width,
            "height": height,
            "iterations": iterations,
            "samples": samples,
            "warmup_runs": warmups,
        },
        "build_seconds": round(build_seconds, 6),
        "compiler": compiler_provenance(build),
        "wall_seconds": summary([item["wall_seconds"] for item in runs]),
        "process_seconds": summary([item["metrics"]["process_seconds"] for item in runs]),
        "gpu_seconds": summary(seconds),
        "cell_updates_per_second": summary([updates / value for value in seconds]),
        "effective_gb_per_second": summary([transferred / value / 1.0e9 for value in seconds]),
        "max_error": max(float(item["output"]["max_error"]) for item in runs),
    }


def benchmark_reduction(samples, elements, iterations, blocks, warmups=3):
    directory = ROOT / "benchmarks" / "reduction_core"
    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="mcu-benchmark-reduction-") as temporary:
        temporary = Path(temporary)
        executable = temporary / "reduction"
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(temporary / "cache")
        build, build_seconds = run(
            [
                sys.executable,
                str(MCU),
                "compile",
                str(directory / manifest["source"]),
                "-O3",
                "-o",
                str(executable),
                "--json",
            ],
            env=environment,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)
        runs = []
        for index in range(samples + warmups):
            metrics_path = temporary / f"metrics-{index}.json"
            run_environment = environment.copy()
            run_environment["MCU_METRICS_FILE"] = str(metrics_path)
            process, wall = run(
                [str(executable), str(elements), str(iterations), str(blocks)],
                env=run_environment,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    process.stderr or process.stdout or f"benchmark exited {process.returncode}"
                )
            output = fields(process.stdout)
            if output.get("ok") != "1":
                raise RuntimeError(f"benchmark correctness failure: {output}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if index >= warmups:
                runs.append({"wall_seconds": wall, "metrics": metrics, "output": output})
    phase = "gpu_execution.kernel.dot_reduce"
    seconds = [item["metrics"]["phases"][phase]["seconds"] for item in runs]
    elements_processed = elements * iterations
    transferred = elements_processed * manifest["bytes_per_element"]
    operations = elements_processed * manifest["operations_per_element"]
    return {
        **manifest,
        "ok": True,
        "parameters": {
            "elements": elements,
            "iterations": iterations,
            "blocks": blocks,
            "samples": samples,
            "warmup_runs": warmups,
        },
        "build_seconds": round(build_seconds, 6),
        "compiler": compiler_provenance(build),
        "wall_seconds": summary([item["wall_seconds"] for item in runs]),
        "process_seconds": summary([item["metrics"]["process_seconds"] for item in runs]),
        "gpu_seconds": summary(seconds),
        "elements_per_second": summary([elements_processed / value for value in seconds]),
        "effective_gb_per_second": summary([transferred / value / 1.0e9 for value in seconds]),
        "gflops": summary([operations / value / 1.0e9 for value in seconds]),
        "maximum_relative_error": max(float(item["output"]["relative_error"]) for item in runs),
    }


def benchmark_histogram(samples, elements, iterations, blocks, warmups=3):
    directory = ROOT / "benchmarks" / "histogram_core"
    manifest = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="mcu-benchmark-histogram-") as temporary:
        temporary = Path(temporary)
        executable = temporary / "histogram"
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(temporary / "cache")
        build, build_seconds = run(
            [
                sys.executable,
                str(MCU),
                "compile",
                str(directory / manifest["source"]),
                "-O3",
                "-o",
                str(executable),
                "--json",
            ],
            env=environment,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)
        profiles = {}
        for profile, bins in manifest["contention_profiles"].items():
            runs = []
            for index in range(samples + warmups):
                metrics_path = temporary / f"metrics-{profile}-{index}.json"
                run_environment = environment.copy()
                run_environment["MCU_METRICS_FILE"] = str(metrics_path)
                process, wall = run(
                    [str(executable), str(elements), str(bins), str(iterations), str(blocks)],
                    env=run_environment,
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        process.stderr or process.stdout or f"benchmark exited {process.returncode}"
                    )
                output = fields(process.stdout)
                if output.get("ok") != "1":
                    raise RuntimeError(f"benchmark correctness failure: {output}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if index >= warmups:
                    runs.append({"wall_seconds": wall, "metrics": metrics, "output": output})
            phase = "gpu_execution.kernel.histogram_atomic"
            seconds = [item["metrics"]["phases"][phase]["seconds"] for item in runs]
            updates = elements * iterations
            profiles[profile] = {
                "bins": bins,
                "gpu_seconds": summary(seconds),
                "atomic_updates_per_second": summary([updates / value for value in seconds]),
                "process_seconds": summary([item["metrics"]["process_seconds"] for item in runs]),
                "wall_seconds": summary([item["wall_seconds"] for item in runs]),
                "verified_updates": min(int(item["output"]["updates"]) for item in runs),
            }
    return {
        **manifest,
        "ok": True,
        "parameters": {
            "elements": elements,
            "iterations": iterations,
            "blocks": blocks,
            "samples": samples,
            "warmup_runs": warmups,
        },
        "build_seconds": round(build_seconds, 6),
        "compiler": compiler_provenance(build),
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MCU's reproducible product benchmark suite")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--elements", type=int, default=8388608)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--only", choices=("all", "stream", "hotspot", "reduction", "histogram"), default="all"
    )
    parser.add_argument("--hotspot-size", type=int, default=1024)
    parser.add_argument("--hotspot-iterations", type=int, default=50)
    parser.add_argument("--reduction-elements", type=int, default=16777216)
    parser.add_argument("--reduction-iterations", type=int, default=20)
    parser.add_argument("--reduction-blocks", type=int, default=128)
    parser.add_argument("--histogram-elements", type=int, default=8388608)
    parser.add_argument("--histogram-iterations", type=int, default=10)
    parser.add_argument("--histogram-blocks", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmarks = []
    if args.only in {"all", "stream"}:
        benchmarks.append(
            benchmark_stream(
                max(1, args.samples),
                args.elements,
                args.iterations,
                max(1, args.warmups),
            )
        )
    if args.only in {"all", "hotspot"}:
        benchmarks.append(
            benchmark_hotspot(
                max(1, args.samples),
                args.hotspot_size,
                args.hotspot_size,
                args.hotspot_iterations,
                max(1, args.warmups),
            )
        )
    if args.only in {"all", "reduction"}:
        benchmarks.append(
            benchmark_reduction(
                max(1, args.samples),
                args.reduction_elements,
                args.reduction_iterations,
                args.reduction_blocks,
                max(1, args.warmups),
            )
        )
    if args.only in {"all", "histogram"}:
        benchmarks.append(
            benchmark_histogram(
                max(1, args.samples),
                args.histogram_elements,
                args.histogram_iterations,
                args.histogram_blocks,
                max(1, args.warmups),
            )
        )
    report = {
        "schema_version": 1,
        "suite": "mcu.product_benchmarks",
        "ok": True,
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "benchmarks": benchmarks,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
