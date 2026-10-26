#!/usr/bin/env python3
"""Measure frontend cache misses and hits in separate compiler processes."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_output(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def measure(target, cache, directory, expect_hit):
    environment = dict(os.environ, MCU_CACHE_DIR=str(cache), MCU_SEMANTIC_STRICT="1")
    process = subprocess.run(
        [sys.executable, str(ROOT / "mcu"), "check", str(target), "--json"],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    report = json.loads(process.stdout)
    events = [e for e in report["timings"]["cache_events"] if e["layer"] == "semantic"]
    if not events or any(e["hit"] != expect_hit for e in events):
        raise RuntimeError(f"Expected semantic cache hit={expect_hit}, observed {events}")
    return report["timings"]["phases_seconds"]["frontend"] * 1000.0, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "examples/sha256/app/main.cpp")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups must be nonnegative")
    target = args.path.resolve()
    if not target.exists():
        parser.error(f"Input does not exist: {target}")
    cold, warm = [], []
    with tempfile.TemporaryDirectory(prefix="mcu-frontend-") as temporary:
        directory = Path(temporary)
        for sample in range(args.warmups + args.samples):
            cache = directory / f"cache-{sample}"
            cold_ms, report = measure(target, cache, directory, False)
            warm_ms, _ = measure(target, cache, directory, True)
            if sample >= args.warmups:
                cold.append(cold_ms)
                warm.append(warm_ms)
    sources = [Path(path) for path in report["metadata"]["sources"]]
    cold_median, warm_median = statistics.median(cold), statistics.median(warm)
    result = {
        "schema_version": 1,
        "suite": "frontend_cache",
        "ok": True,
        "measurement": "mcu check frontend phase; cache miss versus cache hit",
        "samples": args.samples,
        "warmup_pairs": args.warmups,
        "cold_ms": cold,
        "warm_ms": warm,
        "cold_median_ms": cold_median,
        "warm_median_ms": warm_median,
        "reduction_percent": 100.0 * (cold_median - warm_median) / cold_median,
        "machine": {
            "model": (
                command_output(["sysctl", "-n", "hw.model"]) if sys.platform == "darwin" else None
            ),
            "chip": (
                command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                if sys.platform == "darwin"
                else None
            ),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "xcode": (
                command_output(["xcodebuild", "-version"]) if sys.platform == "darwin" else None
            ),
        },
        "revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "working_tree_status": command_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
