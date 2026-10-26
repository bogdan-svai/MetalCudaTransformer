#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
SOURCE = ROOT / "examples" / "multilaunch" / "main.cu"


def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)


def fields(output):
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def verify(samples):
    tiers = {
        "easy": (10, 4096, 0),
        "medium": (100, 4096, 0),
        "hard": (1000, 4096, 0),
        "large": (1, 1000000, 1),
    }
    with tempfile.TemporaryDirectory(prefix="mcu-streams-") as temporary:
        temporary_path = Path(temporary)
        executable = temporary_path / "multilaunch"
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(temporary_path / "cache")
        build = run(
            [sys.executable, str(MCU), "compile", str(SOURCE), "-O2", "-o", str(executable), "--json"],
            cwd=ROOT,
            env=environment,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)
        results = {}
        device = "unknown"
        for tier, (launches, count, launch_only) in tiers.items():
            runs = []
            for sample in range(samples):
                metrics_path = temporary_path / f"{tier}-{sample}.json"
                run_environment = environment.copy()
                run_environment["MCU_METRICS_FILE"] = str(metrics_path)
                process = run(
                    [str(executable), str(launches), str(count), str(launch_only)],
                    env=run_environment,
                )
                if process.returncode != 0:
                    raise RuntimeError(process.stderr or process.stdout)
                values = fields(process.stdout)
                if int(values["result"]) != launches:
                    raise RuntimeError(f"{tier} produced {values['result']} instead of {launches}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                device = metrics["device"]
                runs.append({
                    "synchronous_us": float(values["synchronous_us"]),
                    "batched_us": float(values["batched_us"]),
                    "speedup": float(values["speedup"]),
                    "runtime_metrics": metrics,
                })
            results[tier] = {
                "launches": launches,
                "element_count": count,
                "launch_only": bool(launch_only),
                "runs": runs,
                "median_speedup": statistics.median(item["speedup"] for item in runs),
                "minimum_speedup": min(item["speedup"] for item in runs),
            }
        medium_pass = results["medium"]["median_speedup"] >= 1.25
        large_pass = results["large"]["median_speedup"] >= (1.0 / 1.10)
        return {
            "schema_version": 2,
            "suite": "streams",
            "ok": medium_pass and large_pass,
            "machine": {
                "system": platform.system(),
                "machine": platform.machine(),
                "release": platform.release(),
                "device": device,
            },
            "tiers": results,
            "budgets": {
                "medium_minimum_speedup": 1.25,
                "medium_pass": medium_pass,
                "large_maximum_regression": 0.10,
                "large_pass": large_pass,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Collect MCU scheduler baselines")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(max(1, args.samples))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
