#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "sha256_native"


def main():
    parser = argparse.ArgumentParser(description="Benchmark specialized native Metal SHA-256")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mcu-native-sha256-") as temporary:
        temporary = Path(temporary)
        air, library, executable = (
            temporary / "native.air",
            temporary / "native.metallib",
            temporary / "native",
        )
        commands = [
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(SOURCE / "native.metal"),
                "-o",
                str(air),
            ],
            ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)],
            [
                "xcrun",
                "clang++",
                "-fobjc-arc",
                str(SOURCE / "main.mm"),
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-o",
                str(executable),
            ],
        ]
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                raise SystemExit(result.stderr or result.stdout)
        environment = os.environ.copy()
        environment["MCU_NATIVE_METALLIB"] = str(library)
        result = subprocess.run(
            [str(executable), str(args.samples), str(args.warmups)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise SystemExit(result.stderr or result.stdout)
    seconds = [
        float(line.split("=", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("gpu_seconds=")
    ]
    fields = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line and not line.startswith("gpu_seconds=")
    )
    payload = {
        "schema_version": 1,
        "ok": len(seconds) == args.samples,
        "benchmark": json.loads((SOURCE / "benchmark.json").read_text()),
        "device": fields.get("device"),
        "nonce": int(fields["nonce"]),
        "gpu_seconds": {
            "minimum": min(seconds),
            "median": statistics.median(seconds),
            "maximum": max(seconds),
            "samples": seconds,
        },
        "hashes_per_second": 500000 / statistics.median(seconds),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
