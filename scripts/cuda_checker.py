#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
CASES = ROOT / "tests" / "cuda_suite" / "cases"


def execute(command, **kwargs):
    started = perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)
    return process, perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description="Run the MCU CUDA easy-to-stress checker")
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "stress", "all"], default="all")
    parser.add_argument("--case", help="Run one case id, for example 07")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mcu_cuda_checker.json"))
    args = parser.parse_args()
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(CASES.glob("*.json"))]
    if args.tier != "all":
        manifests = [item for item in manifests if item["tier"] == args.tier]
    if args.case:
        manifests = [item for item in manifests if item["id"] == args.case]
    if not manifests:
        parser.error("no CUDA checker cases matched")

    report = {"schema_version": 1, "cases": [], "ok": True}
    with tempfile.TemporaryDirectory(prefix="mcu-cuda-checker-") as temporary:
        directory = Path(temporary)
        environment = os.environ.copy()
        environment["MCU_CACHE_DIR"] = str(directory / "cache")
        executables = {}
        for manifest in manifests:
            program = manifest["program"]
            if program not in executables:
                source = ROOT / program
                isolated = directory / "sources" / Path(program).stem
                isolated.mkdir(parents=True, exist_ok=True)
                source_copy = isolated / "main.cu"
                shutil.copy2(source, source_copy)
                executable = directory / Path(program).stem
                build, build_seconds = execute(
                    [sys.executable, str(MCU), "compile", str(source_copy), "-O2", "-o", str(executable), "--json"],
                    cwd=ROOT,
                    env=environment,
                )
                if build.returncode != 0:
                    print(f"FAIL build {program}")
                    print(build.stderr or build.stdout)
                    return 1
                executables[program] = (executable, build_seconds)
            executable, build_seconds = executables[program]
            metrics_path = directory / f"{manifest['id']}-metrics.json"
            run_environment = environment.copy()
            run_environment["MCU_METRICS_FILE"] = str(metrics_path)
            try:
                process, run_seconds = execute(
                    [str(executable), *manifest.get("args", [])],
                    env=run_environment,
                    timeout=manifest.get("timeout_seconds", 30),
                )
            except subprocess.TimeoutExpired as exc:
                process = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or "", "case timed out")
                run_seconds = float(manifest.get("timeout_seconds", 30))
            expected = manifest.get("expected", "status=pass")
            passed = process.returncode == 0 and expected in process.stdout
            print(f"{'PASS' if passed else 'FAIL'} {manifest['id']} [{manifest['tier']}] {manifest['name']} ({run_seconds:.3f}s)")
            result = {
                **manifest,
                "passed": passed,
                "exit_code": process.returncode,
                "build_seconds": round(build_seconds, 6),
                "run_seconds": round(run_seconds, 6),
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
            if metrics_path.is_file():
                result["runtime_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            report["cases"].append(result)
            report["ok"] = report["ok"] and passed

    report["passed"] = sum(item["passed"] for item in report["cases"])
    report["total"] = len(report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{report['passed']}/{report['total']} passed; report: {args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
