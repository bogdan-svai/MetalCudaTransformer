#!/usr/bin/env python3
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
SOURCE = ROOT / "examples" / "cublas" / "main.cu"


def run(command, env):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    report = {
        "schema_version": 2,
        "suite": "cublas",
        "ok": False,
        "platform": platform.system(),
        "cases": [],
    }
    if platform.system() != "Darwin":
        report["ok"] = True
        report["skipped"] = "native cuBLAS adapter requires macOS Metal"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    with tempfile.TemporaryDirectory(prefix="mcu-cublas-") as temporary:
        temp = Path(temporary)
        executable = temp / "cublas"
        env = os.environ.copy()
        env["MCU_CACHE_DIR"] = str(temp / "cache")
        build = run([sys.executable, str(MCU), "compile", str(SOURCE), "-O2", "-o", str(executable), "--json"], env)
        report["build"] = {
            "returncode": build.returncode,
            "stdout": build.stdout[-4000:],
            "stderr": build.stderr[-4000:],
        }
        if build.returncode != 0:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1

        for size in (128, 2048):
            metrics = temp / f"metrics-{size}.json"
            run_env = env.copy()
            run_env["MCU_METRICS_FILE"] = str(metrics)
            run_env["MCU_RUNTIME_DIAGNOSTICS"] = "1"
            result = run([str(executable), str(size)], run_env)
            case = {
                "size": size,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            if result.returncode == 10 and "No Metal device is available" in result.stderr:
                case["skipped"] = "Metal device is not available in this execution environment"
                report["cases"].append(case)
                report["ok"] = True
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            if metrics.is_file():
                case["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
            report["cases"].append(case)
            if result.returncode != 0:
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1
            counters = case.get("metrics", {}).get("counters", {})
            if counters.get("cublas_sgemm_backend_mps") != 1:
                case["error"] = "SGEMM did not use MPS backend"
                print(json.dumps(report, indent=2, sort_keys=True))
                return 1

    report["ok"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
