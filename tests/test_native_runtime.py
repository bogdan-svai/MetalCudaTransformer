import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
SOURCE = ROOT / "examples" / "numeric" / "main.cu"


@pytest.mark.skipif(platform.system() != "Darwin", reason="native runtime requires macOS Metal")
def test_native_runtime_matches_cpu_and_reports_allocation_latency(tmp_path):
    output = tmp_path / "native"
    build = subprocess.run(
        [
            sys.executable,
            str(MCU),
            "build",
            str(SOURCE),
            "--native",
            "--output",
            str(output),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert build.returncode == 0, build.stderr or build.stdout
    payload = json.loads(build.stdout)
    plan = payload["build"]["coarsening"]["plans"]["numericTransform"]
    assert plan["coarsening_factor"] == 4
    assert plan["original_block"] == [2200, 1, 1]
    assert plan["physical_block"] == [550, 1, 1]
    executable = Path(payload["build"]["native"]["executable"])
    assert executable.is_file()
    assert (output / "generated.metal").is_file()
    assert (output / "routing_table.txt").is_file()

    run = subprocess.run([str(executable)], text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr or run.stdout
    assert "numeric_count=100000" in run.stdout
    assert "max_error=0" in run.stdout
    assert "allocation_p95_us=" in run.stdout
