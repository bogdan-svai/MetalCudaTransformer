import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def run_command(tmp_path, command, block, *extra):
    target = tmp_path / "coarsening.cu"
    target.write_text(
        f"""
        __global__ void work(float* values, int count) {{
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < count) {{
                values[idx] = values[idx] + 1.0f;
            }}
        }}
        void host() {{
            cudaLaunch(work, 1, {block}, 0, 8192);
        }}
        """,
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(MCU), command, str(target), "--json", *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def read_artifacts():
    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        metal = archive.read("generated.metal").decode("utf-8")
        routing = json.loads(archive.read("routing_table.txt").decode("utf-8"))
    return metal, routing


def test_check_remains_strict_for_oversized_block(tmp_path):
    result, payload = run_command(tmp_path, "check", "2048")

    assert result.returncode == 1
    assert any(item["code"] == "HardwareLimitExceeded" for item in payload["diagnostics"])


def test_build_auto_coarsens_2048_to_1024(tmp_path):
    result, payload = run_command(tmp_path, "build", "2048")

    assert result.returncode == 0
    plan = payload["build"]["coarsening"]["plans"]["work"]
    assert plan == {
        "mode": "auto",
        "axis": "x",
        "coarsening_factor": 2,
        "original_block": [2048, 1, 1],
        "physical_block": [1024, 1, 1],
    }

    metal, routing = read_artifacts()
    assert "mcu_step < 2" in metal
    assert "mcu_thread_idx.x" in metal
    assert "mcu_block_dim.x" in metal
    assert "threadIdx" not in metal
    metrics = routing["translated_kernels"]["work"]["optimization_metrics"]
    assert metrics == plan


def test_build_auto_coarsens_4096_with_factor_4(tmp_path):
    result, payload = run_command(tmp_path, "build", "4096")

    assert result.returncode == 0
    plan = payload["build"]["coarsening"]["plans"]["work"]
    assert plan["coarsening_factor"] == 4
    assert plan["physical_block"] == [1024, 1, 1]


def test_manual_coarsening_factor_4(tmp_path):
    result, payload = run_command(tmp_path, "build", "256", "--coarsening", "4")

    assert result.returncode == 0
    plan = payload["build"]["coarsening"]["plans"]["work"]
    assert plan["mode"] == "manual"
    assert plan["physical_block"] == [64, 1, 1]

    logical_indices = {
        physical_x + step * plan["physical_block"][0]
        for physical_x in range(plan["physical_block"][0])
        for step in range(plan["coarsening_factor"])
    }
    assert logical_indices == set(range(256))


def test_factor_1_preserves_non_coarsened_body(tmp_path):
    result, payload = run_command(tmp_path, "build", "256", "--coarsening", "1")

    assert result.returncode == 0
    plan = payload["build"]["coarsening"]["plans"]["work"]
    assert plan["coarsening_factor"] == 1
    metal, _routing = read_artifacts()
    assert "mcu_step" not in metal
    assert "thread_position_in_threadgroup.x" in metal


def test_non_divisible_manual_factor_fails(tmp_path):
    result, payload = run_command(tmp_path, "build", "250", "--coarsening", "4")

    assert result.returncode == 1
    assert any(item["code"] == "NonDivisibleCoarseningFactor" for item in payload["diagnostics"])


def test_invalid_factor_is_cli_error(tmp_path):
    target = tmp_path / "case.cu"
    target.write_text("__global__ void k() {}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(MCU), "build", str(target), "--coarsening", "3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
