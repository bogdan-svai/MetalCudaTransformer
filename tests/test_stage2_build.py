import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def run_build(tmp_path, source):
    target = tmp_path / "case.cu"
    target.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(MCU), "build", str(target), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def read_build_artifacts():
    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        names = set(archive.namelist())
        metal = archive.read("generated.metal").decode("utf-8")
        routing = json.loads(archive.read("routing_table.txt").decode("utf-8"))
        ast = json.loads(archive.read("ast_output.txt").decode("utf-8"))
    return names, metal, routing, ast


def test_build_generates_msl_signature_body_and_routing_table(tmp_path):
    result, payload = run_build(
        tmp_path,
        """
        __global__ void scale(float* values, int count, float factor) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < count) {
                values[idx] = values[idx] * factor;
            }
        }
        void host() {
            cudaLaunch(scale, 4, 256, 0, 16, 2.0f);
        }
        """,
    )

    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["build"]["translated_kernel_count"] == 1

    names, metal, routing, ast = read_build_artifacts()
    assert names == {"ast_output.txt", "generated.metal", "routing_table.txt"}
    assert ast["command"] == "build"
    assert "kernel void scale_msl(" in metal
    assert "device float* values [[buffer(0)]]" in metal
    assert "constant int& count [[buffer(1)]]" in metal
    assert "constant float& factor [[buffer(2)]]" in metal
    assert "uint3 thread_position_in_grid [[thread_position_in_grid]]" in metal
    assert "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]" in metal
    assert "uint3 threads_per_threadgroup [[threads_per_threadgroup]]" in metal
    assert "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]" in metal
    assert "thread_position_in_threadgroup.x" in metal
    assert "threadgroup_position_in_grid.x" in metal
    assert "threads_per_threadgroup.x" in metal
    assert "threadIdx" not in metal
    assert "blockIdx" not in metal
    assert "blockDim" not in metal

    table = routing["translated_kernels"]["scale"]
    assert table["msl_entry_point"] == "scale_msl"
    assert table["argument_count"] == 3
    assert table["bindings"] == [
        {
            "argument_index": 0,
            "identifier": "values",
            "type": "float*",
            "binding_target": "buffer(0)",
            "space": "device",
        },
        {
            "argument_index": 1,
            "identifier": "count",
            "type": "int",
            "binding_target": "buffer(1)",
            "space": "constant",
        },
        {
            "argument_index": 2,
            "identifier": "factor",
            "type": "float",
            "binding_target": "buffer(2)",
            "space": "constant",
        },
    ]


def test_build_aborts_on_stage1_error_without_codegen(tmp_path):
    result, payload = run_build(
        tmp_path,
        """
        __global__ void bad(float* values) {
            values[threadIdx.x] = 1.0f;
        }
        void host() {
            cudaLaunch(bad, 1, dim3(64, 32), 0);
        }
        """,
    )

    assert result.returncode == 1
    assert payload["ok"] is False
    assert "build" not in payload
    assert any(item["code"] == "HardwareLimitExceeded" for item in payload["diagnostics"])

    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        assert archive.namelist() == ["ast_output.txt"]


def test_syncthreads_is_translated_to_threadgroup_barrier(tmp_path):
    result, _payload = run_build(
        tmp_path,
        """
        __global__ void sync_kernel(float* values) {
            __syncthreads();
            values[threadIdx.x] = 1.0f;
        }
        void host() {
            cudaLaunch(sync_kernel, 1, 32, 0);
        }
        """,
    )

    assert result.returncode == 0
    _names, metal, _routing, _ast = read_build_artifacts()
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in metal
    assert "__syncthreads" not in metal
