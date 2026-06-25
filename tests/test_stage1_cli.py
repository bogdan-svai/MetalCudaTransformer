import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def run_mcu(tmp_path, source, *args):
    target = tmp_path / "case.cu"
    target.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(MCU), "check", str(target), "--json", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def test_valid_kernel_generates_artifact(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """
        __global__ void add(float* a) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            a[idx] = a[idx] + 1.0f;
        }
        void host() {
            cudaLaunch(add, 4, 256, 0);
        }
        """,
    )

    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["metadata"]["kernels"][0]["name"] == "add"
    assert payload["metadata"]["launches"][0]["block"]["total"] == 256

    artifact = ROOT / "mcu_artifacts.zip"
    assert artifact.exists()
    with zipfile.ZipFile(artifact) as archive:
        assert archive.namelist() == ["ast_output.txt"]
        archived = json.loads(archive.read("ast_output.txt").decode("utf-8"))
    assert archived["command"] == "check"


def test_hardware_limit_exceeded_returns_exit_code_1(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """
        __global__ void too_many(float* a) {
            a[threadIdx.x] = 1.0f;
        }
        void host() {
            cudaLaunch(too_many, 1, dim3(33, 32), 0);
        }
        """,
    )

    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["code"] == "HardwareLimitExceeded"


def test_supported_cublas_header_is_accepted(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """
        #include <cublas_v2.h>
        __global__ void k(float* a) { a[threadIdx.x] = 0; }
        void host() { cudaLaunch(k, 1, 32, 0); }
        """,
    )

    assert result.returncode == 0
    assert payload["ok"] is True


def test_unsupported_cuda_library_fails(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """
        #include <curand.h>
        __global__ void k(float* a) { a[threadIdx.x] = 0; }
        void host() { cudaLaunch(k, 1, 32, 0); }
        """,
    )

    assert result.returncode == 1
    assert any(item["code"] == "UnsupportedExternalLibrary" for item in payload["diagnostics"])


def test_divergent_barrier_is_rejected(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """
        __global__ void barrier(float* a) {
            int idx = threadIdx.x;
            if (idx < 16) {
                __syncthreads();
            }
            a[idx] = 1.0f;
        }
        void host() { cudaLaunch(barrier, 1, 32, 0); }
        """,
    )

    assert result.returncode == 1
    assert any(item["code"] == "DivergentThreadgroupBarrier" for item in payload["diagnostics"])


def test_data_dependent_and_early_exit_barriers_are_rejected(tmp_path):
    for body in [
        "if (values[0] != 0) { __syncthreads(); }",
        "if (threadIdx.x == 0) return; __syncthreads();",
        "if (threadIdx.x < 16) __syncthreads();",
    ]:
        result, payload = run_mcu(
            tmp_path,
            f"""__global__ void unsafe_barrier(int* values) {{ {body} }}
void host() {{ cudaLaunch(unsafe_barrier, 1, 32, 0); }}
""",
        )
        assert result.returncode == 1
        assert any(item["code"] == "DivergentThreadgroupBarrier" for item in payload["diagnostics"])


def test_uniform_scalar_condition_may_guard_barrier(tmp_path):
    result, payload = run_mcu(
        tmp_path,
        """__global__ void uniform_barrier(int* values, int enabled) {
    if (enabled) { __syncthreads(); }
    values[threadIdx.x] = enabled;
}
void host() { cudaLaunch(uniform_barrier, 1, 32, 0, 1); }
""",
    )
    assert result.returncode == 0
    assert not any(item["code"] == "DivergentThreadgroupBarrier" for item in payload["diagnostics"])


def test_invalid_path_returns_exit_code_2(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MCU), "check", str(tmp_path / "missing.cu"), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["diagnostics"][0]["code"] == "InvalidTargetPath"
