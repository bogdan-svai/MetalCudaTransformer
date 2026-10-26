import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest

from core.registry.library_adapters import default_library_registry


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def _build(tmp_path, source, name="case"):
    target = tmp_path / f"{name}.cu"
    executable = tmp_path / name
    target.write_text(source, encoding="ascii")
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, str(MCU), "compile", str(target), "-O2", "-o", str(executable), "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return executable, environment


def _run_or_skip_no_metal(executable, environment):
    environment = environment.copy()
    environment["MCU_RUNTIME_DIAGNOSTICS"] = "1"
    result = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    if result.returncode == 10 and "No Metal device is available" in result.stderr:
        pytest.skip("Metal device is not available in this execution environment")
    return result


def test_library_adapter_registry_tracks_supported_and_planned_libraries():
    registry = default_library_registry()
    manifest = registry.to_manifest()

    assert "cublas_v2.h" in registry.supported_libraries()
    assert "curand.h" in registry.unsupported_libraries()
    assert any(
        item["library"] == "cublas_v2.h"
        and item["symbol"] == "cublasSgemm"
        and item["backend"] == "mps_matrix_multiplication"
        for item in manifest
    )


@pytest.mark.skipif(platform.system() != "Darwin", reason="cuBLAS shim requires native runtime")
def test_cublas_sgemm_matches_column_major_cpu_reference_and_metrics(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>

int main() {
    const int m = 2;
    const int n = 2;
    const int k = 3;
    float host_a[m * k] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    float host_b[k * n] = {
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    float host_c[m * n] = {
        1.0f, 1.0f,
        1.0f, 1.0f
    };
    float output[m * n] = {};
    float expected[m * n] = {};
    const float alpha = 2.0f;
    const float beta = 0.5f;

    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += host_a[row + inner * m] * host_b[inner + column * k];
            }
            expected[row + column * m] = alpha * sum + beta * host_c[row + column * m];
        }
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    if (cudaMalloc(&device_a, sizeof(host_a)) != cudaSuccess) return 10;
    if (cudaMalloc(&device_b, sizeof(host_b)) != cudaSuccess) return 11;
    if (cudaMalloc(&device_c, sizeof(host_c)) != cudaSuccess) return 12;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 13;
    if (cudaMemcpyAsync(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 14;
    if (cudaMemcpyAsync(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 15;
    if (cudaMemcpyAsync(device_c, host_c, sizeof(host_c), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 16;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 17;
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return 18;
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, device_a, m, device_b, k, &beta, device_c, m) != CUBLAS_STATUS_SUCCESS) return 19;
    if (cudaMemcpyAsync(output, device_c, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 20;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 21;
    for (int index = 0; index < m * n; ++index) {
        if (std::fabs(output[index] - expected[index]) > 1e-5f) {
            std::printf("bad index=%d got=%f expected=%f\n", index, output[index], expected[index]);
            return 30;
        }
    }
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) return 22;
    cudaStreamDestroy(stream);
    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    std::printf("sgemm=%0.1f,%0.1f,%0.1f,%0.1f\n", output[0], output[1], output[2], output[3]);
    return 0;
}
''')
    metrics_path = tmp_path / "metrics.json"
    environment["MCU_METRICS_FILE"] = str(metrics_path)
    result = _run_or_skip_no_metal(executable, environment)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "sgemm=152.5,200.5,206.5,272.5" in result.stdout
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["counters"]["cublas_calls"] == 1
    assert metrics["counters"]["cublas_sgemm_elements"] == 4
    assert metrics["counters"]["cublas_sgemm_backend_mps"] == 1
    assert metrics["phases"]["cublas_sgemm"]["count"] == 1
    assert metrics["phases"]["cublas_adapter_setup"]["count"] == 1


@pytest.mark.skipif(platform.system() != "Darwin", reason="cuBLAS shim requires native runtime")
def test_cublas_reports_unsupported_modes(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
#include <cublas_v2.h>

int main() {
    float alpha = 1.0f;
    float beta = 0.0f;
    float* data = nullptr;
    cublasHandle_t handle = nullptr;
    if (cudaMalloc(&data, 4 * sizeof(float)) != cudaSuccess) return 10;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 11;
    if (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_NOT_SUPPORTED) return 12;
    if (cublasSgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, 1,
                    &alpha, data, 1, data, 1, &beta, data, 1) != CUBLAS_STATUS_NOT_SUPPORTED) return 13;
    cublasDestroy(handle);
    cudaFree(data);
    std::printf("unsupported_ok\n");
    return 0;
}
''', name="unsupported")
    result = _run_or_skip_no_metal(executable, environment)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "unsupported_ok" in result.stdout
