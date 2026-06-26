import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


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


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 12 runtime API requires native runtime")
def test_device_properties_and_occupancy_runtime_only_program(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    int device = -1;
    cudaDeviceProp prop = {};
    int min_grid = 0;
    int block = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 10;
    if (count != 1) return 11;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0) return 12;
    if (cudaSetDevice(0) != cudaSuccess) return 13;
    if (cudaSetDevice(1) != cudaErrorInvalidDevice) return 14;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 15;
    if (prop.maxThreadsPerBlock != 1024 || prop.warpSize != 32) return 16;
    if (prop.unifiedAddressing != 1 || prop.integrated != 1) return 17;
    if (prop.name[0] == '\0') return 18;
    if (cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, nullptr, 0, 256) != cudaSuccess) return 19;
    if (min_grid < 1 || block != 256) return 20;
    std::printf("device=%s block=%d\n", prop.name, block);
    return 0;
}
''', name="device_props")
    result = _run_or_skip_no_metal(executable, environment)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "block=256" in result.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 12 events require native runtime")
def test_events_time_stream_work_and_async_copy(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(int* values) {
    values[threadIdx.x] += 3;
}

int main() {
    int input[64] = {};
    int output[64] = {};
    int* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float milliseconds = -1.0f;
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 11;
    if (cudaEventCreate(&start) != cudaSuccess) return 12;
    if (cudaEventCreateWithFlags(&stop, cudaEventBlockingSync) != cudaSuccess) return 13;
    if (cudaEventCreateWithFlags(&stop, cudaEventDisableTiming) != cudaErrorNotSupported) return 14;
    if (cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 15;
    if (cudaEventRecord(start, stream) != cudaSuccess) return 16;
    add<<<1, 64, 0, stream>>>(device);
    if (cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 17;
    if (cudaEventRecord(stop, stream) != cudaSuccess) return 18;
    if (cudaEventSynchronize(stop) != cudaSuccess) return 19;
    if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess) return 20;
    if (milliseconds < 0.0f) return 21;
    for (int index = 0; index < 64; ++index) if (output[index] != 3) return 22;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device);
    std::printf("event_ms=%0.4f\n", milliseconds);
    return 0;
}
''', name="events")
    result = _run_or_skip_no_metal(executable, environment)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "event_ms=" in result.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 12 error API requires native runtime")
def test_peek_last_error_get_last_error_and_zero_size_operations(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int value = 7;
    if (cudaMemcpy(&value, &value, 0, cudaMemcpyHostToHost) != cudaSuccess) return 10;
    if (cudaMemset(nullptr, 0, 0) != cudaSuccess) return 11;
    if (cudaSetDevice(7) != cudaErrorInvalidDevice) return 12;
    if (cudaPeekAtLastError() != cudaErrorInvalidDevice) return 13;
    if (cudaPeekAtLastError() != cudaErrorInvalidDevice) return 14;
    if (cudaGetLastError() != cudaErrorInvalidDevice) return 15;
    if (cudaGetLastError() != cudaSuccess) return 16;
    if (cudaDeviceReset() != cudaSuccess) return 17;
    std::printf("error_api_ok\n");
    return 0;
}
''', name="errors")
    result = _run_or_skip_no_metal(executable, environment)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "error_api_ok" in result.stdout
