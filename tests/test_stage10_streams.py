import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
MULTILAUNCH = ROOT / "examples" / "stage10_multilaunch" / "main.cu"


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


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 10 streams require Metal")
def test_easy_stream_ordering_and_async_copies(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
__global__ void add(int* values, int amount) { values[threadIdx.x] += amount; }
int main() {
    int input[64] = {};
    int output[64] = {};
    int* device = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 11;
    if (cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 12;
    for (int index = 0; index < 25; ++index) add<<<1, 64, 0, stream>>>(device, 2);
    if (cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 13;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 14;
    for (int index = 0; index < 64; ++index) if (output[index] != 50) return 20;
    if (cudaStreamDestroy(stream) != cudaSuccess) return 15;
    if (cudaStreamDestroy(stream) == cudaSuccess) return 21;
    cudaFree(device);
    std::printf("easy_stream=%d\n", output[0]);
    return 0;
}
''')
    result = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "easy_stream=50" in result.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 10 streams require Metal")
def test_medium_independent_streams_device_wait_and_safe_free(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
__global__ void add(int* values, int amount) { values[threadIdx.x] += amount; }
int main() {
    int zero[32] = {};
    int output_a[32] = {};
    int output_b[32] = {};
    int* a = nullptr;
    int* b = nullptr;
    int* temporary = nullptr;
    cudaStream_t first = nullptr;
    cudaStream_t second = nullptr;
    cudaMalloc(&a, sizeof(zero));
    cudaMalloc(&b, sizeof(zero));
    cudaMalloc(&temporary, sizeof(zero));
    cudaStreamCreate(&first);
    cudaStreamCreate(&second);
    cudaMemcpyAsync(a, zero, sizeof(zero), cudaMemcpyHostToDevice, first);
    cudaMemcpyAsync(b, zero, sizeof(zero), cudaMemcpyHostToDevice, second);
    for (int index = 0; index < 40; ++index) add<<<1, 32, 0, first>>>(a, 1);
    for (int index = 0; index < 30; ++index) add<<<1, 32, 0, second>>>(b, 2);
    cudaMemcpyAsync(output_a, a, sizeof(output_a), cudaMemcpyDeviceToHost, first);
    cudaMemcpyAsync(output_b, b, sizeof(output_b), cudaMemcpyDeviceToHost, second);
    add<<<1, 32, 0, first>>>(temporary, 1);
    if (cudaFree(temporary) != cudaSuccess) return 10;
    if (cudaDeviceSynchronize() != cudaSuccess) return 11;
    for (int index = 0; index < 32; ++index) {
        if (output_a[index] != 40 || output_b[index] != 60) return 20;
    }
    cudaStreamDestroy(second);
    cudaStreamDestroy(first);
    cudaFree(b);
    cudaFree(a);
    std::printf("medium_streams=%d,%d\n", output_a[0], output_b[0]);
    return 0;
}
''')
    metrics_path = tmp_path / "medium-metrics.json"
    environment["MCU_METRICS_FILE"] = str(metrics_path)
    result = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "medium_streams=40,60" in result.stdout
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["counters"]["streams_created"] == 2
    assert metrics["counters"]["command_buffers_submitted"] <= 10
    assert metrics["counters"]["kernel_batches_submitted"] >= 3
    assert "gpu_execution.kernel.batch" not in metrics["phases"]
    assert metrics["counters"]["async_copy_bytes"] > 0


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 10 benchmark requires Metal")
def test_hard_multilaunch_is_exact_and_batches_waits(tmp_path):
    executable = tmp_path / "multilaunch"
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    build = subprocess.run(
        [sys.executable, str(MCU), "compile", str(MULTILAUNCH), "-O2", "-o", str(executable), "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    metrics_path = tmp_path / "hard-metrics.json"
    environment["MCU_METRICS_FILE"] = str(metrics_path)
    result = subprocess.run([str(executable), "100"], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["result"] == "100"
    assert float(values["speedup"]) > 1.0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["counters"]["kernel_dispatches"] == 201
    assert metrics["counters"]["command_buffers_submitted"] < metrics["counters"]["kernel_dispatches"]
    assert 101 <= metrics["counters"]["kernel_batches_submitted"] <= 102


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 10 stress test requires Metal")
def test_stress_default_ordering_repeated_stream_lifecycle_and_h2h_rejection(tmp_path):
    executable, environment = _build(tmp_path, r'''
#include <cstdio>
__global__ void add(int* values, int amount) { values[threadIdx.x] += amount; }
int main() {
    int zero[32] = {};
    int output[32] = {};
    int* device = nullptr;
    cudaMalloc(&device, sizeof(zero));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(device, zero, sizeof(zero), cudaMemcpyHostToDevice, stream);
    add<<<1, 32>>>(device, 2);
    add<<<1, 32, 0, stream>>>(device, 3);
    cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 10;
    for (int index = 0; index < 32; ++index) if (output[index] != 5) return 20;
    cudaStreamDestroy(stream);

    for (int iteration = 0; iteration < 50; ++iteration) {
        cudaStream_t temporary = nullptr;
        if (cudaStreamCreate(&temporary) != cudaSuccess) return 11;
        add<<<1, 32, 0, temporary>>>(device, 1);
        if (cudaStreamDestroy(temporary) != cudaSuccess) return 12;
    }
    if (cudaMemcpyAsync(output, zero, sizeof(zero), cudaMemcpyHostToHost, nullptr) == cudaSuccess) return 21;
    cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost);
    cudaFree(device);
    for (int index = 0; index < 32; ++index) if (output[index] != 55) return 22;
    std::printf("stress_stream=%d\n", output[0]);
    return 0;
}
''', name="stress")
    result = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "stress_stream=55" in result.stdout


def test_four_value_cuda_launch_configuration_lowers_stream():
    from core.host_transpiler import transform_host_source

    transformed = transform_host_source(
        "void run(int* data, cudaStream_t stream) { work<<<2, 64, 128, stream>>>(data); }"
    )
    assert "cudaLaunchSharedOnStream(work, 2, 64, 128, stream, data);" in transformed


def test_cuda_checker_contains_ordered_tiered_cases():
    cases = ROOT / "tests" / "cuda_suite" / "cases"
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cases.glob("*.json"))]
    assert [item["id"] for item in manifests] == [f"{index:02d}" for index in range(1, 27)]
    assert {item["tier"] for item in manifests} == {"easy", "medium", "hard", "stress"}
    assert all((ROOT / item["program"]).is_file() for item in manifests)
