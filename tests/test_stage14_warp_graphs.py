import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def _json(*arguments):
    result = subprocess.run(
        [sys.executable, str(MCU), *arguments, "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def test_capabilities_report_stage14_warp_and_graph_surface():
    result, payload = _json("capabilities")

    assert result.returncode == 0
    assert payload["mcu_version"] == "0.14.0"
    assert payload["runtime_abi"] >= 6

    warp = {item["name"]: item for item in payload["warp_intrinsics"]}
    assert warp["__shfl_sync"]["status"] == "partial"
    assert warp["__syncwarp"]["stage"] == "14B"
    assert warp["__ballot_sync"]["status"] == "planned"

    graphs = {item["name"]: item for item in payload["cuda_graphs"]}
    assert graphs["explicit_kernel_graph_nodes"]["status"] == "partial"
    assert graphs["explicit_kernel_graph_nodes"]["supported_node_kinds"] == ["kernel", "memcpy_1d", "memset_1d"]
    assert "stream_capture" in graphs

    runtime = {item["name"]: item for item in payload["runtime_apis"]}
    assert runtime["cudaGraphAddKernelNode"]["status"] == "partial"
    assert runtime["cudaGraphAddMemcpyNode"]["status"] == "partial"
    assert runtime["cudaGraphAddMemsetNode"]["status"] == "partial"
    assert runtime["cudaStreamBeginCapture"]["status"] == "partial"


def test_unsupported_warp_intrinsic_is_rejected_before_metal(tmp_path):
    target = tmp_path / "warp.cu"
    target.write_text(
        """
        __global__ void warp_case(int* output) {
            output[threadIdx.x] = __ballot_sync(0xffffffffu, threadIdx.x == 0);
        }
        """,
        encoding="ascii",
    )

    result, payload = _json("build", str(target))

    assert result.returncode == 1
    assert any(item["code"] == "UnsupportedWarpIntrinsic" for item in payload["diagnostics"])


def test_unsupported_warp_mask_is_rejected_before_metal(tmp_path):
    target = tmp_path / "warp_mask.cu"
    target.write_text(
        """
        __global__ void warp_case(int* output, unsigned int mask) {
            int value = threadIdx.x;
            output[threadIdx.x] = __shfl_sync(mask, value, 0);
        }
        """,
        encoding="ascii",
    )

    result, payload = _json("build", str(target))

    assert result.returncode == 1
    assert any(item["code"] == "UnsupportedWarpMask" for item in payload["diagnostics"])


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 14 warp intrinsics require Metal")
def test_supported_warp_intrinsics_match_lane_oracles(tmp_path):
    target = tmp_path / "warp_supported.cu"
    executable = tmp_path / "warp_supported"
    target.write_text(
        r'''
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_kernel(int* output) {
    int lane = threadIdx.x;
    int value = lane + 10;
    int from_zero = __shfl_sync(0xffffffffu, value, 0);
    int from_next = __shfl_down_sync(0xffffffffu, value, 1);
    int any_lane_seven = __any_sync(0xffffffffu, lane == 7);
    int all_in_warp = __all_sync(0xffffffffu, lane < 32);
    unsigned int active = __activemask();
    __syncwarp(0xffffffffu);
    output[lane * 5 + 0] = from_zero;
    output[lane * 5 + 1] = from_next;
    output[lane * 5 + 2] = any_lane_seven;
    output[lane * 5 + 3] = all_in_warp;
    output[lane * 5 + 4] = active == 0xffffffffu ? 1 : 0;
}

int main() {
    int output[32 * 5] = {};
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(output)) != cudaSuccess) return 10;
    warp_kernel<<<1, 32>>>(device);
    if (cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 11;
    for (int lane = 0; lane < 32; ++lane) {
        if (output[lane * 5 + 0] != 10) return 20;
        if (lane < 31 && output[lane * 5 + 1] != lane + 11) return 21;
        if (output[lane * 5 + 2] != 1) return 22;
        if (output[lane * 5 + 3] != 1) return 23;
        if (output[lane * 5 + 4] != 1) return 24;
    }
    cudaFree(device);
    std::printf("warp_supported=%d,%d\n", output[0], output[31 * 5 + 2]);
    return 0;
}
''',
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    build = subprocess.run(
        [sys.executable, str(MCU), "compile", str(target), "-O2", "-o", str(executable), "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    run = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    if run.returncode == 10:
        pytest.skip("Metal device is not available in this execution environment")
    assert run.returncode == 0, run.stderr or run.stdout
    assert "warp_supported=10,1" in run.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 14 graph replay requires Metal")
def test_explicit_kernel_graph_replays_kernel_nodes(tmp_path):
    target = tmp_path / "graph.cu"
    executable = tmp_path / "graph"
    target.write_text(
        r'''
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_value(int* values, int amount) {
    values[threadIdx.x] += amount;
}

int main() {
    int input[32] = {};
    int output[32] = {};
    int* device = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    cudaGraphNode_t add_three = nullptr;
    cudaGraphNode_t add_four = nullptr;
    int amount_three = 3;
    int amount_four = 4;

    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess) return 11;
    if (cudaGraphCreate(&graph, 0) != cudaSuccess) return 12;

    void* first_args[] = {&device, &amount_three};
    cudaKernelNodeParams first = {};
    first.func = (void*)add_value;
    first.gridDim = dim3(1);
    first.blockDim = dim3(32);
    first.kernelParams = first_args;
    if (cudaGraphAddKernelNode(&add_three, graph, nullptr, 0, &first) != cudaSuccess) return 13;

    void* second_args[] = {&device, &amount_four};
    cudaKernelNodeParams second = {};
    second.func = (void*)add_value;
    second.gridDim = dim3(1);
    second.blockDim = dim3(32);
    second.kernelParams = second_args;
    if (cudaGraphAddKernelNode(&add_four, graph, &add_three, 1, &second) != cudaSuccess) return 14;

    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess || node_count != 2) return 15;
    size_t root_count = 0;
    if (cudaGraphGetRootNodes(graph, nullptr, &root_count) != cudaSuccess || root_count != 1) return 16;
    if (cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0) != cudaSuccess) return 17;
    if (cudaGraphLaunch(executable_graph, nullptr) != cudaSuccess) return 18;
    if (cudaGraphLaunch(executable_graph, nullptr) != cudaSuccess) return 19;
    if (cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 20;
    for (int index = 0; index < 32; ++index) {
        if (output[index] != 14) return 21;
    }
    if (cudaGraphExecDestroy(executable_graph) != cudaSuccess) return 22;
    if (cudaGraphDestroy(graph) != cudaSuccess) return 23;
    cudaFree(device);
    std::printf("graph_value=%d\n", output[0]);
    return 0;
}
''',
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    build = subprocess.run(
        [sys.executable, str(MCU), "compile", str(target), "-O2", "-o", str(executable), "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    run = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    if run.returncode == 10:
        pytest.skip("Metal device is not available in this execution environment")
    assert run.returncode == 0, run.stderr or run.stdout
    assert "graph_value=14" in run.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 14 graph replay requires Metal")
def test_graph_memcpy_memset_and_stream_capture(tmp_path):
    target = tmp_path / "graph_memory_capture.cu"
    executable = tmp_path / "graph_memory_capture"
    target.write_text(
        r'''
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(int* values) {
    values[threadIdx.x] += 1;
}

int main() {
    int input[16] = {};
    int output[16] = {};
    int* device = nullptr;
    cudaGraph_t explicit_graph = nullptr;
    cudaGraphExec_t explicit_exec = nullptr;
    cudaGraphNode_t memset_node = nullptr;
    cudaGraphNode_t copy_node = nullptr;
    cudaGraphNode_t kernel_node = nullptr;

    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaGraphCreate(&explicit_graph, 0) != cudaSuccess) return 11;

    cudaMemsetParams memset_params = {};
    memset_params.dst = device;
    memset_params.value = 0;
    memset_params.elementSize = 1;
    memset_params.width = sizeof(input);
    memset_params.height = 1;
    if (cudaGraphAddMemsetNode(&memset_node, explicit_graph, nullptr, 0, &memset_params) != cudaSuccess) return 12;

    for (int index = 0; index < 16; ++index) input[index] = index;
    cudaMemcpy3DParms copy_params = {};
    copy_params.srcPtr = make_cudaPitchedPtr(input, sizeof(input), sizeof(input), 1);
    copy_params.dstPtr = make_cudaPitchedPtr(device, sizeof(input), sizeof(input), 1);
    copy_params.extent = make_cudaExtent(sizeof(input), 1, 1);
    copy_params.kind = cudaMemcpyHostToDevice;
    if (cudaGraphAddMemcpyNode(&copy_node, explicit_graph, &memset_node, 1, &copy_params) != cudaSuccess) return 13;

    void* kernel_args[] = {&device};
    cudaKernelNodeParams kernel_params = {};
    kernel_params.func = (void*)add_one;
    kernel_params.gridDim = dim3(1);
    kernel_params.blockDim = dim3(16);
    kernel_params.kernelParams = kernel_args;
    if (cudaGraphAddKernelNode(&kernel_node, explicit_graph, &copy_node, 1, &kernel_params) != cudaSuccess) return 14;
    if (cudaGraphInstantiate(&explicit_exec, explicit_graph, nullptr, nullptr, 0) != cudaSuccess) return 15;
    if (cudaGraphLaunch(explicit_exec, nullptr) != cudaSuccess) return 16;
    if (cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 17;
    for (int index = 0; index < 16; ++index) {
        if (output[index] != index + 1) return 18;
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t captured_graph = nullptr;
    cudaGraphExec_t captured_exec = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 19;
    for (int index = 0; index < 16; ++index) input[index] = index * 2;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) return 20;
    if (cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 21;
    add_one<<<1, 16, 0, stream>>>(device);
    if (cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 22;
    if (cudaStreamEndCapture(stream, &captured_graph) != cudaSuccess) return 23;
    if (cudaGraphInstantiate(&captured_exec, captured_graph, nullptr, nullptr, 0) != cudaSuccess) return 24;
    if (cudaGraphLaunch(captured_exec, stream) != cudaSuccess) return 25;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 26;
    for (int index = 0; index < 16; ++index) {
        if (output[index] != index * 2 + 1) return 27;
    }

    cudaGraphExecDestroy(captured_exec);
    cudaGraphDestroy(captured_graph);
    cudaStreamDestroy(stream);
    cudaGraphExecDestroy(explicit_exec);
    cudaGraphDestroy(explicit_graph);
    cudaFree(device);
    std::printf("graph_memory_capture=%d\n", output[15]);
    return 0;
}
''',
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    build = subprocess.run(
        [sys.executable, str(MCU), "compile", str(target), "-O2", "-o", str(executable), "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    run = subprocess.run([str(executable)], env=environment, text=True, capture_output=True, check=False)
    if run.returncode == 10:
        pytest.skip("Metal device is not available in this execution environment")
    assert run.returncode == 0, run.stderr or run.stdout
    assert "graph_memory_capture=31" in run.stdout
