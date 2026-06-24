import json
from pathlib import Path
import platform
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def _build_and_run(tmp_path, source):
    target = tmp_path / "case.cu"
    output = tmp_path / "native"
    target.write_text(source, encoding="utf-8")
    build = subprocess.run(
        [sys.executable, str(MCU), "build", str(target), "--native", "--output", str(output), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    payload = json.loads(build.stdout)
    executable = Path(payload["build"]["native"]["executable"])
    run = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
    return run, output


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 atomics require Metal")
def test_typed_atomic_add_min_exchange_and_cas(tmp_path):
    run, output = _build_and_run(tmp_path, r'''
#include <cstdint>
#include <cstdio>

__global__ void atomic_work(
    uint32_t* counter,
    uint32_t* minimum,
    uint32_t* exchanged,
    uint32_t* compared,
    uint32_t* old_values
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(counter, 1u);
    atomicMin(minimum, 19999u - index);
    if (index == 0) {
        old_values[0] = atomicExch(exchanged, 9u);
        old_values[1] = atomicCAS(compared, 7u, 11u);
    }
}

__global__ void signed_minmax(int* minimum, int* maximum) {
    int value = static_cast<int>(threadIdx.x) - 50;
    atomicMin(minimum, value);
    atomicMax(maximum, value);
}

int main() {
    uint32_t initial[4] = {0u, UINT32_MAX, 5u, 7u};
    uint32_t result[6] = {};
    uint32_t* device = nullptr;
    if (cudaMalloc(&device, sizeof(result)) != cudaSuccess) return 10;
    if (cudaMemcpy(device, initial, sizeof(initial), cudaMemcpyHostToDevice) != cudaSuccess) return 11;
    atomic_work<<<20, 1000>>>(device, device + 1, device + 2, device + 3, device + 4);
    if (cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) return 12;
    cudaFree(device);
    if (result[0] != 20000u || result[1] != 0u) return 20;
    if (result[2] != 9u || result[3] != 11u) return 21;
    if (result[4] != 5u || result[5] != 7u) return 22;
    int signed_initial[2] = {INT32_MAX, INT32_MIN};
    int signed_result[2] = {};
    int* signed_device = nullptr;
    if (cudaMalloc(&signed_device, sizeof(signed_result)) != cudaSuccess) return 13;
    cudaMemcpy(signed_device, signed_initial, sizeof(signed_initial), cudaMemcpyHostToDevice);
    signed_minmax<<<1, 100>>>(signed_device, signed_device + 1);
    cudaMemcpy(signed_result, signed_device, sizeof(signed_result), cudaMemcpyDeviceToHost);
    cudaFree(signed_device);
    if (signed_result[0] != -50 || signed_result[1] != 49) return 23;
    std::printf("counter=%u minimum=%u exchange_old=%u cas_old=%u\n", result[0], result[1], result[4], result[5]);
    return 0;
}
''')
    assert run.returncode == 0, run.stderr or run.stdout
    assert "counter=20000 minimum=0 exchange_old=5 cas_old=7" in run.stdout
    metal = (output / "generated.metal").read_text(encoding="utf-8")
    assert "device atomic_uint* counter" in metal
    assert "atomic_fetch_min_explicit" in metal
    assert "mcu_atomic_cas" in metal
    assert "device atomic_int* minimum" in metal


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 runtime requires Metal")
def test_cuda_memset_copy_offsets_bounds_and_zero_bytes(tmp_path):
    run, _output = _build_and_run(tmp_path, r'''
#include <cstdio>
#include <cstring>

__global__ void device_fill(unsigned char* values) {
    if (threadIdx.x == 0) memset(values, 0x5a, 16);
}

int main() {
    unsigned char input[16];
    unsigned char output[16] = {};
    unsigned char host_copy[16] = {};
    for (int i = 0; i < 16; ++i) input[i] = static_cast<unsigned char>(i);
    unsigned char* device = nullptr;
    if (cudaMalloc(&device, 16) != cudaSuccess) return 10;
    if (cudaMemcpy(device, input, 16, cudaMemcpyHostToDevice) != cudaSuccess) return 11;
    if (cudaMemset(device + 4, 0xab, 6) != cudaSuccess) return 12;
    if (cudaMemcpy(output, device, 16, cudaMemcpyDefault) != cudaSuccess) return 13;
    if (cudaMemcpy(host_copy, output, 16, cudaMemcpyHostToHost) != cudaSuccess) return 17;
    unsigned char* second = nullptr;
    if (cudaMalloc(&second, 16) != cudaSuccess) return 18;
    if (cudaMemcpy(second, device, 16, cudaMemcpyDeviceToDevice) != cudaSuccess) return 19;
    if (cudaMemcpy(output, second, 16, cudaMemcpyDeviceToHost) != cudaSuccess) return 21;
    if (cudaMemcpy(output, input, 16, cudaMemcpyHostToDevice) == cudaSuccess) return 22;
    if (cudaMemcpy(output, input, 16, static_cast<cudaMemcpyKind>(99)) == cudaSuccess) return 23;
    if (cudaMemcpy(device + 1, device, 8, cudaMemcpyDeviceToDevice) == cudaSuccess) return 27;
    device_fill<<<1, 1>>>(second);
    if (cudaMemcpy(output, second, 16, cudaMemcpyDeviceToHost) != cudaSuccess) return 24;
    for (int i = 0; i < 16; ++i) if (output[i] != 0x5a) return 25;
    if (cudaMemcpy(output, device, 16, cudaMemcpyDeviceToHost) != cudaSuccess) return 26;
    if (cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToHost) != cudaSuccess) return 14;
    if (cudaMemset(nullptr, 0, 0) != cudaSuccess) return 15;
    if (cudaMemset(device + 15, 0, 2) == cudaSuccess) return 16;
    cudaFree(second);
    cudaFree(device);
    for (int i = 0; i < 16; ++i) {
        unsigned char expected = (i >= 4 && i < 10) ? 0xab : static_cast<unsigned char>(i);
        if (output[i] != expected) return 20;
    }
    std::puts("memory_ok");
    return 0;
}
''')
    assert run.returncode == 0, run.stderr or run.stdout
    assert "memory_ok" in run.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 dynamic shared validation requires Metal")
def test_dynamic_threadgroup_runtime_rejects_invalid_sizes(tmp_path):
    run, _output = _build_and_run(tmp_path, r'''
__global__ void dynamic_kernel(unsigned int* output) {
    extern __shared__ unsigned int values[];
    if (threadIdx.x == 0) values[0] = 1u;
    __syncthreads();
    output[threadIdx.x] = values[0];
}

int main() {
    unsigned int* device = nullptr;
    if (cudaMalloc(&device, 32 * sizeof(unsigned int)) != cudaSuccess) return 10;
    dynamic_kernel<<<1, 32, 3>>>(device);
    if (cudaGetLastError() == cudaSuccess) return 20;
    if (cudaGetLastError() != cudaSuccess) return 22;
    dynamic_kernel<<<1, 32, 65536>>>(device);
    if (cudaGetLastError() == cudaSuccess) return 21;
    cudaFree(device);
    return 0;
}
''')
    assert run.returncode == 0, run.stderr or run.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 shared memory requires Metal")
@pytest.mark.parametrize("dynamic", [False, True])
def test_static_and_dynamic_threadgroup_reduction(tmp_path, dynamic):
    declaration = "extern __shared__ unsigned int values[];" if dynamic else "__shared__ unsigned int values[32];"
    launch = "reduce<<<1, 32, 32 * sizeof(unsigned int)>>>(device, device + 32);" if dynamic else "reduce<<<1, 32>>>(device, device + 32);"
    run, output = _build_and_run(tmp_path, f'''
#include <cstdio>

__global__ void reduce(unsigned int* input, unsigned int* output) {{
    {declaration}
    unsigned int lane = threadIdx.x;
    if (lane == 0) memset(values, 0, 32 * sizeof(unsigned int));
    __syncthreads();
    values[lane] = input[lane];
    __syncthreads();
    for (unsigned int stride = 16; stride > 0; stride /= 2) {{
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }}
    if (lane == 0) output[0] = values[0];
}}

int main() {{
    unsigned int input[32];
    for (unsigned int i = 0; i < 32; ++i) input[i] = i + 1;
    unsigned int* device = nullptr;
    if (cudaMalloc(&device, 33 * sizeof(unsigned int)) != cudaSuccess) return 10;
    cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice);
    {launch}
    unsigned int result = 0;
    cudaMemcpy(&result, device + 32, sizeof(result), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (result != 528u) return 20;
    std::printf("shared_sum=%u\\n", result);
    return 0;
}}
''')
    assert run.returncode == 0, run.stderr or run.stdout
    assert "shared_sum=528" in run.stdout
    routing = json.loads((output / "routing_table.txt").read_text(encoding="utf-8"))
    route = routing["translated_kernels"]["reduce"]
    if dynamic:
        assert route["threadgroup_memory"]["slot"] == 0
    else:
        assert "threadgroup_memory" not in route


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 shared atomics require Metal")
def test_threadgroup_atomic_storage_and_plain_accesses_are_lowered(tmp_path):
    run, output = _build_and_run(tmp_path, r'''
#include <cstdio>

__global__ void shared_counter(unsigned int* output) {
    __shared__ unsigned int counters[2];
    unsigned int lane = threadIdx.x;
    if (lane < 2) counters[lane] = 0u;
    __syncthreads();
    atomicAdd(&counters[lane & 1u], 1u);
    __syncthreads();
    if (lane < 2) output[lane] = counters[lane];
}

int main() {
    unsigned int* device = nullptr;
    unsigned int output[2] = {};
    if (cudaMalloc(&device, sizeof(output)) != cudaSuccess) return 10;
    shared_counter<<<1, 64>>>(device);
    cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (output[0] != 32u || output[1] != 32u) return 20;
    std::printf("shared_atomic=%u,%u\n", output[0], output[1]);
    return 0;
}
''')
    assert run.returncode == 0, run.stderr or run.stdout
    assert "shared_atomic=32,32" in run.stdout
    metal = (output / "generated.metal").read_text(encoding="utf-8")
    assert "threadgroup atomic_uint counters[2]" in metal
    assert "atomic_store_explicit(&counters[lane]" in metal
    assert "atomic_load_explicit(&counters[lane]" in metal


@pytest.mark.skipif(platform.system() != "Darwin", reason="Stage 8 integration requires Metal")
def test_generic_multilaunch_reduction_and_smallest_nonce_match_cpu_oracle(tmp_path):
    run, _output = _build_and_run(tmp_path, r'''
#include <cstdint>
#include <cstdio>
#include <climits>

__host__ __device__ uint32_t mix_word(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__global__ void hash_items(const uint32_t* input, uint32_t* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = mix_word(input[index]);
}

__global__ void reduce_pairs(const uint32_t* input, uint32_t* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int parents = (count + 1) / 2;
    if (index < parents) {
        uint32_t left = input[index * 2];
        uint32_t right = index * 2 + 1 < count ? input[index * 2 + 1] : left;
        output[index] = mix_word(left ^ (right + 0x9e3779b9u));
    }
}

__global__ void find_smallest(uint32_t seed, uint32_t limit, uint32_t* best) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t nonce = index; nonce < limit; nonce += stride) {
        if ((mix_word(seed ^ nonce) & 0x3ffu) == 0u) atomicMin(best, nonce);
    }
}

int main() {
    const int count = 7;
    uint32_t input[count] = {3u, 5u, 8u, 13u, 21u, 34u, 55u};
    uint32_t cpu_a[count];
    uint32_t cpu_b[count];
    for (int i = 0; i < count; ++i) cpu_a[i] = mix_word(input[i]);
    int current = count;
    uint32_t* source = cpu_a;
    uint32_t* destination = cpu_b;
    while (current > 1) {
        int parents = (current + 1) / 2;
        for (int i = 0; i < parents; ++i) {
            uint32_t left = source[i * 2];
            uint32_t right = i * 2 + 1 < current ? source[i * 2 + 1] : left;
            destination[i] = mix_word(left ^ (right + 0x9e3779b9u));
        }
        uint32_t* temporary = source;
        source = destination;
        destination = temporary;
        current = parents;
    }
    uint32_t expected_root = source[0];

    uint32_t* device_input = nullptr;
    uint32_t* device_a = nullptr;
    uint32_t* device_b = nullptr;
    uint32_t* device_best = nullptr;
    cudaMalloc(&device_input, sizeof(input));
    cudaMalloc(&device_a, sizeof(input));
    cudaMalloc(&device_b, sizeof(input));
    cudaMalloc(&device_best, sizeof(uint32_t));
    cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice);
    hash_items<<<1, 32>>>(device_input, device_a, count);
    current = count;
    while (current > 1) {
        reduce_pairs<<<1, 32>>>(device_a, device_b, current);
        uint32_t* temporary = device_a;
        device_a = device_b;
        device_b = temporary;
        current = (current + 1) / 2;
    }
    uint32_t gpu_root = 0;
    cudaMemcpy(&gpu_root, device_a, sizeof(gpu_root), cudaMemcpyDeviceToHost);
    if (gpu_root != expected_root) return 20;

    const uint32_t limit = 100000u;
    uint32_t expected_nonce = UINT32_MAX;
    for (uint32_t nonce = 0; nonce < limit; ++nonce) {
        if ((mix_word(gpu_root ^ nonce) & 0x3ffu) == 0u) {
            expected_nonce = nonce;
            break;
        }
    }
    uint32_t gpu_nonce = UINT32_MAX;
    cudaMemcpy(device_best, &gpu_nonce, sizeof(gpu_nonce), cudaMemcpyHostToDevice);
    find_smallest<<<8, 128>>>(gpu_root, limit, device_best);
    cudaMemcpy(&gpu_nonce, device_best, sizeof(gpu_nonce), cudaMemcpyDeviceToHost);
    cudaFree(device_best);
    cudaFree(device_b);
    cudaFree(device_a);
    cudaFree(device_input);
    if (gpu_nonce != expected_nonce) return 21;
    std::printf("oracle_root=%u oracle_nonce=%u\n", gpu_root, gpu_nonce);
    return 0;
}
''')
    assert run.returncode == 0, run.stderr or run.stdout
    assert "oracle_root=" in run.stdout
    assert "oracle_nonce=" in run.stdout
