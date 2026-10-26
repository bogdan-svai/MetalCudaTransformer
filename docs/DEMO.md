# Vector transform demo

The vector example applies an unsigned integer transform and compares every
result with a CPU calculation. The complete input is [examples/vector/main.cu](../examples/vector/main.cu).

## Requirements

Use macOS on Apple Silicon, Python 3.11+, native libclang, Xcode and the Metal
Toolchain. Run `./install.sh` from the checkout; no NVIDIA GPU or CUDA toolkit is
needed. The commands below run from the source directory.

## Input CUDA code

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void transform(unsigned int* values, int count, unsigned int factor, unsigned int addend) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = values[index] * factor + addend;
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    int count = std::atoi(argv[1]);
    int launches = std::atoi(argv[2]);
    unsigned int factor = (unsigned int)std::strtoul(argv[3], nullptr, 10);
    unsigned int addend = (unsigned int)std::strtoul(argv[4], nullptr, 10);
    if (count < 1 || count > 1000000 || launches < 1 || launches > 1000) return 3;
    std::vector<unsigned int> input(count);
    std::vector<unsigned int> expected(count);
    std::vector<unsigned int> output(count);
    for (int index = 0; index < count; ++index) input[index] = expected[index] = (unsigned int)(index % 17);
    for (int launch = 0; launch < launches; ++launch) {
        for (int index = 0; index < count; ++index) expected[index] = expected[index] * factor + addend;
    }
    unsigned int* device = nullptr;
    cudaMalloc(&device, count * sizeof(unsigned int));
    cudaMemcpy(device, input.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    int block = 256;
    int grid = (count + block - 1) / block;
    for (int launch = 0; launch < launches; ++launch) transform<<<grid, block>>>(device, count, factor, addend);
    cudaMemcpy(output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(device);
    for (int index = 0; index < count; ++index) if (output[index] != expected[index]) return 20;
    std::printf("status=pass count=%d launches=%d\n", count, launches);
    return 0;
}
```

The arguments are element count, number of launches, multiplication factor and
addend. The demo uses 256 threads per block, checks all results, and returns 20
if an element differs from the CPU calculation.

## Build and run

```bash
./.venv/bin/python mcu check examples/vector/main.cu
./.venv/bin/python mcu build examples/vector/main.cu --native --output .mcu/vector
./.mcu/vector/main 1024 1 2 1
```

On a successful run, the source prints exactly:

```text
status=pass count=1024 launches=1
```

The success message is printed after every GPU result matches its CPU reference.
Native compilation requires Xcode and the Metal Toolchain; execution also requires
a usable Metal device.

## Inspect generated Metal without a native build

```bash
./.venv/bin/python mcu build examples/vector/main.cu
unzip -p mcu_artifacts.zip generated.metal
```

The generated entry point is `transform_msl`. `values` is a writable `device`
pointer at buffer 0. `count`, `factor` and `addend` are constant scalar bindings
at buffers 1, 2 and 3. The emitted body includes these actual generated lines:

```metal
int index = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
if (index < count) values[index] = values[index] * factor + addend;
```

CUDA block and thread indices map to Metal threadgroup and thread positions.
The native host rewrite binds the arguments, dispatches the kernel through the
Objective-C++ runtime, and copies results back for the CPU comparison. A passing
source check alone does not establish that the native result is correct.
