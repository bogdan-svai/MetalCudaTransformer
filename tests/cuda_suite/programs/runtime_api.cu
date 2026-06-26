#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void add_one(unsigned int* values, int count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        values[index] += 1u;
    }
}

static int run_device() {
    int count = 0;
    int device = -1;
    cudaDeviceProp properties = {};
    int min_grid = 0;
    int block = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1) return 10;
    if (cudaGetDevice(&device) != cudaSuccess || device != 0) return 11;
    if (cudaSetDevice(0) != cudaSuccess) return 12;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 13;
    if (properties.maxThreadsPerBlock != 1024 || properties.warpSize != 32) return 14;
    if (properties.unifiedAddressing != 1 || properties.name[0] == '\0') return 15;
    if (cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, nullptr, 0, 128) != cudaSuccess) return 16;
    if (min_grid < 1 || block != 128) return 17;
    std::printf("status=pass device=%s block=%d\n", properties.name, block);
    return 0;
}

static int run_events() {
    const int count = 256;
    unsigned int input[count] = {};
    unsigned int output[count] = {};
    unsigned int* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float milliseconds = -1.0f;
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 20;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 21;
    if (cudaEventCreate(&start) != cudaSuccess) return 22;
    if (cudaEventCreateWithFlags(&stop, cudaEventBlockingSync) != cudaSuccess) return 23;
    if (cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 24;
    if (cudaEventRecord(start, stream) != cudaSuccess) return 25;
    add_one<<<2, 128, 0, stream>>>(device, count);
    if (cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 26;
    if (cudaEventRecord(stop, stream) != cudaSuccess) return 27;
    if (cudaEventSynchronize(stop) != cudaSuccess) return 28;
    if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess || milliseconds < 0.0f) return 29;
    for (int index = 0; index < count; ++index) {
        if (output[index] != 1u) return 30;
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device);
    std::printf("status=pass event_ms=%0.4f\n", milliseconds);
    return 0;
}

static int run_errors() {
    int value = 1;
    if (cudaMemcpy(&value, &value, 0, cudaMemcpyHostToHost) != cudaSuccess) return 40;
    if (cudaMemset(nullptr, 0, 0) != cudaSuccess) return 41;
    if (cudaSetDevice(4) != cudaErrorInvalidDevice) return 42;
    if (cudaPeekAtLastError() != cudaErrorInvalidDevice) return 43;
    if (cudaGetLastError() != cudaErrorInvalidDevice) return 44;
    if (cudaGetLastError() != cudaSuccess) return 45;
    if (cudaDeviceReset() != cudaSuccess) return 46;
    std::printf("status=pass error_api\n");
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "device";
    if (std::strcmp(mode, "device") == 0) return run_device();
    if (std::strcmp(mode, "events") == 0) return run_events();
    if (std::strcmp(mode, "errors") == 0) return run_errors();
    return 2;
}
