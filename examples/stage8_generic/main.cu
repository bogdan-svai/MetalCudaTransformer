#include <cstdio>

__device__ void increment_shared(unsigned int* value) {
    atomicAdd(value, 1u);
}

__global__ void count_threads(unsigned int* output) {
    __shared__ unsigned int count;
    if (threadIdx.x == 0) {
        count = 0u;
    }
    __syncthreads();
    increment_shared(&count);
    __syncthreads();
    if (threadIdx.x == 0) {
        output[0] = count;
    }
}

int main() {
    unsigned int* device_output = nullptr;
    unsigned int output = 0;
    if (cudaMalloc(&device_output, sizeof(output)) != cudaSuccess) return 10;
    count_threads<<<1, 64>>>(device_output);
    if (cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 11;
    cudaFree(device_output);
    if (output != 64u) return 12;
    std::printf("stage8_shared_atomic=%u\n", output);
    return 0;
}
