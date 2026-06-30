#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_intrinsic_kernel(int* output) {
    int lane = threadIdx.x;
    int value = lane + 3;
    output[lane * 4 + 0] = __shfl_sync(0xffffffffu, value, 0);
    output[lane * 4 + 1] = __shfl_down_sync(0xffffffffu, value, 1);
    output[lane * 4 + 2] = __any_sync(0xffffffffu, lane == 11);
    output[lane * 4 + 3] = __all_sync(0xffffffffu, lane < 32);
    __syncwarp();
}

int main() {
    int output[32 * 4] = {};
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(output)) != cudaSuccess) return 10;
    warp_intrinsic_kernel<<<1, 32>>>(device);
    if (cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 11;
    for (int lane = 0; lane < 32; ++lane) {
        if (output[lane * 4 + 0] != 3) return 20;
        if (lane < 31 && output[lane * 4 + 1] != lane + 4) return 21;
        if (output[lane * 4 + 2] != 1) return 22;
        if (output[lane * 4 + 3] != 1) return 23;
    }
    cudaFree(device);
    std::printf("status=pass warp_intrinsics=%d\n", output[0]);
    return 0;
}
