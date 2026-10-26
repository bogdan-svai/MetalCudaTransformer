#include <cstdio>

__global__ void optimizer_probe(int* output) {
    int values[4];
    for (int i = 0; i < 4; ++i) {
        values[i] = i + 1;
    }
    output[0] = values[0] + values[3];
}

int main() {
    int result = 0;
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 10;
    optimizer_probe<<<1, 1>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess) return 11;
    if (cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 12;
    cudaFree(device);
    if (result != 5) return 20;
    std::printf("ok=1 result=%d\n", result);
    return 0;
}
