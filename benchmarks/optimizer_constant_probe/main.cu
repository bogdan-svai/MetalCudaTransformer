#include <cstdio>
#include <cstdlib>

__global__ void optimizer_constant_probe(int* output, int count) {
    const unsigned int table[64] = {
        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
        8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
        16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u,
        24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u,
        32u, 33u, 34u, 35u, 36u, 37u, 38u, 39u,
        40u, 41u, 42u, 43u, 44u, 45u, 46u, 47u,
        48u, 49u, 50u, 51u, 52u, 53u, 54u, 55u,
        56u, 57u, 58u, 59u, 60u, 61u, 62u, 63u
    };
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = (int)(table[3] + table[60]);
}

int main() {
    const int count = 1 << 20;
    int* host = (int*)std::malloc((size_t)count * sizeof(int));
    int* device = nullptr;
    if (!host || cudaMalloc(&device, (size_t)count * sizeof(int)) != cudaSuccess) return 10;
    optimizer_constant_probe<<<4096, 256>>>(device, count);
    if (cudaDeviceSynchronize() != cudaSuccess) return 11;
    if (cudaMemcpy(host, device, (size_t)count * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 12;
    cudaFree(device);
    for (int index = 0; index < count; ++index) if (host[index] != 63) return 20;
    std::printf("ok=1 result=%d count=%d\n", host[0], count);
    std::free(host);
    return 0;
}
