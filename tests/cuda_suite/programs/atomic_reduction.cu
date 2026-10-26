#include <cstdio>
#include <cstdlib>

__global__ void reduce_blocks(unsigned int* output) {
    __shared__ unsigned int values[256];
    unsigned int lane = threadIdx.x;
    values[lane] = 1u;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) atomicAdd(output, values[0]);
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int blocks = std::atoi(argv[1]);
    int threads = std::atoi(argv[2]);
    if (blocks < 1 || blocks > 4096 || threads < 1 || threads > 256 || (threads & (threads - 1))) return 3;
    unsigned int result = 0;
    unsigned int* device = nullptr;
    cudaMalloc(&device, sizeof(result));
    cudaMemset(device, 0, sizeof(result));
    reduce_blocks<<<blocks, threads>>>(device);
    cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (result != (unsigned int)(blocks * threads)) return 20;
    std::printf("status=pass reduced=%u\n", result);
    return 0;
}
