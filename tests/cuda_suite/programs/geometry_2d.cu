#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void write_ids(unsigned int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) output[y * width + x] = (unsigned int)(y * width + x);
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    int width = std::atoi(argv[1]);
    int height = std::atoi(argv[2]);
    int block_x = std::atoi(argv[3]);
    int block_y = std::atoi(argv[4]);
    if (width < 1 || height < 1 || block_x * block_y > 1024) return 3;
    int count = width * height;
    std::vector<unsigned int> output(count, 0u);
    unsigned int* device = nullptr;
    cudaMalloc(&device, count * sizeof(unsigned int));
    cudaMemset(device, 0, count * sizeof(unsigned int));
    dim3 block(block_x, block_y);
    dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
    write_ids<<<grid, block>>>(device, width, height);
    cudaMemcpy(output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(device);
    for (int index = 0; index < count; ++index) if (output[index] != (unsigned int)index) return 20;
    std::printf("status=pass cells=%d\n", count);
    return 0;
}
