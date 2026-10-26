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
