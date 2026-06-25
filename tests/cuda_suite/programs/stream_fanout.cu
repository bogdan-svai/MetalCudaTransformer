#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void add_value(unsigned int* values, int count, unsigned int amount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += amount;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    int stream_count = std::atoi(argv[1]);
    int launches = std::atoi(argv[2]);
    int count = std::atoi(argv[3]);
    if (stream_count < 1 || stream_count > 4 || launches < 1 || launches > 1000 || count < 1) return 3;
    std::vector<unsigned int> zero(count, 0u);
    std::vector<unsigned int> outputs[4];
    unsigned int* devices[4] = {};
    cudaStream_t streams[4] = {};
    int block = 256;
    int grid = (count + block - 1) / block;
    for (int stream = 0; stream < stream_count; ++stream) {
        outputs[stream].resize(count);
        cudaMalloc(&devices[stream], count * sizeof(unsigned int));
        cudaStreamCreate(&streams[stream]);
        cudaMemcpyAsync(devices[stream], zero.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[stream]);
        for (int launch = 0; launch < launches; ++launch) {
            add_value<<<grid, block, 0, streams[stream]>>>(devices[stream], count, (unsigned int)(stream + 1));
        }
        cudaMemcpyAsync(outputs[stream].data(), devices[stream], count * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[stream]);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 10;
    for (int stream = 0; stream < stream_count; ++stream) {
        unsigned int expected = (unsigned int)(launches * (stream + 1));
        for (int index = 0; index < count; ++index) if (outputs[stream][index] != expected) return 20;
        cudaStreamDestroy(streams[stream]);
        cudaFree(devices[stream]);
    }
    std::printf("status=pass streams=%d launches=%d\n", stream_count, launches);
    return 0;
}
