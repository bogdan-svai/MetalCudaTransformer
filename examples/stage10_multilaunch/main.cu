#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void add_one(unsigned int* values, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += 1u;
}

static double elapsed_us(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop
) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

int main(int argc, char** argv) {
    int launches = argc > 1 ? std::atoi(argv[1]) : 100;
    int count = argc > 2 ? std::atoi(argv[2]) : 4096;
    bool launch_only = argc > 3 && std::atoi(argv[3]) != 0;
    if (launches < 1 || launches > 5000 || count < 1 || count > 2000000) return 2;
    const int block = 256;
    const int grid = (count + block - 1) / block;
    std::vector<unsigned int> zeros(count, 0u);
    std::vector<unsigned int> output(count, 0u);
    unsigned int* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(unsigned int)) != cudaSuccess) return 10;

    cudaMemcpy(device, zeros.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    add_one<<<grid, block>>>(device, count);
    if (cudaDeviceSynchronize() != cudaSuccess) return 11;
    std::chrono::steady_clock::time_point sync_start;
    if (launch_only) {
        cudaMemcpy(device, zeros.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);
        sync_start = std::chrono::steady_clock::now();
    } else {
        sync_start = std::chrono::steady_clock::now();
        cudaMemcpy(device, zeros.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    for (int index = 0; index < launches; ++index) {
        add_one<<<grid, block>>>(device, count);
        if (cudaDeviceSynchronize() != cudaSuccess) return 12;
    }
    std::chrono::steady_clock::time_point sync_stop;
    if (launch_only) {
        sync_stop = std::chrono::steady_clock::now();
        cudaMemcpy(output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        sync_stop = std::chrono::steady_clock::now();
    }
    for (int index = 0; index < count; ++index) if (output[index] != (unsigned int)launches) return 20;

    cudaStream_t stream = nullptr;
    if (!launch_only && cudaStreamCreate(&stream) != cudaSuccess) return 13;
    if (launch_only) {
        cudaMemcpy(device, zeros.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    auto async_start = std::chrono::steady_clock::now();
    if (!launch_only && cudaMemcpyAsync(
            device, zeros.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream
        ) != cudaSuccess) return 14;
    for (int index = 0; index < launches; ++index) {
        if (launch_only) add_one<<<grid, block>>>(device, count);
        else add_one<<<grid, block, 0, stream>>>(device, count);
    }
    if (!launch_only && cudaMemcpyAsync(
            output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream
        ) != cudaSuccess) return 15;
    if ((launch_only ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream)) != cudaSuccess) return 16;
    auto async_stop = std::chrono::steady_clock::now();
    if (launch_only) {
        cudaMemcpy(output.data(), device, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    if (!launch_only && cudaStreamDestroy(stream) != cudaSuccess) return 17;
    for (int index = 0; index < count; ++index) if (output[index] != (unsigned int)launches) return 21;
    cudaFree(device);

    double synchronous = elapsed_us(sync_start, sync_stop);
    double asynchronous = elapsed_us(async_start, async_stop);
    std::printf("launches=%d\n", launches);
    std::printf("count=%d\n", count);
    std::printf("launch_only=%d\n", launch_only ? 1 : 0);
    std::printf("synchronous_us=%.3f\n", synchronous);
    std::printf("batched_us=%.3f\n", asynchronous);
    std::printf("speedup=%.3f\n", synchronous / asynchronous);
    std::printf("result=%u\n", output[0]);
    return 0;
}
