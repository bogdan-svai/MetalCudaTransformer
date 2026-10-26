// General CUDA histogram benchmark with configurable atomic contention.

#include <cstdio>
#include <cstdlib>

__global__ void histogram_atomic(
    const unsigned int* input,
    unsigned int* histogram,
    int count,
    int bins
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int current = index; current < count; current += stride) {
        unsigned int bin = input[current] % (unsigned int)bins;
        atomicAdd(&histogram[bin], 1u);
    }
}

static unsigned int input_value(unsigned int index) {
    unsigned int value = index * 1664525u + 1013904223u;
    value ^= value >> 16;
    return value;
}

int main(int argc, char** argv) {
    int count = argc > 1 ? std::atoi(argv[1]) : 8388608;
    int bins = argc > 2 ? std::atoi(argv[2]) : 256;
    int iterations = argc > 3 ? std::atoi(argv[3]) : 10;
    int blocks = argc > 4 ? std::atoi(argv[4]) : 512;
    if (count <= 0 || bins <= 0 || iterations <= 0 || blocks <= 0) return 2;

    size_t input_bytes = (size_t)count * sizeof(unsigned int);
    size_t histogram_bytes = (size_t)bins * sizeof(unsigned int);
    unsigned int* host_input = (unsigned int*)std::malloc(input_bytes);
    unsigned int* observed = (unsigned int*)std::malloc(histogram_bytes);
    unsigned int* expected = (unsigned int*)std::calloc((size_t)bins, sizeof(unsigned int));
    if (!host_input || !observed || !expected) return 3;
    for (int index = 0; index < count; ++index) {
        unsigned int value = input_value((unsigned int)index);
        host_input[index] = value;
        expected[value % (unsigned int)bins] += (unsigned int)iterations;
    }

    unsigned int* device_input = nullptr;
    unsigned int* device_histogram = nullptr;
    if (cudaMalloc(&device_input, input_bytes) != cudaSuccess) return 10;
    if (cudaMalloc(&device_histogram, histogram_bytes) != cudaSuccess) return 11;
    if (cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 12;
    if (cudaMemset(device_histogram, 0, histogram_bytes) != cudaSuccess) return 13;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        histogram_atomic<<<blocks, 256>>>(device_input, device_histogram, count, bins);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 14;
    if (cudaMemcpy(observed, device_histogram, histogram_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 15;

    unsigned long long total = 0;
    for (int bin = 0; bin < bins; ++bin) {
        if (observed[bin] != expected[bin]) return 20;
        total += observed[bin];
    }
    unsigned long long expected_total = (unsigned long long)count * (unsigned long long)iterations;
    if (total != expected_total) return 21;
    std::printf("ok=1\n");
    std::printf("elements=%d\n", count);
    std::printf("bins=%d\n", bins);
    std::printf("iterations=%d\n", iterations);
    std::printf("blocks=%d\n", blocks);
    std::printf("updates=%llu\n", total);

    cudaFree(device_histogram);
    cudaFree(device_input);
    std::free(expected);
    std::free(observed);
    std::free(host_input);
    return 0;
}
