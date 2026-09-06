// CUDA Core dot-product reduction using grid-stride accumulation and
// threadgroup shared memory. The operation is part of the BabelStream suite,
// but this standalone MCU workload is not an official BabelStream result.

#include <cmath>
#include <cstdio>
#include <cstdlib>

__global__ void dot_reduce(
    const float* a,
    const float* b,
    float* partial,
    int count
) {
    __shared__ float shared_values[256];
    int lane = threadIdx.x;
    int index = blockIdx.x * blockDim.x + lane;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int current = index; current < count; current += stride) {
        sum += a[current] * b[current];
    }
    shared_values[lane] = sum;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (lane < offset) {
            shared_values[lane] += shared_values[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) partial[blockIdx.x] = shared_values[0];
}

static float value_a(int index) {
    return 0.25f + (float)(index % 251) * 0.00390625f;
}

static float value_b(int index) {
    return 0.5f + (float)(index % 127) * 0.0078125f;
}

int main(int argc, char** argv) {
    int count = argc > 1 ? std::atoi(argv[1]) : 16777216;
    int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    int blocks = argc > 3 ? std::atoi(argv[3]) : 128;
    if (count <= 0 || iterations <= 0 || blocks <= 0) return 2;
    size_t bytes = (size_t)count * sizeof(float);

    float* host_a = (float*)std::malloc(bytes);
    float* host_b = (float*)std::malloc(bytes);
    float* host_partial = (float*)std::malloc((size_t)blocks * sizeof(float));
    if (!host_a || !host_b || !host_partial) return 3;
    double expected = 0.0;
    for (int index = 0; index < count; ++index) {
        host_a[index] = value_a(index);
        host_b[index] = value_b(index);
        expected += (double)host_a[index] * (double)host_b[index];
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_partial = nullptr;
    if (cudaMalloc(&device_a, bytes) != cudaSuccess) return 10;
    if (cudaMalloc(&device_b, bytes) != cudaSuccess) return 11;
    if (cudaMalloc(&device_partial, (size_t)blocks * sizeof(float)) != cudaSuccess) return 12;
    if (cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 13;
    if (cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 14;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        dot_reduce<<<blocks, 256>>>(device_a, device_b, device_partial, count);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 15;
    if (cudaMemcpy(host_partial, device_partial, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 16;

    double observed = 0.0;
    for (int block = 0; block < blocks; ++block) observed += host_partial[block];
    double relative_error = std::fabs(observed - expected) / std::fabs(expected);
    if (relative_error > 1.0e-5) return 20;
    std::printf("ok=1\n");
    std::printf("elements=%d\n", count);
    std::printf("iterations=%d\n", iterations);
    std::printf("blocks=%d\n", blocks);
    std::printf("expected=%0.12g\n", expected);
    std::printf("observed=%0.12g\n", observed);
    std::printf("relative_error=%0.12g\n", relative_error);

    cudaFree(device_partial);
    cudaFree(device_b);
    cudaFree(device_a);
    std::free(host_partial);
    std::free(host_b);
    std::free(host_a);
    return 0;
}
