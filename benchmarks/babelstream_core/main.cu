// STREAM-style CUDA Core benchmark derived from the operation set used by
// BabelStream. This is an MCU compatibility variant, not an official
// conforming BabelStream implementation or result.

#include <cmath>
#include <cstdio>
#include <cstdlib>

__global__ void stream_init(float* a, float* b, float* c, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        a[index] = 0.1f;
        b[index] = 0.2f;
        c[index] = 0.0f;
    }
}

__global__ void stream_copy(const float* a, float* c, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) c[index] = a[index];
}

__global__ void stream_mul(float* b, const float* c, float scalar, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) b[index] = scalar * c[index];
}

__global__ void stream_add(const float* a, const float* b, float* c, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) c[index] = a[index] + b[index];
}

__global__ void stream_triad(float* a, const float* b, const float* c, float scalar, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) a[index] = b[index] + scalar * c[index];
}

int main(int argc, char** argv) {
    int count = argc > 1 ? std::atoi(argv[1]) : 8388608;
    int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    if (count <= 0 || iterations <= 0) return 2;

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    const size_t bytes = (size_t)count * sizeof(float);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    if (cudaMalloc(&a, bytes) != cudaSuccess) return 10;
    if (cudaMalloc(&b, bytes) != cudaSuccess) return 11;
    if (cudaMalloc(&c, bytes) != cudaSuccess) return 12;

    stream_init<<<blocks, threads>>>(a, b, c, count);
    if (cudaDeviceSynchronize() != cudaSuccess) return 13;
    const float scalar = 0.4f;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stream_copy<<<blocks, threads>>>(a, c, count);
        stream_mul<<<blocks, threads>>>(b, c, scalar, count);
        stream_add<<<blocks, threads>>>(a, b, c, count);
        stream_triad<<<blocks, threads>>>(a, b, c, scalar, count);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 14;

    float observed[6] = {};
    if (cudaMemcpy(&observed[0], a, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 15;
    if (cudaMemcpy(&observed[1], a + count - 1, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 16;
    if (cudaMemcpy(&observed[2], b, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 17;
    if (cudaMemcpy(&observed[3], b + count - 1, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 18;
    if (cudaMemcpy(&observed[4], c, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 19;
    if (cudaMemcpy(&observed[5], c + count - 1, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 20;

    float expected_a = 0.1f;
    float expected_b = 0.2f;
    float expected_c = 0.0f;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        expected_c = expected_a;
        expected_b = scalar * expected_c;
        expected_c = expected_a + expected_b;
        expected_a = expected_b + scalar * expected_c;
    }
    const float expected[6] = {
        expected_a, expected_a, expected_b, expected_b, expected_c, expected_c,
    };
    for (int index = 0; index < 6; ++index) {
        float tolerance = 1.0e-5f * (1.0f + std::fabs(expected[index]));
        if (std::fabs(observed[index] - expected[index]) > tolerance) return 30 + index;
    }

    std::printf("ok=1\n");
    std::printf("elements=%d\n", count);
    std::printf("iterations=%d\n", iterations);
    std::printf("bytes_per_element=%zu\n", sizeof(float));
    std::printf("a=%0.9g\n", observed[0]);
    std::printf("b=%0.9g\n", observed[2]);
    std::printf("c=%0.9g\n", observed[4]);

    cudaFree(c);
    cudaFree(b);
    cudaFree(a);
    return 0;
}
