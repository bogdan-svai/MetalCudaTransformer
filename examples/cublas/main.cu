#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

static float value_at(int row, int column, int rows) {
    return static_cast<float>(((row + 1) * 17 + (column + 1) * 31 + rows) % 97) / 16.0f;
}

int main(int argc, char** argv) {
    int size = argc > 1 ? std::atoi(argv[1]) : 128;
    if (size <= 0) {
        return 2;
    }
    const int m = size;
    const int n = size;
    const int k = size;
    const float alpha = 1.25f;
    const float beta = 0.5f;
    const std::size_t bytes = static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * sizeof(float);

    float* host_a = static_cast<float*>(std::malloc(bytes));
    float* host_b = static_cast<float*>(std::malloc(bytes));
    float* host_c = static_cast<float*>(std::malloc(bytes));
    float* output = static_cast<float*>(std::malloc(bytes));
    if (!host_a || !host_b || !host_c || !output) {
        return 3;
    }
    for (int column = 0; column < size; ++column) {
        for (int row = 0; row < size; ++row) {
            host_a[row + column * size] = value_at(row, column, size);
            host_b[row + column * size] = value_at(column, row, size) * 0.5f;
            host_c[row + column * size] = value_at(row + column, row, size) * 0.25f;
        }
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    if (cudaMalloc(&device_a, bytes) != cudaSuccess) return 10;
    if (cudaMalloc(&device_b, bytes) != cudaSuccess) return 11;
    if (cudaMalloc(&device_c, bytes) != cudaSuccess) return 12;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 13;
    if (cudaMemcpyAsync(device_a, host_a, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return 14;
    if (cudaMemcpyAsync(device_b, host_b, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return 15;
    if (cudaMemcpyAsync(device_c, host_c, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return 16;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 17;
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return 18;
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, device_a, m, device_b, k, &beta, device_c, m) != CUBLAS_STATUS_SUCCESS) return 19;
    if (cudaMemcpyAsync(output, device_c, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 20;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 21;

    double max_error = 0.0;
    int checked = size <= 256 ? size : 32;
    for (int column = 0; column < checked; ++column) {
        for (int row = 0; row < checked; ++row) {
            double sum = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                sum += static_cast<double>(host_a[row + inner * size]) *
                    static_cast<double>(host_b[inner + column * size]);
            }
            double expected = static_cast<double>(alpha) * sum +
                static_cast<double>(beta) * static_cast<double>(host_c[row + column * size]);
            double error = std::fabs(static_cast<double>(output[row + column * size]) - expected);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    double tolerance = size <= 256 ? 1e-3 : 2e-2;
    if (max_error > tolerance) {
        std::printf("max_error=%0.8f tolerance=%0.8f\n", max_error, tolerance);
        return 30;
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(device_c);
    cudaFree(device_b);
    cudaFree(device_a);
    std::free(output);
    std::free(host_c);
    std::free(host_b);
    std::free(host_a);
    std::printf("cublas_sgemm_size=%d max_error=%0.8f\n", size, max_error);
    return 0;
}
