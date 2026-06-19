#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void numericTransform(float* values, int count, float scale, float bias) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        values[index] = values[index] * scale + bias;
    }
}

int main() {
    const int count = 100000;
    const int block = 2200;
    const int grid = (count + block - 1) / block;
    const float scale = 1.25f;
    const float bias = 0.5f;
    std::vector<float> input(count);
    std::vector<float> expected(count);
    std::vector<float> output(count);

    for (int index = 0; index < count; ++index) {
        input[index] = static_cast<float>((index % 1000) - 500) / 32.0f;
        expected[index] = input[index] * scale + bias;
    }

    float* device_values = nullptr;
    if (cudaMalloc(&device_values, input.size() * sizeof(float)) != cudaSuccess) {
        return 10;
    }
    if (cudaMemcpy(device_values, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return 11;
    }
    numericTransform<<<grid, block>>>(device_values, count, scale, bias);
    if (cudaGetLastError() != cudaSuccess) {
        return 12;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 13;
    }
    if (cudaMemcpy(output.data(), device_values, output.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 14;
    }
    cudaFree(device_values);

    float maximum_error = 0.0f;
    for (int index = 0; index < count; ++index) {
        maximum_error = std::max(maximum_error, std::fabs(output[index] - expected[index]));
    }
    if (maximum_error > 1e-6f) {
        std::cerr << "maximum_error=" << maximum_error << '\n';
        return 15;
    }

    std::vector<double> allocation_microseconds;
    allocation_microseconds.reserve(200);
    for (int iteration = 0; iteration < 210; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        void* pointer = nullptr;
        cudaError_t allocated = cudaMalloc(&pointer, 4096);
        auto stop = std::chrono::steady_clock::now();
        if (allocated != cudaSuccess || cudaFree(pointer) != cudaSuccess) {
            return 16;
        }
        if (iteration >= 10) {
            allocation_microseconds.push_back(
                std::chrono::duration<double, std::micro>(stop - start).count()
            );
        }
    }
    std::sort(allocation_microseconds.begin(), allocation_microseconds.end());
    double p95_us = allocation_microseconds[static_cast<std::size_t>(allocation_microseconds.size() * 0.95)];
    if (p95_us > 2000.0) {
        return 17;
    }

    std::cout << "numeric_count=" << count
              << " max_error=" << maximum_error
              << " allocation_p95_us=" << p95_us << '\n';
    return 0;
}
