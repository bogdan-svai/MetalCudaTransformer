// HotSpot-style five-point thermal stencil derived from the Rodinia workload.
// This is an MCU CUDA Core compatibility variant, not an official Rodinia run.

#include <cmath>
#include <cstdio>
#include <cstdlib>

__global__ void hotspot_step(
    const float* input,
    const float* power,
    float* output,
    int width,
    int height,
    float diffusion,
    float power_scale,
    float ambient_scale,
    float ambient
) {
    __shared__ float tile[324];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int clamped_x = x < width ? x : width - 1;
    int clamped_y = y < height ? y : height - 1;
    int local = (ty + 1) * 18 + tx + 1;
    tile[local] = input[clamped_y * width + clamped_x];

    if (tx == 0) {
        int west = clamped_x > 0 ? clamped_x - 1 : 0;
        tile[(ty + 1) * 18] = input[clamped_y * width + west];
    }
    if (tx == 15) {
        int east = clamped_x + 1 < width ? clamped_x + 1 : width - 1;
        tile[(ty + 1) * 18 + 17] = input[clamped_y * width + east];
    }
    if (ty == 0) {
        int north = clamped_y > 0 ? clamped_y - 1 : 0;
        tile[tx + 1] = input[north * width + clamped_x];
    }
    if (ty == 15) {
        int south = clamped_y + 1 < height ? clamped_y + 1 : height - 1;
        tile[17 * 18 + tx + 1] = input[south * width + clamped_x];
    }
    __syncthreads();

    if (x < width && y < height) {
        float center = tile[local];
        float north = tile[local - 18];
        float south = tile[local + 18];
        float west = tile[local - 1];
        float east = tile[local + 1];
        float laplacian = north + south + west + east - 4.0f * center;
        output[y * width + x] = center
            + diffusion * laplacian
            + power_scale * power[y * width + x]
            + ambient_scale * (ambient - center);
    }
}

static float initial_temperature(int x, int y) {
    return 60.0f + (float)((x * 13 + y * 7) % 101) * 0.125f;
}

static float initial_power(int x, int y) {
    return 0.25f + (float)((x * 3 + y * 11) % 29) * 0.03125f;
}

int main(int argc, char** argv) {
    int width = argc > 1 ? std::atoi(argv[1]) : 1024;
    int height = argc > 2 ? std::atoi(argv[2]) : width;
    int iterations = argc > 3 ? std::atoi(argv[3]) : 50;
    if (width <= 0 || height <= 0 || iterations <= 0) return 2;
    int count = width * height;
    size_t bytes = (size_t)count * sizeof(float);

    float* host_input = (float*)std::malloc(bytes);
    float* host_power = (float*)std::malloc(bytes);
    float* cpu_a = (float*)std::malloc(bytes);
    float* cpu_b = (float*)std::malloc(bytes);
    float* observed = (float*)std::malloc(bytes);
    if (!host_input || !host_power || !cpu_a || !cpu_b || !observed) return 3;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            host_input[index] = initial_temperature(x, y);
            host_power[index] = initial_power(x, y);
            cpu_a[index] = host_input[index];
        }
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_power = nullptr;
    if (cudaMalloc(&device_a, bytes) != cudaSuccess) return 10;
    if (cudaMalloc(&device_b, bytes) != cudaSuccess) return 11;
    if (cudaMalloc(&device_power, bytes) != cudaSuccess) return 12;
    if (cudaMemcpy(device_a, host_input, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 13;
    if (cudaMemcpy(device_power, host_power, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 14;

    const float diffusion = 0.125f;
    const float power_scale = 0.0025f;
    const float ambient_scale = 0.0005f;
    const float ambient = 80.0f;
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        hotspot_step<<<blocks, threads>>>(
            device_a, device_power, device_b, width, height,
            diffusion, power_scale, ambient_scale, ambient
        );
        float* temporary = device_a;
        device_a = device_b;
        device_b = temporary;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 15;
    if (cudaMemcpy(observed, device_a, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 16;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int y = 0; y < height; ++y) {
            int north_y = y > 0 ? y - 1 : 0;
            int south_y = y + 1 < height ? y + 1 : height - 1;
            for (int x = 0; x < width; ++x) {
                int west_x = x > 0 ? x - 1 : 0;
                int east_x = x + 1 < width ? x + 1 : width - 1;
                int index = y * width + x;
                float center = cpu_a[index];
                float laplacian = cpu_a[north_y * width + x]
                    + cpu_a[south_y * width + x]
                    + cpu_a[y * width + west_x]
                    + cpu_a[y * width + east_x]
                    - 4.0f * center;
                cpu_b[index] = center
                    + diffusion * laplacian
                    + power_scale * host_power[index]
                    + ambient_scale * (ambient - center);
            }
        }
        float* temporary = cpu_a;
        cpu_a = cpu_b;
        cpu_b = temporary;
    }

    float maximum_error = 0.0f;
    double checksum = 0.0;
    for (int index = 0; index < count; ++index) {
        float error = std::fabs(observed[index] - cpu_a[index]);
        if (error > maximum_error) maximum_error = error;
        checksum += observed[index];
    }
    if (maximum_error > 0.001f) return 20;
    std::printf("ok=1\n");
    std::printf("width=%d\n", width);
    std::printf("height=%d\n", height);
    std::printf("iterations=%d\n", iterations);
    std::printf("max_error=%0.9g\n", maximum_error);
    std::printf("checksum=%0.12g\n", checksum);

    cudaFree(device_power);
    cudaFree(device_b);
    cudaFree(device_a);
    std::free(observed);
    std::free(cpu_b);
    std::free(cpu_a);
    std::free(host_power);
    std::free(host_input);
    return 0;
}
