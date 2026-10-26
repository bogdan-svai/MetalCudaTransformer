#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_capture_value(int* values, int amount) {
    values[threadIdx.x] += amount;
}

int main() {
    int input[32] = {};
    int output[32] = {};
    int* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    int amount = 5;

    for (int index = 0; index < 32; ++index) {
        input[index] = index;
    }
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 11;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) return 12;
    if (cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream) != cudaSuccess) return 13;
    add_capture_value<<<1, 32, 0, stream>>>(device, amount);
    if (cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return 14;
    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) return 15;
    if (cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0) != cudaSuccess) return 16;
    if (cudaGraphLaunch(executable_graph, stream) != cudaSuccess) return 17;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 18;

    for (int index = 0; index < 32; ++index) {
        if (output[index] != index + amount) return 19;
    }
    cudaGraphExecDestroy(executable_graph);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(device);
    std::printf("status=pass graph_capture=%d\n", output[31]);
    return 0;
}
