#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_value(int* values, int amount) {
    values[threadIdx.x] += amount;
}

int main() {
    int input[64] = {};
    int output[64] = {};
    int* device = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    cudaGraphNode_t first_node = nullptr;
    cudaGraphNode_t second_node = nullptr;
    int first_amount = 2;
    int second_amount = 5;

    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 10;
    if (cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess) return 11;
    if (cudaGraphCreate(&graph, 0) != cudaSuccess) return 12;

    void* first_args[] = {&device, &first_amount};
    cudaKernelNodeParams first = {};
    first.func = (void*)add_value;
    first.gridDim = dim3(1);
    first.blockDim = dim3(64);
    first.kernelParams = first_args;
    if (cudaGraphAddKernelNode(&first_node, graph, nullptr, 0, &first) != cudaSuccess) return 13;

    void* second_args[] = {&device, &second_amount};
    cudaKernelNodeParams second = {};
    second.func = (void*)add_value;
    second.gridDim = dim3(1);
    second.blockDim = dim3(64);
    second.kernelParams = second_args;
    if (cudaGraphAddKernelNode(&second_node, graph, &first_node, 1, &second) != cudaSuccess) return 14;

    if (cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0) != cudaSuccess) return 15;
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (cudaGraphLaunch(executable_graph, nullptr) != cudaSuccess) return 16;
    }
    if (cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 17;
    for (int index = 0; index < 64; ++index) {
        if (output[index] != 21) return 18;
    }

    if (cudaGraphExecDestroy(executable_graph) != cudaSuccess) return 19;
    if (cudaGraphDestroy(graph) != cudaSuccess) return 20;
    cudaFree(device);
    std::printf("status=pass graph_value=%d\n", output[0]);
    return 0;
}
