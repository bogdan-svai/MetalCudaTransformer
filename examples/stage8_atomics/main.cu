__global__ void unsigned_operations(
    unsigned int* add_value,
    unsigned int* minimum,
    unsigned int* maximum,
    unsigned int* exchanged,
    unsigned int* compared
) {
    atomicAdd(add_value, 1u);
    atomicMin(minimum, threadIdx.x);
    atomicMax(maximum, threadIdx.x);
    atomicExch(exchanged, 7u);
    atomicCAS(compared, 3u, 9u);
}

__global__ void signed_operations(int* minimum, int* maximum) {
    int value = static_cast<int>(threadIdx.x) - 16;
    atomicMin(minimum, value);
    atomicMax(maximum, value);
}

int main() {
    return 0;
}
