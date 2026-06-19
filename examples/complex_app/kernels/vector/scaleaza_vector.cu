__global__ void scaleazaVectorKernel(float* valori, int elemente, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < elemente) {
        valori[idx] = valori[idx] * factor;
    }
}
