__global__ void matriceActivareKernel(float* matrice, int latura, float bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < latura && col < latura) {
        int pozitie = row * latura + col;
        matrice[pozitie] = matrice[pozitie] + bias;
    }
}
