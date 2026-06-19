__global__ void inmultireMatrice(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float valoare = 0;
        for (int i = 0; i < N; ++i) {
            valoare += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = valoare;
    }
}
