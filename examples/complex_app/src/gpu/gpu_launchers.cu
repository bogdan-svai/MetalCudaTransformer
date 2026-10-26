__global__ void scaleazaVectorKernel(float* valori, int elemente, float factor);
__global__ void matriceActivareKernel(float* matrice, int latura, float bias);

int ruleazaKerneluriGpu(int elemente, int laturaMatrice, float factor) {
    float* bufferVector = 0;
    float* bufferMatrice = 0;
    int blocuriVector = (elemente + 255) / 256;
    dim3 blocuriMatrice((laturaMatrice + 31) / 32, (laturaMatrice + 31) / 32);
    dim3 fireMatrice(32, 32);

    cudaLaunch(scaleazaVectorKernel, blocuriVector, 256, bufferVector, elemente, factor);
    cudaLaunch(matriceActivareKernel, blocuriMatrice, fireMatrice, bufferMatrice, laturaMatrice, factor);

    return blocuriVector + blocuriMatrice.x * blocuriMatrice.y;
}
