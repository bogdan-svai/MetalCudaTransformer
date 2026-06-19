#include <iostream>

// Prototipurile kernel-urilor pentru Linker
__global__ void adunareVectori(float* A, float* B, float* C, int n);
__global__ void inmultireMatrice(float* A, float* B, float* C, int N);

void ruleazaPipelineCalcul() {
    int dimensiune = 1024;
    // Simulăm un apel de tip bridge pe care parserul îl interceptează dinamic
    cudaLaunch(adunareVectori, 1, 256, dimensiune);
    cudaLaunch(inmultireMatrice, 16, 16, dimensiune);
}
