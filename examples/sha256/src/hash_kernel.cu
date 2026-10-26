#include "../include/sha256_types.h"

__device__ void sha256_compress(const BYTE* block, SHA256_STATE* result);
__device__ void sha256_write_digest(SHA256_STATE* state, BYTE* output);

__global__ void sha256_kernel(BYTE* block, BYTE* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        SHA256_STATE state;
        sha256_compress(block, &state);
        sha256_write_digest(&state, output);
    }
}
