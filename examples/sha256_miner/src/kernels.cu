#include "sha256.h"

__global__ void hash_leaves(const BYTE* records, const int* lengths, BYTE* hashes, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        sha256_bytes(records + index * SHA256_RECORD_SIZE, lengths[index], hashes + index * SHA256_DIGEST_SIZE);
    }
}

__global__ void merkle_round(const BYTE* input, BYTE* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int parents = (count + 1) / 2;
    if (index < parents) {
        BYTE pair[64];
        int left = index * 2;
        int right = left + 1 < count ? left + 1 : left;
        for (int byte = 0; byte < 32; ++byte) {
            pair[byte] = input[left * 32 + byte];
            pair[32 + byte] = input[right * 32 + byte];
        }
        sha256_bytes(pair, 64, output + index * 32);
    }
}

__global__ void find_nonce(const BYTE* root, WORD limit, WORD* best) {
    WORD index = blockIdx.x * blockDim.x + threadIdx.x;
    WORD stride = blockDim.x * gridDim.x;
    for (WORD nonce = index; nonce < limit; nonce += stride) {
        BYTE candidate[36];
        BYTE digest[32];
        for (int byte = 0; byte < 32; ++byte) candidate[byte] = root[byte];
        candidate[32] = (BYTE)(nonce >> 24);
        candidate[33] = (BYTE)(nonce >> 16);
        candidate[34] = (BYTE)(nonce >> 8);
        candidate[35] = (BYTE)nonce;
        sha256_bytes(candidate, 36, digest);
        if (digest[0] == 0 && digest[1] == 0) atomicMin(best, nonce);
    }
}

__global__ void hash_block(const BYTE* root, WORD nonce, BYTE* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        BYTE candidate[36];
        for (int byte = 0; byte < 32; ++byte) candidate[byte] = root[byte];
        candidate[32] = (BYTE)(nonce >> 24);
        candidate[33] = (BYTE)(nonce >> 16);
        candidate[34] = (BYTE)(nonce >> 8);
        candidate[35] = (BYTE)nonce;
        sha256_bytes(candidate, 36, output);
    }
}
