#include "../include/sha256_types.h"
#include <cstdio>
#include <cstring>

__global__ void sha256_kernel(BYTE* block, BYTE* output);

int main() {
    BYTE padded_block[64] = {0};
    padded_block[0] = 'a';
    padded_block[1] = 'b';
    padded_block[2] = 'c';
    padded_block[3] = 0x80;
    padded_block[63] = 24;

    const BYTE expected[SHA256_DIGEST_SIZE] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    BYTE digest[SHA256_DIGEST_SIZE] = {0};
    BYTE* device_block = nullptr;
    BYTE* device_digest = nullptr;

    if (cudaMalloc(&device_block, sizeof(padded_block)) != cudaSuccess) return 10;
    if (cudaMalloc(&device_digest, sizeof(digest)) != cudaSuccess) return 11;
    if (cudaMemcpy(device_block, padded_block, sizeof(padded_block), cudaMemcpyHostToDevice) != cudaSuccess) return 12;
    sha256_kernel<<<1, 1>>>(device_block, device_digest);
    if (cudaGetLastError() != cudaSuccess) return 13;
    if (cudaMemcpy(digest, device_digest, sizeof(digest), cudaMemcpyDeviceToHost) != cudaSuccess) return 14;
    cudaFree(device_digest);
    cudaFree(device_block);

    if (std::memcmp(digest, expected, sizeof(digest)) != 0) return 15;
    std::printf("sha256_abc=ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n");
    return 0;
}
