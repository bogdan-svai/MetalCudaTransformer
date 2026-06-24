#include "sha256.h"

#include <climits>
#include <cstdio>
#include <cstring>

__global__ void hash_leaves(const BYTE* records, const int* lengths, BYTE* hashes, int count);
__global__ void merkle_round(const BYTE* input, BYTE* output, int count);
__global__ void find_nonce(const BYTE* root, WORD limit, WORD* best);
__global__ void hash_block(const BYTE* root, WORD nonce, BYTE* output);

static void print_hex(const char* name, const BYTE* value) {
    std::printf("%s=", name);
    for (int index = 0; index < SHA256_DIGEST_SIZE; ++index) std::printf("%02x", value[index]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: miner <input-file>\n");
        return 2;
    }
    std::FILE* input = std::fopen(argv[1], "r");
    if (!input) return 3;
    int count = 0;
    WORD nonce_limit = 0;
    if (std::fscanf(input, "%d %u", &count, &nonce_limit) != 2 || count < 1 || count > STAGE9_MAX_RECORDS) {
        std::fclose(input);
        return 4;
    }
    BYTE records[STAGE9_MAX_RECORDS * STAGE9_RECORD_SIZE] = {0};
    int lengths[STAGE9_MAX_RECORDS] = {0};
    for (int index = 0; index < count; ++index) {
        char value[STAGE9_RECORD_SIZE + 1] = {0};
        if (std::fscanf(input, "%32s", value) != 1) {
            std::fclose(input);
            return 5;
        }
        lengths[index] = (int)std::strlen(value);
        std::memcpy(records + index * STAGE9_RECORD_SIZE, value, lengths[index]);
    }
    std::fclose(input);

    BYTE* device_records = nullptr;
    int* device_lengths = nullptr;
    BYTE* device_a = nullptr;
    BYTE* device_b = nullptr;
    WORD* device_nonce = nullptr;
    BYTE* device_block_hash = nullptr;
    if (cudaMalloc(&device_records, sizeof(records)) != cudaSuccess) return 10;
    if (cudaMalloc(&device_lengths, sizeof(lengths)) != cudaSuccess) return 11;
    if (cudaMalloc(&device_a, STAGE9_MAX_RECORDS * SHA256_DIGEST_SIZE) != cudaSuccess) return 12;
    if (cudaMalloc(&device_b, STAGE9_MAX_RECORDS * SHA256_DIGEST_SIZE) != cudaSuccess) return 13;
    if (cudaMalloc(&device_nonce, sizeof(WORD)) != cudaSuccess) return 14;
    if (cudaMalloc(&device_block_hash, SHA256_DIGEST_SIZE) != cudaSuccess) return 15;
    cudaMemcpy(device_records, records, sizeof(records), cudaMemcpyHostToDevice);
    cudaMemcpy(device_lengths, lengths, sizeof(lengths), cudaMemcpyHostToDevice);

    hash_leaves<<<1, 32>>>(device_records, device_lengths, device_a, count);
    BYTE first_leaf[SHA256_DIGEST_SIZE] = {0};
    cudaMemcpy(first_leaf, device_a, sizeof(first_leaf), cudaMemcpyDeviceToHost);
    int current = count;
    while (current > 1) {
        merkle_round<<<1, 32>>>(device_a, device_b, current);
        BYTE* temporary = device_a;
        device_a = device_b;
        device_b = temporary;
        current = (current + 1) / 2;
    }

    BYTE root[SHA256_DIGEST_SIZE] = {0};
    cudaMemcpy(root, device_a, sizeof(root), cudaMemcpyDeviceToHost);
    WORD nonce = UINT_MAX;
    cudaMemcpy(device_nonce, &nonce, sizeof(nonce), cudaMemcpyHostToDevice);
    find_nonce<<<32, 128>>>(device_a, nonce_limit, device_nonce);
    cudaMemcpy(&nonce, device_nonce, sizeof(nonce), cudaMemcpyDeviceToHost);
    if (nonce == UINT_MAX) return 20;
    hash_block<<<1, 1>>>(device_a, nonce, device_block_hash);
    BYTE block_hash[SHA256_DIGEST_SIZE] = {0};
    cudaMemcpy(block_hash, device_block_hash, sizeof(block_hash), cudaMemcpyDeviceToHost);

    cudaFree(device_block_hash);
    cudaFree(device_nonce);
    cudaFree(device_b);
    cudaFree(device_a);
    cudaFree(device_lengths);
    cudaFree(device_records);

    print_hex("sha256_first", first_leaf);
    print_hex("merkle_root", root);
    std::printf("nonce=%u\n", nonce);
    print_hex("block_hash", block_hash);
    return 0;
}
