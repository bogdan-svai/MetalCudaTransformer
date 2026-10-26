#include "sha256.h"

__device__ WORD rotate_right(WORD value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

__device__ void sha256_initialize(SHA256_STATE* state) {
    state->words[0] = 0x6a09e667u;
    state->words[1] = 0xbb67ae85u;
    state->words[2] = 0x3c6ef372u;
    state->words[3] = 0xa54ff53au;
    state->words[4] = 0x510e527fu;
    state->words[5] = 0x9b05688cu;
    state->words[6] = 0x1f83d9abu;
    state->words[7] = 0x5be0cd19u;
}

__device__ void sha256_compress(const BYTE* block, SHA256_STATE* state) {
    const WORD constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    WORD schedule[64];
    for (int index = 0; index < 16; ++index) {
        int offset = index * 4;
        schedule[index] = ((WORD)block[offset] << 24)
            | ((WORD)block[offset + 1] << 16)
            | ((WORD)block[offset + 2] << 8)
            | (WORD)block[offset + 3];
    }
    for (int index = 16; index < 64; ++index) {
        WORD x = schedule[index - 15];
        WORD y = schedule[index - 2];
        WORD sigma_zero = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
        WORD sigma_one = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10);
        schedule[index] = schedule[index - 16] + sigma_zero + schedule[index - 7] + sigma_one;
    }

    WORD a = state->words[0];
    WORD b = state->words[1];
    WORD c = state->words[2];
    WORD d = state->words[3];
    WORD e = state->words[4];
    WORD f = state->words[5];
    WORD g = state->words[6];
    WORD h = state->words[7];
    for (int index = 0; index < 64; ++index) {
        WORD upper_one = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        WORD choose = (e & f) ^ ((~e) & g);
        WORD temporary_one = h + upper_one + choose + constants[index] + schedule[index];
        WORD upper_zero = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        WORD majority = (a & b) ^ (a & c) ^ (b & c);
        WORD temporary_two = upper_zero + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }
    state->words[0] += a;
    state->words[1] += b;
    state->words[2] += c;
    state->words[3] += d;
    state->words[4] += e;
    state->words[5] += f;
    state->words[6] += g;
    state->words[7] += h;
}

__device__ void sha256_bytes(const BYTE* input, int length, BYTE* output) {
    BYTE padded[128] = {0};
    for (int index = 0; index < length; ++index) padded[index] = input[index];
    padded[length] = 0x80u;
    int blocks = length < 56 ? 1 : 2;
    WORD bit_length = (WORD)length * 8u;
    int tail = blocks * SHA256_BLOCK_SIZE;
    padded[tail - 4] = (BYTE)(bit_length >> 24);
    padded[tail - 3] = (BYTE)(bit_length >> 16);
    padded[tail - 2] = (BYTE)(bit_length >> 8);
    padded[tail - 1] = (BYTE)bit_length;

    SHA256_STATE state;
    sha256_initialize(&state);
    sha256_compress(padded, &state);
    if (blocks == 2) sha256_compress(padded + SHA256_BLOCK_SIZE, &state);
    for (int word = 0; word < 8; ++word) {
        output[word * 4] = (BYTE)(state.words[word] >> 24);
        output[word * 4 + 1] = (BYTE)(state.words[word] >> 16);
        output[word * 4 + 2] = (BYTE)(state.words[word] >> 8);
        output[word * 4 + 3] = (BYTE)state.words[word];
    }
}
