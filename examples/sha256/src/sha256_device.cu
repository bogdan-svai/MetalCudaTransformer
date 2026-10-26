#include "../include/sha256_types.h"

__device__ WORD rotate_right(WORD value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

__device__ WORD choose_word(WORD x, WORD y, WORD z) {
    return (x & y) ^ ((~x) & z);
}

__device__ WORD majority_word(WORD x, WORD y, WORD z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ WORD sigma_upper_zero(WORD value) {
    return rotate_right(value, 2) ^ rotate_right(value, 13) ^ rotate_right(value, 22);
}

__device__ WORD sigma_upper_one(WORD value) {
    return rotate_right(value, 6) ^ rotate_right(value, 11) ^ rotate_right(value, 25);
}

__device__ WORD sigma_lower_zero(WORD value) {
    return rotate_right(value, 7) ^ rotate_right(value, 18) ^ (value >> 3);
}

__device__ WORD sigma_lower_one(WORD value) {
    return rotate_right(value, 17) ^ rotate_right(value, 19) ^ (value >> 10);
}

__device__ void sha256_compress(const BYTE* block, SHA256_STATE* result) {
    const WORD constants[SHA256_ROUNDS] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    WORD schedule[SHA256_ROUNDS];
    for (int i = 0; i < 16; ++i) {
        int offset = i * 4;
        schedule[i] = ((WORD)block[offset] << 24)
            | ((WORD)block[offset + 1] << 16)
            | ((WORD)block[offset + 2] << 8)
            | (WORD)block[offset + 3];
    }
    for (int i = 16; i < SHA256_ROUNDS; ++i) {
        schedule[i] = sigma_lower_one(schedule[i - 2]) + schedule[i - 7]
            + sigma_lower_zero(schedule[i - 15]) + schedule[i - 16];
    }

    WORD a = 0x6a09e667;
    WORD b = 0xbb67ae85;
    WORD c = 0x3c6ef372;
    WORD d = 0xa54ff53a;
    WORD e = 0x510e527f;
    WORD f = 0x9b05688c;
    WORD g = 0x1f83d9ab;
    WORD h = 0x5be0cd19;

    for (int i = 0; i < SHA256_ROUNDS; ++i) {
        WORD temporary_one = h + sigma_upper_one(e) + choose_word(e, f, g)
            + constants[i] + schedule[i];
        WORD temporary_two = sigma_upper_zero(a) + majority_word(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }

    result->words[0] = 0x6a09e667 + a;
    result->words[1] = 0xbb67ae85 + b;
    result->words[2] = 0x3c6ef372 + c;
    result->words[3] = 0xa54ff53a + d;
    result->words[4] = 0x510e527f + e;
    result->words[5] = 0x9b05688c + f;
    result->words[6] = 0x1f83d9ab + g;
    result->words[7] = 0x5be0cd19 + h;
}

__device__ void sha256_write_digest(SHA256_STATE* state, BYTE* output) {
    for (int word = 0; word < 8; ++word) {
        output[word * 4] = (BYTE)(state->words[word] >> 24);
        output[word * 4 + 1] = (BYTE)(state->words[word] >> 16);
        output[word * 4 + 2] = (BYTE)(state->words[word] >> 8);
        output[word * 4 + 3] = (BYTE)state->words[word];
    }
}
