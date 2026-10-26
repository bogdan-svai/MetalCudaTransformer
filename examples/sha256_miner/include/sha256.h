#ifndef MCU_SHA256_SHA256_H
#define MCU_SHA256_SHA256_H

typedef unsigned char BYTE;
typedef unsigned int WORD;

#define SHA256_DIGEST_SIZE 32
#define SHA256_BLOCK_SIZE 64
#define SHA256_RECORD_SIZE 32
#define SHA256_MAX_RECORDS 16

typedef struct {
    WORD words[8];
} SHA256_STATE;

__device__ void sha256_bytes(const BYTE* input, int length, BYTE* output);

#endif
