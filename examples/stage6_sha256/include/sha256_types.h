#ifndef MCU_STAGE6_SHA256_TYPES_H
#define MCU_STAGE6_SHA256_TYPES_H

typedef unsigned char BYTE;
typedef unsigned int WORD;

#define SHA256_ROUNDS 64
#define SHA256_DIGEST_SIZE 32

typedef struct {
    WORD words[8];
} SHA256_STATE;

#endif
