#include <metal_stdlib>
using namespace metal;

constant uint k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

inline uint rotr(uint value, uint bits) { return rotate(value, 32u - bits); }

inline uint sha256_first_word(const device uchar* root, uint nonce) {
    uint w[64];
    for (uint i = 0; i < 8; ++i) {
        uint o = i * 4;
        w[i] = (uint(root[o]) << 24) | (uint(root[o + 1]) << 16) |
               (uint(root[o + 2]) << 8) | uint(root[o + 3]);
    }
    w[8] = nonce;
    w[9] = 0x80000000u;
    for (uint i = 10; i < 15; ++i) w[i] = 0u;
    w[15] = 288u;
    for (uint i = 16; i < 64; ++i) {
        uint x = w[i - 15], y = w[i - 2];
        uint s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
        uint s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au;
    uint e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    for (uint i = 0; i < 64; ++i) {
        uint s1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint t1 = h + s1 + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
        uint s0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    return a + 0x6a09e667u;
}

kernel void native_find_nonce(
    const device uchar* root [[buffer(0)]],
    constant uint& limit [[buffer(1)]],
    device atomic_uint* best [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < limit && (sha256_first_word(root, gid) >> 16) == 0u)
        atomic_fetch_min_explicit(best, gid, memory_order_relaxed);
}
