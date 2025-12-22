#pragma once
#include <cstdint>

// secp256k1 prime p words
__device__ void scalar_mul(const Big256 scalar, PointJ* res);
__device__ inline void get_secp256k1_p(uint32_t p[8]) {
    static const uint32_t p_words[8] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    for (int i = 0; i < 8; i++) p[i] = p_words[i];
}

// Curve base point G (compressed format ready)
__device__ __constant__ uint32_t Gx_const[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};
__device__ __constant__ uint32_t Gy_const[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

// exponent p-2 for inversion (Fermat's little theorem)
__device__ __constant__ uint32_t EXP_P_MINUS_2[8] = {
    0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};
