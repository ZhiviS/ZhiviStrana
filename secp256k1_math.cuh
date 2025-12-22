#ifndef SECP256K1_MATH_CUH
#define SECP256K1_MATH_CUH
#include "secp256k1_math.cuh"
typedef uint32_t Big256[8];

typedef struct {
    Big256 X, Y, Z;
} PointJ;

// Объявления с extern "C" для совместимости
__device__ void scalar_mul(const Big256 scalar, PointJ* res);

extern "C" __global__ void init_base_pub_kernel(const Big256 d_min, PointJ* out_base_pub);

extern "C" __global__ void search_kernel_incremental(
    PointJ base_pub_jac_in,
    const Big256 step,
    uint64_t start_offset,
    uint64_t iterations_per_thread,
    const uint8_t* d_target_x_prefix,
    int prefix_len,
    uint8_t target_prefix,
    unsigned long long* d_matches
);

#endif
