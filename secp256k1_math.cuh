#pragma once
#include <cstdint>

typedef uint32_t Big256[8];

extern "C" __global__ void search_pubkeys(
    const Big256* priv_keys,
    uint8_t target_prefix,
    const uint8_t* target_x_prefix,
    int prefix_len,
    unsigned long long* matches
);
