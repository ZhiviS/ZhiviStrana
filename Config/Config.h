#pragma once
#include <string>
#include <cstdint>

struct Config
{
    std::string pub_key_x;
    std::string target_prefix;

    std::string d_min;
    std::string d_max;

    uint64_t num_iters = 0;
    uint32_t num_threads = 1;
    std::string mode;

    bool Load(const std::string& filename);
};
