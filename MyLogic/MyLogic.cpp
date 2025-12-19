#include "MyLogic.h"
#include <iostream>
#include <chrono>

void RunMyLogic(const Config& cfg)
{
    using clock = std::chrono::high_resolution_clock;

    std::cout << "[MyLogic] CPU test run\n";
    std::cout << "Iterations : " << cfg.num_iters << "\n";
    std::cout << "Threads    : " << cfg.num_threads << "\n\n";

    uint64_t iters = cfg.num_iters;
    uint64_t counter = 0;

    auto t0 = clock::now();

    for (uint64_t i = 0; i < iters; ++i) {
        counter++;
    }

    auto t1 = clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    double seconds = dt.count();
    double speed = counter / seconds;

    std::cout << "Processed : " << counter << "\n";
    std::cout << "Time (s)  : " << seconds << "\n";
    std::cout << "Speed    : " << speed << " iter/sec\n";
}
