// MyLogic.cpp
#include "MyLogic.h"

#include "../Int.h"
#include "../Point.h"
#include "../SECP256K1.h"

#include <iostream>
#include <cstdint>
#include <string>

void RunMyLogic()
{
    // --- диапазон ---
    Int d_min, d_max;
    d_min.SetBase10((char*)"38528057173582953847011032336563488877335");
    d_max.SetBase10((char*)"38528057173582963485153971459144962127728");

    const uint64_t num_iters = 1000000; // тест

    // range = d_max - d_min
    Int range;
    range.Sub(&d_max, &d_min);

    // step = range / num_iters
    Int step;
    step.Set(&range);
    Int it((uint64_t)num_iters);
    step.Div(&it);

    // --- крипта ---
    Secp256K1 secp;
    secp.Init();

    Int sum_d;
    sum_d.SetInt32(0);

    uint64_t count = 0;

    Int d;
    d.Set(&d_min);

    for (uint64_t i = 0; i < num_iters; i++)
    {
        // Compute public key
        Point P = secp.ComputePublicKey(&d);

        // compressed pubkey hex
        std::string pubhex = secp.GetPublicKeyHex(true, P);

        // фильтр: 02 + 145d
        if (pubhex.size() >= 6 &&
            pubhex[0] == '0' && pubhex[1] == '2' &&
            pubhex[2] == '1' && pubhex[3] == '4' &&
            pubhex[4] == '5' && pubhex[5] == 'd')
        {
            sum_d.Add(&d);
            count++;
        }

        // d += step
        d.Add(&step);
    }

    if (count == 0) {
        std::cout << "No matches\n";
        return;
    }

    // mean = sum / count
    Int mean;
    mean.Set(&sum_d);
    Int cnt((uint64_t)count);
    mean.Div(&cnt);

    // mid = (d_min + d_max) / 2
    Int mid;
    mid.Add(&d_min, &d_max);
    mid.ShiftR(1);

    // rem = mean - mid
    Int rem;
    rem.Sub(&mean, &mid);

    std::cout << "Processed : " << num_iters << "\n";
    std::cout << "Matched   : " << count << "\n";
    std::cout << "Mean d    : " << mean.GetBase10() << "\n";
    std::cout << "Mid d     : " << mid.GetBase10() << "\n";
    std::cout << "Rem       : " << rem.GetBase10() << "\n";
}
