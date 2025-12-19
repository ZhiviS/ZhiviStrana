#include <cuda_runtime.h>
#include <cstdio>

__global__ void MyKernel()
{
    // временно пусто
}

void LaunchMyKernel()
{
    MyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
