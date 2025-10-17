#include <cstdio>
#include "cuda_runtime.h"

__global__ void hello() {
    printf("Hello world from GPU\n");
}

int main() {
    int deviceCnt = 0;
    cudaGetDeviceCount(&deviceCnt);
    if (deviceCnt == 0) {
        printf("No GPU available :(\n");
        return 1;
    }
    printf("%d GPU(s) available :)\n", deviceCnt);
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}