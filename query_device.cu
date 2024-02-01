#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp properties;

    cudaGetDevice(&device); // Get device ID
    cudaGetDeviceProperties(&properties, device); // Get device properties

    std::cout << "Maximum threads per block: " << properties.maxThreadsPerBlock << std::endl;

    return 0;
}
