#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_mul.hpp"

// Check if TestFunction is already defined
#ifndef TestFunction
    // Define the default function if TestFunction is not already defined
    #define TestFunction matrixMul
#endif

#define STR(x)   #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))


int main() {
    int i = 0;
    matrix a, b, ans, y;

    FILE *file_a = fopen("matrices_a.bin", "rb");
    FILE *file_b = fopen("matrices_b.bin", "rb");
    FILE *file_ans = fopen("matrices_ans.bin", "rb");

    // Define CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0.0;

    // Print the function being used for testing
    SHOW_DEFINE(TestFunction);

    while (true) {
        printf("\rMultiplying matrix pair %d", i);

        if (readMatrix(file_a, &a)) {
            //printf("\nread error\n");
            break;
        }

        if (readMatrix(file_b, &b)) {
            printf("\nInput files dont match\n");
            break;
        }

        if (readMatrix(file_ans, &ans)) {
            printf("\nInput files dont match\n");
            break;
        }

        // Start timing here.
        cudaEventRecord(start);

        y = TestFunction(a, b);

        // Stop timing here.
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        float dist=distHeuristic(y, ans);
        if ( dist> 0.05) {
            printf("\nWrong Data dist is at:%f\n",dist);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
            return 1;
        }
        i += 1;
    }

    printf("\nTotal multiplication time: %f milliseconds\n", totalTime);

    // Destroy CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
