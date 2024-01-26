#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_mul.hpp"
#include <cmath>


#ifndef TestFunction
    #define TestFunction matrixMul
#endif

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))

int main() {
    matrix a, b, ans, y;
    FILE *file_a = fopen("matrices_a.bin", "rb");
    FILE *file_b = fopen("matrices_b.bin", "rb");
    FILE *file_ans = fopen("matrices_ans.bin", "rb");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0.0;

    SHOW_DEFINE(TestFunction);

    float totalDist = 0.0f;
    float maxDist = 0.0f;
    int count = 0;

    while (true) {
        printf("\rMultiplying matrix pair %d", count);

        if (readMatrix(file_a, &a) || readMatrix(file_b, &b) || readMatrix(file_ans, &ans)) {
            if (feof(file_a) && feof(file_b) && feof(file_ans)) {
                break; // End of files
            }
            printf("\nInput files don't match or read error\n");
            break;
        }

        cudaEventRecord(start);
        y = TestFunction(a, b);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        float dist = distHeuristic(y, ans);
        totalDist += dist;
        maxDist = max(dist, maxDist);
        if (isnan(dist)) {
            printf("\nWarning: Distance is NaN\n");
        }
        else if (dist > 0.05) {
            printf("\nWarning: High Discrepancy Detected - Distance: %f\n", dist);
        }

        count++;
    }

    if (count > 0) {
        printf("\nAverage Distance: %f\n", totalDist / count);
        printf("Maximum Distance: %f\n", maxDist);
        printf("Total multiplication time: %f milliseconds\n", totalTime);
    } else {
        printf("\nNo matrix pairs were processed.\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    fclose(file_a);
    fclose(file_b);
    fclose(file_ans);

    return 0;
}
