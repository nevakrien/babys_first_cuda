#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NUM_MATRIX_PAIRS 1000
#define MIN_ROWS 1
#define MAX_ROWS 1000
#define MIN_COLS 1
#define MAX_COLS 1000

// Function for generating a random matrix
matrix generateRandomMatrix(int rows, int cols) {
    matrix m;
    m.data = (float *)malloc(rows * cols * sizeof(float));
    m.rows = rows;
    m.cols = cols;
    for (int i = 0; i < rows * cols; i++) {
        m.data[i] = (float)rand() / RAND_MAX * 100.0;  // Random float between 0 and 100
    }
    return m;
}

// Function for random number generation in a range
int randomInRange(int min, int max) {
    return min + rand() % (max - min + 1);
}

int main() {
    srand(time(NULL));  // Initialize random number generator

    FILE *file_a = fopen("matrices_a.bin", "wb");
    FILE *file_b = fopen("matrices_b.bin", "wb");
    FILE *file_ans = fopen("matrices_ans.bin", "wb");

    if (!file_a || !file_b || !file_ans) {
        // Handle file opening errors
        perror("Error opening file");
        return 1;
    }

    for (int i = 0; i < NUM_MATRIX_PAIRS; i++) {
        int rows_a = randomInRange(MIN_ROWS, MAX_ROWS);
        int cols_a = randomInRange(MIN_COLS, MAX_COLS);
        int rows_b = cols_a;  // Ensuring multiplication is possible
        int cols_b = randomInRange(MIN_COLS, MAX_COLS);

        matrix a = generateRandomMatrix(rows_a, cols_a);
        matrix b = generateRandomMatrix(rows_b, cols_b);
        matrix ans = matrixMul(a, b);

        writeMatrix(file_a, &a);
        writeMatrix(file_b, &b);
        writeMatrix(file_ans, &ans);

        // Free dynamically allocated memory
        free(a.data);
        free(b.data);
        free(ans.data);

        // Update the progress indicator
        printf("\rProcessing matrix pair %d out of %d", i + 1, NUM_MATRIX_PAIRS);
        fflush(stdout);  // Ensures the output is printed immediately
    }
    printf("\n");  // Move to the next line after the progress is complete

    fclose(file_a);
    fclose(file_b);
    fclose(file_ans);

    return 0;
}
