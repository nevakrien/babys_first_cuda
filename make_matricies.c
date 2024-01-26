#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_MATRICES 1000 // Number of matrices to generate
#define MAX_ROWS 100    // Maximum number of rows in a matrix
#define MAX_COLS 100    // Maximum number of columns in a matrix

void writeMatrix(FILE *file, float *matrix, int rows, int cols) {
    // Write matrix dimensions
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);
    
    // Write matrix data
    fwrite(matrix, sizeof(float), rows * cols, file);
}

void readMatrix(FILE *file, float **matrix, int *rows, int *cols) {
    // Read matrix dimensions
    fread(rows, sizeof(int), 1, file);
    fread(cols, sizeof(int), 1, file);

    // Allocate memory and read matrix data
    *matrix = (float *)malloc((*rows) * (*cols) * sizeof(float));
    fread(*matrix, sizeof(float), (*rows) * (*cols), file);
}

int main() {
    FILE *file;
    float *matrices[NUM_MATRICES];
    int rows[NUM_MATRICES], cols[NUM_MATRICES];

    // Seed random number generator
    srand(time(NULL));

    // Open file for writing
    file = fopen("matrices.bin", "wb");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    // Generate and write matrices
    for (int i = 0; i < NUM_MATRICES; i++) {
        rows[i] = rand() % MAX_ROWS + 1;
        cols[i] = rand() % MAX_COLS + 1;
        matrices[i] = (float *)malloc(rows[i] * cols[i] * sizeof(float));

        for (int j = 0; j < rows[i] * cols[i]; j++) {
            matrices[i][j] = (float)rand() / RAND_MAX;
        }

        writeMatrix(file, matrices[i], rows[i], cols[i]);
    }

    // Close file
    fclose(file);

    // Open file for reading
    file = fopen("matrices.bin", "rb");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    // Read and verify matrices
    for (int i = 0; i < NUM_MATRICES; i++) {
        float *M1;
        int readRows, readCols;
        readMatrix(file, &M1, &readRows, &readCols);

        if (readRows != rows[i] || readCols != cols[i]) {
            printf("Matrix dimensions do not match\n");
        } else {
            for (int j = 0; j < rows[i] * cols[i]; j++) {
                if (matrices[i][j] != M1[j]) {
                    printf("Matrix data does not match\n");
                    break;
                }
            }
        }

        free(M1);
    }

    // Cleanup
    for (int i = 0; i < NUM_MATRICES; i++) {
        free(matrices[i]);
    }
    fclose(file);

    return 0;
}
