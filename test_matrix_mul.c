#include "matrix_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

matrix createRandomMatrix(int rows, int cols) {
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        m.data[i] = (float)rand() / RAND_MAX; // Random float between 0 and 1
    }
    return m;
}

void deleteMatrix(matrix m) {
    free(m.data);
}


void testRead(int rows, int cols) {
    printf("Testing ReadVsOriginal with %dx%d matrix...\n", rows, cols);
    matrix original = createRandomMatrix(rows, cols);
    matrix ans;

    FILE *file = fopen("temp_matrix.bin", "w");
    writeMatrix(file, &original);
    fclose(file);

    file = fopen("temp_matrix.bin", "r");
    readMatrix(file, &ans);
    fclose(file);

    if (compareMatrices(ans, original)) {
        printf("regular read passed\n");
    } else {
        printf("regular read failed\n");
    }

    deleteMatrix(original);
    free(ans.data);
    remove("temp_matrix.bin");
}


void testReadVsOriginal(int rows, int cols) {
    printf("Testing ReadVsOriginal with %dx%d matrix...\n", rows, cols);
    matrix original = createRandomMatrix(rows, cols);
    transposed_matrix transposed;

    FILE *file = fopen("temp_matrix.bin", "w");
    writeMatrix(file, &original);
    fclose(file);

    file = fopen("temp_matrix.bin", "r");
    readTransposedMatrix(file, &transposed);
    fclose(file);

    if (compareMatrices((matrix) {.data = transposed.data, .rows = transposed.rows, .cols = transposed.cols}, original)) {
        printf("Test ReadVsOriginal failed: Transposed read should not equal original\n");
    } else {
        printf("Test ReadVsOriginal passed\n");
    }

    deleteMatrix(original);
    free(transposed.data);
    remove("temp_matrix.bin");
}

void testWriteVsOriginal(int rows, int cols) {
    printf("Testing WriteVsOriginal with %dx%d matrix...\n", rows, cols);
    matrix original = createRandomMatrix(rows, cols);
    any_matrix transposed;
    transposed.regular = original;

    FILE *file = fopen("temp_matrix.bin", "w");
    writeTransposedMatrix(file, &transposed.transposed);
    fclose(file);

    file = fopen("temp_matrix.bin", "r");
    matrix readBack;
    readMatrix(file, &readBack);
    fclose(file);

    if (compareMatrices(readBack, original)) {
        printf("Test WriteVsOriginal failed: Transposed write should not equal original\n");
    } else {
        printf("Test WriteVsOriginal passed\n");
    }

    deleteMatrix(original);
    deleteMatrix(readBack);
    remove("temp_matrix.bin");
}

matrix copyMatrix(matrix m) {
    matrix copy;
    copy.rows = m.rows;
    copy.cols = m.cols;
    int size = m.rows * m.cols;
    copy.data = malloc(size * sizeof(float));
    if (copy.data != NULL) {
        for (int i = 0; i < size; i++) {
            copy.data[i] = m.data[i];
        }
    }
    return copy;
}

void testReadVsWriteTransposed(int rows, int cols) {
    printf("Testing ReadVsWriteTransposed with %dx%d matrix...\n", rows, cols);
    matrix original = createRandomMatrix(rows, cols);
    any_matrix transposedWrite, transposedRead;

    // Create a separate copy for transposedWrite
    transposedWrite.regular = copyMatrix(original);

    FILE *file = fopen("temp_matrix.bin", "w");
    writeTransposedMatrix(file, &transposedWrite.transposed);
    fclose(file);

    file = fopen("temp_matrix.bin", "r");
    readTransposedMatrix(file, &transposedRead.transposed);
    fclose(file);

    if (!compareMatrices((matrix) {.data = transposedWrite.transposed.data, .rows = transposedWrite.transposed.rows, .cols = transposedWrite.transposed.cols}, 
                         (matrix) {.data = transposedRead.transposed.data, .rows = transposedRead.transposed.rows, .cols = transposedRead.transposed.cols})) {
        printf("Test ReadVsWriteTransposed failed: Read transposed should equal written transposed\n");
    } else {
        printf("Test ReadVsWriteTransposed passed\n");
    }

    deleteMatrix(original);
    deleteMatrix(transposedWrite.regular);
    free(transposedRead.transposed.data);
    remove("temp_matrix.bin");
}



int main() {
    srand(time(NULL)); 

    testRead(100,103);

    // Test with 5x5 matrix
    testReadVsOriginal(5, 5);
    testWriteVsOriginal(5, 5);
    testReadVsWriteTransposed(5, 5);

    // Test with 5x7 matrix
    testReadVsOriginal(5, 7);
    testWriteVsOriginal(5, 7);
    testReadVsWriteTransposed(5, 7);

    return 0;
}
