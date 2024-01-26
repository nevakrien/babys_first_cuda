#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <stdio.h>
#include <stdbool.h>

typedef struct matrix {
    float *data;
    int rows;
    int cols; 
} matrix;

//same as matrix just stored transposed
typedef struct transposed_matrix{
    float *data;
    int rows;
    int cols;
} transposed_matrix;


typedef union {
    matrix regular;
    transposed_matrix transposed ;
} any_matrix;

int readMatrix(FILE *file,  matrix *out);
int writeMatrix(FILE *file,  matrix *in);

int readTransposedMatrix(FILE *file,  transposed_matrix *out);
int writeTransposedMatrix(FILE *file,  transposed_matrix *in);
matrix matrixMul(matrix a,matrix b);
bool compareMatrices(matrix m1, matrix m2);

#endif /* MATRIX_IO_H */
