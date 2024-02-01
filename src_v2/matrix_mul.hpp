#ifndef MATRIX_MUL_HPP
#define MATRIX_MUL_HPP

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
//__global__ void regularMatrixMul(matrix a,matrix b,matrix ans);
float distHeuristic(matrix m1, matrix m2);

//include what we are testing on
#if defined(TestFunction) && TestFunction == sharedMatrixMul
#include "caching_tricks.hpp"
#endif

#if defined(TestFunction) && TestFunction == preMatrixMul
#include "shared_pre.hpp"
#endif

#if defined(TestFunction) && TestFunction == noLoopMatrixMul
#include "no_loop.hpp"
#endif

#endif /* MATRIX_MUL_HPP */
