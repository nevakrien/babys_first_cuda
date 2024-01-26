#include "matrix_mul.h"
#include <omp.h>
#include <stdlib.h>

#define CAP_SIZE 100

matrix matrixMulOMP(matrix a, matrix b){
    if(a.cols != b.rows){
        return (matrix){NULL, -1, -1};
    }

    matrix ans;
    ans.rows = a.rows;
    ans.cols = b.cols;
    ans.data = (float *)calloc(ans.rows * ans.cols, sizeof(float));
    if(!ans.data){
        return ans;
    }

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < ans.rows; i++){
        for(int j = 0; j < ans.cols; j++){
            
            for(int k = 0; k < a.cols; k++){
            	#pragma omp critical
            	ans.data[i*ans.cols+j]+=a.data[i*a.cols+k]*b.data[k*b.cols+j];
            }
            
  
        }
    }

    return ans;
}

matrix matrixMulTransposed(matrix a, transposed_matrix b) {
    // Corrected check: Ensure a.cols (n) matches b.cols (n), 
    // as b.cols is the number of columns of the original B (which should match a.cols)
    if (a.cols != b.cols) {
        printf("Shape error: a.cols=%d, b.cols=%d, a.rows=%d, b.rows=%d\n", a.cols, b.cols, a.rows, b.rows);
        return (matrix){NULL, -1, -1};
    }

    matrix ans;
    ans.rows = a.rows;   // m
    ans.cols = b.rows;   // p, since b is a transposed matrix (original B was n x p)
    ans.data = (float *)calloc(ans.rows * ans.cols, sizeof(float));
    if (!ans.data) {
        printf("Malloc error\n");
        return ans;
    }

    // Matrix multiplication
    for (int i = 0; i < ans.rows; i++) {     // Iterate over rows of 'a' (m)
        for (int j = 0; j < ans.cols; j++) { // Iterate over cols of 'b' (p)
            for (int k = 0; k < a.cols; k++) { // Iterate over cols of 'a' and rows of 'b' (n)
                ans.data[i * ans.cols + j] += a.data[i * a.cols + k] * b.data[j * b.cols + k];
            }
        }
    }

    return ans;
}

matrix matrixMulTransposedOMP(matrix a, transposed_matrix b) {
    // Corrected check: Ensure a.cols (n) matches b.cols (n), 
    // as b.cols is the number of columns of the original B (which should match a.cols)
    if (a.cols != b.cols) {
        printf("Shape error: a.cols=%d, b.cols=%d, a.rows=%d, b.rows=%d\n", a.cols, b.cols, a.rows, b.rows);
        return (matrix){NULL, -1, -1};
    }

    matrix ans;
    ans.rows = a.rows;   // m
    ans.cols = b.rows;   // p, since b is a transposed matrix (original B was n x p)
    ans.data = (float *)calloc(ans.rows * ans.cols, sizeof(float));
    if (!ans.data) {
        printf("Malloc error\n");
        return ans;
    }

    // Matrix multiplication
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ans.rows; i++) {     // Iterate over rows of 'a' (m)
        for (int j = 0; j < ans.cols; j++) { // Iterate over cols of 'b' (p)
            for (int k = 0; k < a.cols; k++) { // Iterate over cols of 'a' and rows of 'b' (n)
                #pragma omp critical
                ans.data[i * ans.cols + j] += a.data[i * a.cols + k] * b.data[j * b.cols + k];
            }
        }
    }

    return ans;
}


int main(){
	int i=0;
	matrix a,ans,y;
	transposed_matrix b;

	FILE *file_a = fopen("matrices_a.bin", "rb");
    FILE *file_b = fopen("matrices_b.bin", "rb");
    FILE *file_ans = fopen("matrices_ans.bin", "rb");

	while(i<CAP_SIZE){
		i+=1;
		printf("\rMultiplying matrix pair %d", i );

		if(readMatrix(file_a,&a)){
			printf("read error\n");
			break;
		}

		if(readTransposedMatrix(file_b,&b)){
			printf("read error\n");
			break;
		}

		if(readMatrix(file_ans,&ans)){
			printf("read error\n");
			break;
		}

		y=matrixMulTransposed(a,b);
		if(!compareMatrices(y,ans)){
			printf("Wrong Data\n");
			return 1;
		}
	}

	printf("\n");

	return 0;
}