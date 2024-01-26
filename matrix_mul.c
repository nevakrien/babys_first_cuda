#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_mul.h"

// typedef struct matrix {
//     float *data;
//     int rows;
//     int cols; 
// };

int readMatrix(FILE *file, matrix *out) {
    if (fread(&(out->rows), sizeof(int), 1, file) != 1) {
        return 1; 
    }
    if (fread(&(out->cols), sizeof(int), 1, file) != 1) {
        return 2;
    }
    int size = (out->rows) * (out->cols);
    out->data = (float *)malloc(size * sizeof(float));
    if (out->data == NULL) {
        return 3;
    } // Added closing bracket here

    if (fread(out->data, sizeof(float), size, file) != size) {
        free(out->data);
        return 4;
    }
    
    return 0; 
}


int writeMatrix(FILE *file,  matrix *in) {
    if (fwrite(&(in->rows), sizeof(int), 1, file) != 1) {
        return 1; 
    }
    if (fwrite(&(in->cols), sizeof(int), 1, file) != 1) {
        return 2;
    }

    if (fwrite(in->data, sizeof(float), in->rows * in->cols, file) != (size_t)(in->rows * in->cols)) {
        return 3; 
    }

    return 0;
}

// Writes a matrix to a file, transposing it.
int writeTransposedMatrix(FILE *file, transposed_matrix *in) {
    if (fwrite(&(in->cols), sizeof(int), 1, file) != 1) { // Swapping for file
        return 1; 
    }
    if (fwrite(&(in->rows), sizeof(int), 1, file) != 1) {
        return 2;
    }

    // Write elements in column-major order
    for (int c = 0; c < in->cols; c++) {
	    for (int r = 0; r < in->rows; r++) {
	        float val = in->data[r * in->cols + c];
	        if (fwrite(&val, sizeof(float), 1, file) != 1) {
	            return 3;
	        }
	    }
	}

    return 0;
}

// Reads a matrix from a file, storing it in transposed format.
int readTransposedMatrix(FILE *file, transposed_matrix *out) {
    int rows, cols;
    if (fread(&cols, sizeof(int), 1, file) != 1) { // Swapped for file
        return 1; 
    }
    if (fread(&rows, sizeof(int), 1, file) != 1) {
        return 2;
    }
    out->rows = rows;
    out->cols = cols;
    int size = rows * cols;
    out->data = (float *)malloc(size * sizeof(float));
    if (out->data == NULL) {
        return 3;
    }

    // Read elements and store in column-major order
	for (int c = 0; c < cols; c++) {
	    for (int r = 0; r < rows; r++) {
	        if (fread(&(out->data[r * cols + c]), sizeof(float), 1, file) != 1) {
	            free(out->data);
	            return 4;
	        }
	    }
	}

    return 0; 
}



matrix matrixMul(matrix a,matrix b){
	if(a.cols!=b.rows){
		return (matrix){NULL,-1,-1};
	}

	matrix ans;

	ans.rows=a.rows;
	ans.cols=b.cols;

	ans.data=(float *)calloc(ans.rows*ans.cols,sizeof(float));
	if(!ans.data){
		return ans;
	}

	for(int i=0;i<ans.rows;i++){
		for(int j=0;j<ans.cols;j++){
			for(int k=0; k<a.cols;k++){
				ans.data[i*ans.cols+j]+=a.data[i*a.cols+k]*b.data[k*b.cols+j];
			}
		}
	}

	return ans;

}

bool compareMatrices(matrix m1, matrix m2){
	if(m1.cols!=m2.cols){
		return false;
	}

	if(m1.rows!=m2.rows){
		return false;
	}

	int size=m1.rows*m1.cols;
	
	for(int i=0;i<size;i++){
		if(m1.data[i]!=m2.data[i]){
			return false;
		}
	}

	return true;
}