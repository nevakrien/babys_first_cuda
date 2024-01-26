#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_mul.hpp"
#include <cuda_runtime.h>
#include <math.h>

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



static __global__ void regularMatrixMul(matrix a,matrix b,matrix ans) {
    int row = blockIdx.x* blockDim.x + threadIdx.x;
    int col = blockIdx.y* blockDim.y + threadIdx.y;

    if (row < ans.rows && col < ans.cols) {
        for (int k = 0; k < a.cols; k++) {
            ans.data[row * ans.cols + col] += a.data[row * a.cols + k] * b.data[k * b.cols + col];
        }
    }

}

matrix matrixMul(matrix a,matrix b){
    //memory
    int a_size=a.rows*a.cols;
    float * device_a;
    cudaMalloc((void**)&device_a, a_size * sizeof(float));
    cudaMemcpy(device_a, a.data, a_size* sizeof(float), cudaMemcpyHostToDevice);


    int b_size=b.rows*b.cols;
    float * device_b;
    cudaMalloc((void**)&device_b, b_size * sizeof(float));
    cudaMemcpy(device_b, b.data, b_size* sizeof(float), cudaMemcpyHostToDevice);

    int ans_size=a.rows*b.cols;
    float * device_ans;
    cudaMalloc((void**)&device_ans, ans_size * sizeof(float));
    cudaMemset(device_ans, 0, ans_size * sizeof(float));

    //setup 
    a.data=device_a;
    b.data=device_b;
    matrix ans=(matrix){device_ans,a.rows,b.cols};
    
    constexpr int x=16;
    constexpr int y=16;

    dim3 numThreads(x,y,1);
    dim3 numBlocks((ans.rows+x-1)/x,(ans.cols+x-1)/x,1);
    regularMatrixMul<<<numBlocks,numThreads>>>(a,b,ans); //yes I know I need threads getting to it

    //collecting
    float * host_ans =(float *)malloc(ans_size * sizeof(float));
    if (!host_ans){
        return {NULL,0,0};
    }

    cudaError_t err = cudaMemcpy(host_ans, device_ans, ans_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // Handle the error, for example, print the CUDA error string
        printf("\nCUDA Error: %s\n", cudaGetErrorString(err));
    }
    ans.data=host_ans;

    //freeing
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_ans);

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

static double computeNorm(const matrix m) {
    double norm = 0.0;
    int size = m.rows * m.cols;
    for (int i = 0; i < size; i++) {
        norm += m.data[i] * m.data[i];
    }
    return sqrt(norm);
}

float distHeuristic(matrix m1, matrix m2) {
    if (m1.cols != m2.cols || m1.rows != m2.rows) {
        return INFINITY; // Indicate dimension mismatch
    }

    int size = m1.rows * m1.cols;
    double diffNorm = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = m1.data[i] - m2.data[i];
        diffNorm += diff * diff;
    }

    diffNorm = sqrt(diffNorm);
    double norm1 = computeNorm(m1);
    double norm2 = computeNorm(m2);

    //if (norm1 == 0 && norm2 == 0) return 0; // Both matrices are zero matrices

    double sumOfNorms = norm1 + norm2;
    if (sumOfNorms == 0) return 0; // only happens when both are the zero matriz

    return diffNorm / sumOfNorms;
}