extern "C" {
#include "matrix_mul.h"
}
#include <cuda_runtime.h>
#include <stdlib.h>

#define CAP_SIZE 100

/* 
I set for 3 hours+ on this code because I forgot floats are weird
for the love of god remember to check for margin and not perfection

previous dispair:

this code fails and I  have no idea why
I already mostly eliminated the possibilty its wrong test data. 
since all the other implementations agree on the result
I then started being very paranoid about syncing. then I ran a sanitizer
and also did other checks... nothing.
*/
__global__ void regularMatrixMul(matrix a,matrix b,matrix ans) {
    int row = blockIdx.x;//over simplefied on purpose //* blockDim.y + threadIdx.y;
    int col = blockIdx.y;//over simplefied on purpose //* blockDim.x + threadIdx.x;

    if (row < ans.rows && col < ans.cols) {
        for (int k = 0; k < a.cols; k++) {
            ans.data[row * ans.cols + col] += a.data[row * a.cols + k] * b.data[k * b.cols + col];
        }
    }

}

matrix matrixMulCuda(matrix a,matrix b){
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

    //run
    cudaDeviceSynchronize(); //yet another needless sync in a desprate attempt to fix this code
    dim3 numBlocks(ans.rows,ans.cols,1);
    regularMatrixMul<<<numBlocks,1>>>(a,b,ans); //yes I know I need threads getting to it

    //collecting
    float * host_ans =(float *)malloc(ans_size * sizeof(float));
    if (!host_ans){
        return {NULL,0,0};
    }

    
    cudaDeviceSynchronize(); //first paranoia driven needles sync
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

int main(){
    int i=0;
    matrix a,b,ans,y;

    FILE *file_a = fopen("matrices_a.bin", "rb");
    FILE *file_b = fopen("matrices_b.bin", "rb");
    FILE *file_ans = fopen("matrices_ans.bin", "rb");

    while(i<CAP_SIZE){
        i+=1;
        printf("\rMultiplying matrix pair %d", i );

        if(readMatrix(file_a,&a)){
            printf("\nread error\n");
            break;
        }

        if(readMatrix(file_b,&b)){
            printf("\nread error\n");
            break;
        }

        if(readMatrix(file_ans,&ans)){
            printf("\nread error\n");
            break;
        }

        y=matrixMulCuda(a,b);
        if(distHeuristic(y,ans)<0.05){
            printf("\nWrong Data\n");
            cudaError_t err=cudaGetLastError();
            if (err != cudaSuccess) {
                // Handle the error, for example, print the CUDA error string
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
            return 1;
        }
    }

    printf("\n");

    return 0;
}