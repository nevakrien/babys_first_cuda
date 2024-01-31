#include "shared_pre.hpp"

static constexpr int step=16;

static __global__ void preMatrixMulKernal(matrix a,matrix b,matrix ans) {
    
    int row = blockIdx.x;
    int col = blockIdx.y;
    float sum=0.0;

    for (int k = threadIdx.z; k < a.cols; k+=step) {
        sum += a.data[row * a.cols + k] * b.data[k * b.cols + col];
    }
    atomicAdd(&ans.data[row * ans.cols + col],sum);
}

matrix preMatrixMul(matrix a,matrix b){
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

    //dim3 numThreads(x,y,1);
    dim3 numThreads(1,1,step);
    dim3 numBlocks(ans.rows,ans.cols,1);
    preMatrixMulKernal<<<numBlocks,numThreads>>>(a,b,ans); 

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