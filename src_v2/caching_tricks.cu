#include "caching_tricks.hpp"
#include <iostream>
//too confusing dont include #include "matrix_mul.hpp"//order matters this checks for the other header

static __global__ void sharedMatrixMulKernal(matrix a, matrix b, matrix ans, size_t sharedMemorySize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ float sharedA[];

    // Calculate the part of the matrix each thread is responsible for
    int part = (a.cols + blockDim.x - 1) / blockDim.x; 
    int start = part * threadIdx.x;
    int end = min(start + part, a.cols);

    // Load elements into shared memory
    if(threadIdx.y==0){
        for (int i = start; i < end; i++) {
            for (int j = 0; j < a.rows; j++) {
                int sharedIndex = (i - start) * a.rows + j;
                int globalIndex = i * a.rows + j;
                if (sharedIndex < blockDim.x * a.cols && globalIndex < a.cols * a.rows) {
                    sharedA[sharedIndex] = a.data[globalIndex];
                }
            }
        }
    }
    __syncthreads();

    // Compute and write back results
    if (row < ans.rows && col < ans.cols) {
        for (int k = 0; k < a.cols; k++) {
            ans.data[row * ans.cols + col] += sharedA[threadIdx.x * a.cols + k] * b.data[k * b.cols + col];
        }
    }
}

matrix sharedMatrixMul(matrix a,matrix b){
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
    size_t sharedMemSize = x*a.cols* sizeof(float);

    // std::cout << "threads (" << numThreads.x << ", " << numThreads.y << ", " << numThreads.z << ")"
    //       << " blocks (" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << ")"
    //       << " mem size " << sharedMemSize << std::endl;

    //ugly fucking hack
    if (sharedMemSize > 48 * 1024) {
        cudaFuncSetAttribute(sharedMatrixMulKernal, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    }
    sharedMatrixMulKernal<<<numBlocks,numThreads,sharedMemSize>>>(a,b,ans,sharedMemSize); //yes I know I need threads getting to it

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