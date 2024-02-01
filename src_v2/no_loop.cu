#include "no_loop.hpp"

static __global__ void noLoopMatrixMulKernal(matrix a,matrix b,matrix ans) {
    
    int row = blockIdx.x;
    int col = blockIdx.y;
    __shared__ float sharedSum[1];

    //int id=threadIdx.x-1+32*(threadIdx.y-1);
    int id=threadIdx.x+32*threadIdx.y;//+32*32*threadIdx.z; //purposfuly overkilling
    
    if (id== 0) {
        sharedSum[0] = 0.0f;
    }

    __syncthreads();

    //if(0<=id<a.cols){
        atomicAdd(sharedSum,a.data[row * a.cols + id] * b.data[id * b.cols + col]);
    //}
    
    __syncthreads();
    if (id == 0) {
        ans.data[row * ans.cols + col]=sharedSum[0];
    }
    
}

matrix noLoopMatrixMul(matrix a,matrix b){
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

    if(a.cols>=1024){
        printf("\nthread overflow avoiding crash\n");
        return {NULL,0,0};
    }
    //dim3 numThreads(x,y,1);
    //dim3 numThreads(1+a.cols%32,1+a.cols/32,1);
    dim3 numThreads(a.cols%32,a.cols/32,1);
    dim3 numBlocks(ans.rows,ans.cols,1);

    printf("\ncols %d\n",a.cols);
    printf("Kernel launch parameters:\n");
    printf("Block dimensions: %d x %d x %d\n", numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Thread dimensions: %d x %d x %d\n", numThreads.x, numThreads.y, numThreads.z);
    noLoopMatrixMulKernal<<<numBlocks,numThreads>>>(a,b,ans); 

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

// static int highestBitPosition(unsigned int n) {
//     int pos = 0;
//     while (n >>= 1) pos++;
//     return pos;
// }