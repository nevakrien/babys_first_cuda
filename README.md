# babys_first_cuda
my first attempt at cuda 
this is a matrix multiplication setup 


test data was made with single threaded c that we can use and we then go and use cuda to do stuff for real.

after I got it working I moved everything to v1 and am now working on optimizing it using my working code


# builds
nvcc src_v2/test_mul.cu matrix_mul.o
nvcc src_v2/test_mul.cu matrix_mul.o caching_tricks.o -DTestFunction=sharedMatrixMul