///@author Sudhanva Kulkarni
/// this file contains code to generat the diagonal RBF kernel and being cholesky
#include <cuda_runtime.h>
#include <stdio>

#define CUDA_CHECK(err)         \
    do {                        \
        cudaError_t e = (err);   \
        if(e != cudaSuccess) {  \
        fprintf(stderr, "CUDA error %s, at %d : %s",            \
        cudaGetErrorString(e), __LINE__, __FILE__);\
            exit(EXIT_FAILURE);}         \
    } while(0)                  \


__global__ void double_array(float* __restrict__ A, int n) {
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    {
        A[i] *= 2.0f;
    }
}



int main() {

    int n = 50;
    float* h_arr = nullptr;
    cudaHostAlloc((void**) &h_arr, n*sizeof(float), cudaHostAllocDefault);

    float* d_arr = nullptr;
    cudaMalloc((void**)&d_arr, n*sizeof(float));

    for(int i = 0; i < n; i++) h_arr[i] = 1.0f;

    cudaMemcpy(d_arr, h_arr, n*sizeof(float), cudaMemcpyHostToDevice);

    int BlockSize = 1024;
    int GridSize = n/1024;

    cudaStream_t stream;
    

}

