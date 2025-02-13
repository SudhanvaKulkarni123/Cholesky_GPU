#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>


using namespace std;
// Error-checking macros
#define CUDA_CHECK(err)                                             \
    do {                                                            \
        cudaError_t e = (err);                                      \
        if(e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(e));     \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CUBLAS_CHECK(err)                                           \
    do {                                                            \
        cublasStatus_t s = (err);                                   \
        if (s != CUBLAS_STATUS_SUCCESS)  {                          \
            fprintf(stderr, "cuBLAS error %s:%d\n",                 \
                    __FILE__, __LINE__);                            \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0)


#define CUSOLVER_CHECK(err)                                         \
    do {                                                            \
        cusolverStatus_t s = (err);                                \
        if(s != CUSOLVER_STATUS_SUCCESS) {                          \
        fprintf(stderr, "cuSOLVER error %s:%d\n",                   \
        __FILE__, __LINE__);                                        \
        exit(EXIT_FAILURE);                                         \
        }                                                           \
    } while(0)                                                      \


#define CURAND_CHECK(err)                                           \
    do {                                                            \
        curandStatus_t s = (err);                                   \
        if (s != CURAND_STATUS_SUCCESS) {                           \
            fprintf(stderr, "CURAND error %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

//---------------------------------------------------------------------
// Utility function: compute permutation indices from a permutation vector.
void perm_to_ind(vector<int>& perm, vector<int>& ind, int n) {
    ind.resize(n);
    for (int i = 0; i < n; i++) 
        ind[i] = i;
    for (int i = n - 1; i >= 0; i--) {
        swap(ind[perm[i]], ind[i]);
    }
}

//---------------------------------------------------------------------
// CUDA Kernels

// Convert a double array to a float array.
__global__ void convertDoubleToFloat(const double* __restrict__ d_in,
                                     float* __restrict__ s_out,
                                     int totalElems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElems) {
        s_out[idx] = static_cast<float>(d_in[idx]);
    }
}

// Extract the diagonal (for a matrix in column-major order).
__global__ void extractDiagonal(const double* __restrict__ d_in,
                                double* __restrict__ D, int n, int ld)
{
    // Use a grid-stride loop over diagonal indices.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        int offset = i + i * ld;
        D[i] = d_in[offset];
    }
}

// Normalize a matrix by dividing each column by its diagonal element.
// The matrix is assumed to be stored in column-major order.
__global__ void normalizeMatrix(double* __restrict__ d_in,
                                const double* __restrict__ D, int n, int ld)
{
    for (int j = blockIdx.x; j < n; j += gridDim.x) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int offset = i + j * ld;
            d_in[offset] /= D[j];
        }
    }
}

// Reduction kernel to compute the infinity norm of a vector.
__global__ void infNormVecKernel(const double* __restrict__ x, double* blockMax, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    double localMax = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double val = fabs(x[i]);
        localMax = fmax(localMax, val);
    }
    sdata[tid] = localMax;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        blockMax[blockIdx.x] = sdata[0];
    }
}

// Kernel to perform diagonal scaling: new_v[i] = v[i] * D[i].
__global__ void diag_scal(const double* v, double* new_v, const double* D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        new_v[i] = v[i] * D[i];
    }
}

// Kernel to permute a vector according to a given index array.
__global__ void permute(const double* orig_vec, const int* ind, double* new_vec, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        new_vec[i] = orig_vec[ind[i]];
    }
}


