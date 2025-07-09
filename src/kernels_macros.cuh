#ifndef KERNELS_MACROS_CUH
#define KERNELS_MACROS_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>




namespace cg = cooperative_groups;

enum class DistType {
    RandomLogUniform,   // Type 1
    Clustered,          // Type 2
    Arithmetic,         // Type 3
    Geometric           // Type 4
};


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
            fprintf(stderr, "cuBLAS error %s:%d : %d\n",                 \
                    __FILE__, __LINE__, s);                            \
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



// Convert a double array to a float array.
__global__ void convertDoubleToFloat(const double* __restrict__ d_in,
                                     float* __restrict__ s_out,
                                     int totalElems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 for(; idx < totalElems; idx += blockDim.x * gridDim.x){
        s_out[idx] = static_cast<float>(d_in[idx]);
    }
}



template< typename T>
__global__ void fixDiag(T* d_A, int n, float eps)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        atomicMaxFloat(&d_A[i*n + i], abs(d_A[i*n + i]));
        atomicMaxFloat(&d_A[i*n + i], eps);

    }
}


//Convert Single to double
__global__ void convertFloattoDouble(const float* __restrict__ s_in, 
                                        double* __restrict__ d_out,
                                            int total_elems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(; idx < total_elems; idx += blockDim.x * gridDim.x){
        d_out[idx] = static_cast<double>(s_in[idx]);
    }

}

//float to half
__global__ void convertFloatToHalf(const float* __restrict__ src, __half* __restrict__ dst, int rows, int cols, int ld_src, int ld_dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            dst[row + col * ld_dst] = __float2half(src[row + col * ld_src]);
        }
    }
}












__global__ void Equilibrate_s1(const float* __restrict__ d_in, 
                                   int* __restrict__ D, int n, int ld)
{
    //two dimensional kernel launch
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < n; tid += blockDim.x * gridDim.x) {
        int offset = tid + tid * ld;
        int exp = 0;
        frexp(d_in[offset], &exp);
        D[tid] = exp + 127;  // Store the exponent adjusted for E8M0
    }

    return;

}

__global__ void Equilibrate_s2(const float* __restrict__ d_in,
                                   int* __restrict__ D, int n, int ld)                                  
{
    //use elems in D to scale the matrix

}

// Extract the diagonal (for a matrix in column-major order).
__global__ void extractDiagonal(const float* __restrict__ d_in,
                                float* __restrict__ D, int n, int ld, bool po2 = false)
{
    // Use a grid-stride loop over diagonal indices.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        int offset = i + i * ld;
        int exp = 0;
        D[i] = float(d_in[offset]);  
    }
}


///kerneel to update diagonal with O(nb) work 
__global__ void update_diag(float* __restrict__ updated_diag, float* __restrict__ L_block, int n, int k, int ld)
{
    
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {

        for(int j = 0; j < k; j++)
        {
            updated_diag[i] -= L_block[ld*j + i]*L_block[ld*j + i];
        }

    }

}
__global__ void can_switch(const float* __restrict__ diag_A, 
    float* __restrict__ updated_diag, 
    float* __restrict__ mach_eps, int num_prec, float eps_prime, int n, 
    int* to_ret)  // ✅ Store as int for atomic operations
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int count = 0;
        

        while (i < n)  
        {
        while(count < num_prec) {
            if (mach_eps[count] * updated_diag[i] >= eps_prime * diag_A[i]) {
            atomicExch(to_ret, count);  // ✅ Atomically set to false (0)
            return;
            }
            count++;
        }
        i += gridDim.x * blockDim.x;  
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

/// Reduction kernel to compute the infinity norm of a vector. Need to reduce the resultrs of blockMax......
__global__ void infNormVecKernel(const double* __restrict__ x, double* blockMax, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    *blockMax = 0.0;
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

//atomicCAS for max(double, double)
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);

        // Ensure we don't overwrite with a smaller value
        if (old_val >= val) return old_val;  

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (old != assumed);

    return __longlong_as_double(old);
}

__global__ void vanilla_Max(const double* __restrict__ vec, double* blockMax, int n)        //default args are only used for submatrix max norm
{
    extern __shared__ double sdata[];
    int local_tid  = threadIdx.x;                             // Local index within the block
    int global_tid = blockIdx.x * blockDim.x + local_tid;      // Global index
    double t_val = 0.0;

    for (int i = global_tid; i < n; i += blockDim.x * gridDim.x)
    {
        t_val = fmax(t_val, fabs(vec[i]));
    }
    
    // Store the thread's result in shared memory.
    sdata[local_tid] = t_val;
    __syncthreads();

    // Intra-block reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] = fmax(sdata[local_tid], sdata[local_tid + s]);
        }
        __syncthreads();
    }

    // Write the block's maximum to the output array.
    if (local_tid == 0) {
        blockMax[blockIdx.x] = sdata[0];
    }
}

__global__ void infNormKernel(const double* __restrict__ A, double* rowSums, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += fabsf(A[col * rows + row]);  // Column-major access
    }
    rowSums[row] = sum;
}

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void maxReduceKernel(const double* __restrict__ rowSums, double* result, int n) {
    extern __shared__ double smaxdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    smaxdata[tid] = (i < n) ? rowSums[i] : -DBL_MAX;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smaxdata[tid] = fmax(smaxdata[tid], smaxdata[tid + s]);
        __syncthreads();
    }

    // Write the block's local max to global result
    if (tid == 0)
        atomicMaxDouble(result, smaxdata[0]);
}



__global__ void vanilla_Max(const float* __restrict__ vec, float* blockMax, int n)
{
    extern __shared__ float sfdata[];
    int local_tid  = threadIdx.x;                             // Local index within the block
    int global_tid = blockIdx.x * blockDim.x + local_tid;      // Global index
    float t_val = 0.0;

    for (int i = global_tid; i < n; i += blockDim.x * gridDim.x)
    {
        t_val = fmaxf(t_val, fabs(vec[i]));
    }
    
    // Store the thread's result in shared memory.
    sfdata[local_tid] = t_val;
    __syncthreads();

    // Intra-block reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sfdata[local_tid] = fmaxf(sfdata[local_tid], sfdata[local_tid + s]);
        }
        __syncthreads();
    }

    // Write the block's maximum to the output array.
    if (local_tid == 0) {
        blockMax[blockIdx.x] = sfdata[0];
    }
}

//the below func assumes matrix is stored in column major so we will transpose the 2D threads to make suree that we can still use memory coalescing and all that
__global__ void vanilla_Max_2D_col_major(const float* __restrict__ mat, float* blockMax, 
                                         int rows, int cols, int ld) {
    extern __shared__ float sfdata[];

    // Compute thread's unique (x, y) coordinates with transposed indexing
    int ty = threadIdx.x;  // Swap thread x <--> y
    int tx = threadIdx.y;
    int by = blockIdx.x;   // Swap block x <--> y
    int bx = blockIdx.y;

    int global_y = by * blockDim.x + ty;  // Now iterating over columns
    int global_x = bx * blockDim.y + tx;  // Now iterating over rows

    int local_tid  = ty * blockDim.y + tx; // Flattened thread ID inside block
    float t_val = 0.0;

    // Strided access (column-major optimized)
    for (int x = global_x; x < rows; x += gridDim.y * blockDim.y) { // Iterate over rows first
        for (int y = global_y; y < cols; y += gridDim.x * blockDim.x) { // Iterate over columns next
            int idx = x + y * ld;  // Column-major indexing: (row + col * ld)
            t_val = fmaxf(t_val, fabs(mat[idx]));
        }
    }

    // Store thread's max value in shared memory
    sfdata[local_tid] = t_val;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sfdata[local_tid] = fmaxf(sfdata[local_tid], sfdata[local_tid + s]);
        }
        __syncthreads();
    }

    // Write block's max to global memory
    if (local_tid == 0) {
        int block_index = bx * gridDim.x + by; // Adjust block ID mapping
        blockMax[block_index] = sfdata[0];
    }
}

__global__ void final_max_reduce(const float* __restrict__ blockMax, float* max, int nBlocks)
{
    extern __shared__ float sfdata[];
    int tid = threadIdx.x;

    // Load the blockMax values into shared memory.
    // If the number of threads is greater than nBlocks, initialize extras with -DBL_MAX.
    if (tid < nBlocks)
        sfdata[tid] = blockMax[tid];
    else
        sfdata[tid] = 0.0;
    __syncthreads();

    // Reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sfdata[tid] = fmaxf(sfdata[tid], sfdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the final maximum.
    if (tid == 0)
        *max = sfdata[0];
}



__global__ void final_max_reduce(const double* __restrict__ blockMax, double* max, int nBlocks)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    // Load the blockMax values into shared memory.
    // If the number of threads is greater than nBlocks, initialize extras with -DBL_MAX.
    if (tid < nBlocks)
        sdata[tid] = blockMax[tid];
    else
        sdata[tid] = 0.0;
    __syncthreads();

    // Reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the final maximum.
    if (tid == 0)
        *max = sdata[0];
}




#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ double reduce_max(cg::thread_group g, double *temp, double val)
{
    int lane = g.thread_rank();

    // Store initial value before reduction
    temp[lane] = fabs(val);
    g.sync();

    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        if (lane < i) temp[lane] = fmax(temp[lane], fabs(temp[lane + i]));
        g.sync();
    }
    return temp[0];  // Ensure return from thread 0
}

__device__ double thread_max(double *input, int n) 
{
    double maxim = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n/4; i += stride)
    {
        double4 in = ((double4*)input)[i];
        maxim = fmax(fmax(fmax(fabs(in.x), fabs(in.y)), fabs(in.z)), fabs(in.w));
    }

    // Handle remainder if n is not a multiple of 4
    int rem_start = (n / 4) * 4;
    for (int i = rem_start + idx; i < n; i += stride)
    {
        maxim = fmax(maxim, fabs(input[i]));
    }

    return maxim;
}

__global__ void max_kernel_block(double *maxim, double *input, int n)
{
    double my_max = thread_max(input, n);

    extern __shared__ double temp[];
    auto g = cg::this_thread_block();
    double block_max = reduce_max(g, temp, my_max);

    if (g.thread_rank() == 0) 
        atomicMax((unsigned long long*)maxim, __double_as_longlong(block_max));
}



// Kernel to perform diagonal scaling: new_v[i] = v[i] * D[i].
__global__ void diag_scal(const double* v, double* new_v, const double* D, int N) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x* gridDim.x)
    {
     new_v[i] = v[i]*D[i];
    }

}

// Kernel to permute a vector according to a given index array.
__global__ void permute(const double* orig_vec, const int* ind, double* new_vec, int N) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x* gridDim.x)
 {
        new_vec[i] = orig_vec[ind[i]];
    }
}

//division lmao
__global__ void computeAlphaKernel(double rz, double pAp, double* d_alpha) {
    // Use a single thread to perform the division.
    *d_alpha = rz / pAp;
}

//check for nan
__global__ void hasNan(const float* __restrict__ A, bool* to_ret, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        if(isnan(A[i])) {
            *to_ret = true;
            return;
        }
    }

}




//helper func to copy submatrices-
cudaError_t CopySubmatrixFloat(const float* src, float* dst,
                               int ld,   // full matrix leading dimension (number of rows)
                               int i,    // starting row index for submatrix
                               int j,    // starting column index for submatrix
                               int subH, // number of rows in submatrix
                               int subW, // number of columns in submatrix
                               cudaMemcpyKind kind)
{

    const float* srcSubMatrix = src + j * ld + i;
    size_t srcPitch = ld * sizeof(float);
    size_t dstPitch = subH * sizeof(float);
    size_t widthInBytes = subH * sizeof(float);
    size_t height = subW;

    return cudaMemcpy2D(dst, dstPitch,
                        srcSubMatrix, srcPitch,
                        widthInBytes, height,
                        kind);
}

__global__ void set_identity(float* __restrict__ matr, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) 
    {
        matr[i*n + i] = 1.0f;
    }
}


__global__ void x_r_update(double* __restrict__  x, double* __restrict__  r, const double* rz, const double* pAp, const double* __restrict__ p, const double* __restrict__ Ap, int n)
{
    // x += alpha*p
    // r -= alpha*Ap
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ double alpha[];
    if(threadIdx.x == 0) {
        alpha[0] = (*rz)/(*pAp);
    }
    __syncthreads();

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x* gridDim.x)
    {
        x[i] += alpha[0]*p[i];
        r[i] -= alpha[0]*Ap[i]; 
    }

}

__global__ void update_search_dir(double* __restrict__ p, const double* z, const double* rz_new, double* rz,  const int n)
{
    extern __shared__ double beta[];
    if (threadIdx.x == 0) {
        beta[0] = (*rz_new) / (*rz);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        p[idx] = z[idx] + beta[0] * p[idx];
    }

}

// Kernel: Log-uniform random distribution (Type 1)
__global__ void construct_diag_logrand(double* diag, const double* cond, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        double log_min = log10(1.0 / (*cond));
        double log_max = 0.0;
        double r = curand_uniform_double(&state);
        double log_sigma = log_min + r * (log_max - log_min);
        double sigma = pow(10.0, log_sigma);
        diag[idx * n + idx] = sigma;
    }
}

// Kernel: Clustered distribution (Type 2)
__global__ void construct_diag_clustered(double* diag, const double* cond, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = (idx < n - 1) ? 1.0 : 1.0 / (*cond);
        diag[idx * n + idx] = val;
    }
}

// Kernel: Arithmetic decay (Type 3)
__global__ void construct_diag_arith(double* diag, const double* cond, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double i = static_cast<double>(idx);
        double factor = (1.0 - i / (n - 1.0)) * (1.0 - 1.0 / (*cond));
        diag[idx * n + idx] = 1.0 - factor;
    }
}

// Kernel: Geometric decay (Type 4)
__global__ void construct_diag_geom(double* diag, const double* cond, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = static_cast<double>(idx) / (n - 1.0);
        diag[idx * n + idx] = pow(*cond, -exponent);
    }
}

// Dispatcher function
inline void launch_construct_diag(double* d_diagVals, const double* d_cond, int n, DistType dist, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    unsigned long long seed = 123456789ULL;

    switch (dist) {
        case DistType::RandomLogUniform:
            construct_diag_logrand<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n, seed);
            break;
        case DistType::Clustered:
            construct_diag_clustered<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
            break;
        case DistType::Arithmetic:
            construct_diag_arith<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
            break;
        case DistType::Geometric:
            construct_diag_geom<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
            break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void scaleColumnsByDiag(const double* __restrict__ diagVals, 
                                   double* __restrict__ Q, 
                                   int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;

    while(idx < total) {
        int col = idx / n;  // each column has 'n' entries
        Q[idx] *= diagVals[col];
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void copy_diag(float* __restrict__ diag_A, float* __restrict__ updated_diag, const float* __restrict__ A, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        diag_A[i] = A[n*i + i];
        updated_diag[i] = A[n*i + i];
    }
}


// __global__ void check_switch(const float* __restrict__ diag_A, const float* __restrict__ updated_diag, )
// {

// }

#endif


