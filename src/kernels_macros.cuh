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
#include <cusolverDn.h>
#include <curand.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_conversion.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;


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

__device__ cutlass::float_e4m3_t float_to_fp8(float val) {
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> converter;
    return converter(val);
}


__device__ float fp8_to_float(cutlass::float_e4m3_t val, int8_t x) {
    cutlass::NumericConverter<float, cutlass::float_e4m3_t> converter;
    return pow(2.0f, x)*converter(val);
}

__device__ float fp8_to_float(cutlass::float_e4m3_t val, float x) {
    cutlass::NumericConverter<float, cutlass::float_e4m3_t> converter;
    return x*converter(val);

}

__device__ int closest_power_of_two(float x) {
    int exponent;
    frexpf(x, &exponent);  // Extract exponent from x = m * 2^e
    int lower = 1 << (exponent - 1);  // 2^(e-1)
    int upper = 1 << exponent;        // 2^e
    return (upper - x < x - lower) ? upper : lower;
}

// __device__ cutlass::float_e4m3_t float_to_mxfp8(float x, int pwr) {
//     float scaled_x = x/powf(2.0f, pwr);  // Scale by 2^pwr
//     return cutlass::float_e4m3_t(scaled_x); // Con

// }

//---------------------------------------------------------------------
// CUDA Kernels

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

// Atomic function for floating-point max
__device__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);
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

//find MAX with BlockReduce
// __global__ void blockMaxReduce(const float* __restrict__ d_in, float* d_out, int num_items) {

//     using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     float val = (tid < num_items) ? d_in[tid] : -FLT_MAX;

//     // Perform block-wide max reduction
//     float block_max = BlockReduce(temp_storage).Reduce(val, cub::Max());

//     // Store max from each block in global memory
//     if (threadIdx.x == 0) {
//         d_out[blockIdx.x] = floor(log2(block_max));
//     }
// }

// __global__ void floatToMX8(const float* __restrict__ src, cutlass::float_e4m3_t* __restrict__ dst, int* __restrict__ dst_pow, int rows, int cols, int ld_src, int ld_dst)
// {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;

//     blockMaxReduce(src, ,blockDim.x);
//     for (; row < rows; row += gridDim.x * blockDim.x) {
//         for (; col < cols; col += gridDim.y * blockDim.y) {
//             dst[row + col * ld_dst] = __float2half(src[row + col * ld_src]);
//         }
//     }


// }

//float to half
__global__ void convertFloatToHalf(const float* __restrict__ src, cutlass::half_t* __restrict__ dst, int rows, int cols, int ld_src, int ld_dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            dst[row + col * ld_dst] = __float2half(src[row + col * ld_src]);
        }
    }
}

__global__ void floatToFp8E4M3Kernel(const float* __restrict__ src, uint8_t* __restrict__ dst, int rows, int cols, int ld_src, int ld_dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            float val = src[row + col * ld_src];
            dst[row + col * ld_dst] = float_to_fp8(val);
        }
    }
}

__global__ void roundAndCastFp8(const float* __restrict__ src, const float* __restrict__ max_val, cutlass::float_e4m3_t* __restrict__ dst, int rows, int cols, int ld_src, int ld_dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            float inv_max;
            float scaled_val;
            uint8_t fp8_val;

            // Compute reciprocal of max_val (approximate)
            asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(inv_max) : "f"(*max_val));

            // Multiply instead of dividing
            scaled_val = src[row + col * ld_src] * inv_max;

            // Convert FP32 -> FP8 (E4M3) using PTX

            dst[row + col * ld_dst] = float_to_fp8(scaled_val);
        }
    }
}

//src pow is the MX scaling
template<typename scaling_type>
__global__ void Fp8toFloat(const cutlass::float_e4m3_t* __restrict__ src, const scaling_type* __restrict__ src_pow, const float* __restrict__ dst, int rows, int cols, int ld, int r)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockIdx.x;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            int index = (row) + (col) * ld;
            int scal_index = row/r + (col/r)*ld;
            
            // Store result in output array
            dst[index] = fp8_to_float(src[index], src_pow[scal_index]);
        }
    }

}

__global__ void FloatToMXFP8(const float* __restrict__ src, int8_t* __restrict__ dst_pow, cutlass::float_e4m3_t* __restrict__ dst, int rows, int cols, int ld, int r)
{
    extern __shared__ float sfdata[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // Correct tid computation

    // Ensure within bounds
    if (row >= rows || col >= cols) return;

    // Compute index for column-major layout
    int index = row + col * ld;
    int block_row = (row / r) * r;
    int block_col = (col / r) * r;
    
    int scal_index = (block_row / r) + (block_col / r) * (ld / r); // Scaling factor index

    // Load data into shared memory for reduction
    sfdata[tid] = (row < rows && col < cols) ? fabsf(src[index]) : 0;
    __syncthreads();

    // 1D Max Reduction (Parallel Reduction in Shared Memory)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sfdata[tid] = fmaxf(fabsf(sfdata[tid + s]), fabsf(sfdata[tid]));
        }
        __syncthreads();
    }

    // Store the max value for the r-by-r block
    if (tid == 0) dst_pow[scal_index] = sfdata[0];

    __syncthreads();  

    // divide by max and Convert to FP8
    float scale = sfdata[0];  // Max value for this r-by-r block

    for (int r_offset = 0; r_offset < r; r_offset++) {
        for (int c_offset = 0; c_offset < r; c_offset++) {
            int new_row = block_row + r_offset;
            int new_col = block_col + c_offset;

            if (new_row < rows && new_col < cols) {
                int new_index = new_row + new_col * ld;  // Column-major index
                float inv_max;
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(inv_max) : "f"(scale));
                auto scaled_val = src[new_index] * inv_max;

                // Convert FP32 -> FP8 (E4M3) using PTX
                dst[new_index] = float_to_fp8(scaled_val); 
                
            }
        }
    }

    
}



//need to call cub for 32*32 blocks. Find max of each and then use warp reduce
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}







__global__ void transposeScaleCastFp8(
    const float* __restrict__ src,   // Input FP32 matrix
    const float* __restrict__ max_val, // Max value for scaling
    cutlass::float_e4m3_t* __restrict__ dst, // Output FP8 matrix
    int rows, int cols, int ld_src, int ld_dst) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    for (; col < cols; col += gridDim.y * blockDim.y) {
    float inv_max;
    float scaled_val;

        // Compute reciprocal of max_val using hardware reciprocal approximation
    asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(inv_max) : "f"(*max_val));

    for (; row < rows; row += gridDim.x * blockDim.x) {

            // Multiply instead of dividing (more efficient)
            scaled_val = src[row + col*ld_src] * inv_max;

            // Transpose and convert FP32 -> FP8 (E4M3)
            dst[col + row*ld_dst] = float_to_fp8(scaled_val);
        }
    }
}


__global__ void floatToFp8E5M2Kernel(const float* __restrict__ src, uint8_t* __restrict__ dst, int rows, int cols, int ld_src, int ld_dst) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for (; row < rows; row += gridDim.x * blockDim.x) {
        for (; col < cols; col += gridDim.y * blockDim.y) {
            float val = src[row + col * ld_src];

            dst[row + col * ld_dst] = float_to_fp8(val);
        }
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


///kerneel to update diagonal with O(nb) work 
__global__ void update_diag(const float* __restrict__ diag_A, float* __restrict__ updated_diag, float* __restrict__ L_block, int n, int k, int ld)
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
    float mach_eps, float eps_prime, int n, 
    int* to_ret)  // ✅ Store as int for atomic operations
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < n)  
        {
            if (mach_eps * updated_diag[i] >= eps_prime * diag_A[i]) {
            atomicExch(to_ret, 0);  // ✅ Atomically set to false (0)
            return;
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


__global__ void construct_diag_geom(double* __restrict__ D, const double* cond, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        D[n*i + i] = pow(1.0/(*cond), double(i)/ double(n - 1)); 
    }
}

__global__ void construct_diag_arith(double* __restrict__ D, const double* cond, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        double frac = static_cast<double>(i) / (n - 1);

        // Interpolate from 1.0 at i=0 to 1.0 / (*cond) at i=n-1:
        double val  = 1.0 + frac * (1.0 / (*cond) - 1.0);

        D[i*n + i] = val;
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


