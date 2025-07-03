#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>         // ✅ Include for thrust::sequence
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <cutlass/array.h>
#include "kernels_macros.cuh"
#include "fp8_utils.cuh"

#define MACH_EPS_E2M1 0.5f
#define MACH_EPS_E4M3 0.125f
#define MACH_EPS_FP16 0.0009765625f
#define MACH_EPS_FP32 1.1920929e-7f

enum class precisions : uint8_t {
    fp4 = 0,
    fp8 = 1,
    fp16 = 2,
    fp32 = 3
};

std::ofstream gemms_file("gemmers.csv");

void switching_chol(float* d_A, int n, int b
                    float* d_diagVals, float* updated_diag, float eps_prime, bool perturb_diag, int microscal_size,
                    cuSolverDnHandle_t cusolverH, cublasLtHandle_t cublasltH,
                    cudaStream_t stream, uint8_t* scales) 
{

    float one = 1.0f;
    float negOne = -1.0f;

    int devInfo_h = 0;
    int* devInfo  = nullptr;
    CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );

    __nv_fp8_e8m0* d_scales = reinterpret_cast<__nv_fp8_e8m0*>(scales);

    //allocate n*r buffer for schur update
    void* d_schur = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_schur, sizeof(float) * n * b));

    void* d_schur_8 = nullptr;
    void* d_schur_16 = nullptr;
    void* d_schur_32 = nullptr;


    

    // float* A_00 = nullptr;
    // cudaHostAlloc((void**)&A_00, sizeof(float) * r * r, cudaHostAllocDefault);
    int vecThreads = 256;
    int vecBlocks = (n * n + vecThreads - 1) / vecThreads;

    float* updated_diag = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&updated_diag, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(updated_diag, d_diagVals, n * sizeof(float), cudaMemcpyDeviceToDevice));


    float* dev_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_blockMax, vecBlocks * sizeof(float)));

    int num_blocks = (n + b - 1) / b;
    int lwork = 0;
        CUSOLVER_CHECK(
            cusolverDnSpotrf_bufferSize(
                cusolverH,
                CUBLAS_FILL_MODE_LOWER, // We'll store the factor in the "upper" part for row-major
                r,                      // max block size
                d_A_sub,                    // just pass a valid device pointer
                ld,
                &lwork
            )
        );

    float* panel_workspace = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&panel_workspace, sizeof(float)*lwork));

    for(int i = 0; i < num_blocks; ++i) {
        int start_row = i * b;
        int end_row = min(start_row + b, n);
        int block_size = end_row - start_row;

        // (A) Copy the submatrix to device
        float* d_A_sub = d_A + start_row * n + start_row;

        // (B) Convert to half precision if needed
        // ... (conversion code here)
        //need to get the 
        

        // (C) Perform Cholesky factorization on the submatrix
        CUSOLVER_CHECK(cusolverDnSpotrf(
            cusolverH,
            CUBLAS_FILL_MODE_LOWER,
            block_size,         // Size of submatrix (n x n)
            d_A_sub,            // Pointer to the submatrix
            ld,                 // Leading dimension of d_A_sub
            d_Workspace,        // Workspace (must be allocated)
            lwork,
            devInfo             // Info output
        ));

        //now call TRSM
        CUBLAS_CHECK(cublasStrsm(
            cublasH,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            block_size, block_size, // Size of the submatrix
            &one,                   // Alpha
            d_A_sub,                // Pointer to the submatrix (L)
            ld,                     // Leading dimension of d_A_sub
            d_A_sub,                // Pointer to the submatrix (A)
            ld                      // Leading dimension of d_A_sub
        ));

        //update the diagonal values
        update_diag<<<(block_size + 255) / 256, 256, 0, stream>>>(
            updated_diag + start_row, d_A_sub + block_size,n - i*block_size, block_size, n 
        );

        //now decide which GEMM to use based on can_switch
        int * to_ret = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&to_ret, sizeof(int)));
        can_switch<<<(block_size + 255) / 256, 256, 0, stream>>>(
            d_diagVals + start_row, updated_diag + start_row,  {MACH_EPS_FP8, MACH_EPS_FP16, MACH_EPS_FP32}, 3, eps_prime, n - i*b, to_ret);
        
        if (to_ret[0] == 1) {
            // Use FP8 GEMM -- for this first round to fp8

            convert_fp32_to_mxe4m3<<<(n*n + 255) / 256, 256, 0, stream>>>(
                d_A_sub, // Input matrix in FP32
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur), // Output matrix in FP8
                n,       // rows
                r,       // cols
                reinterpret_cast<__nv_fp8_e8m0*>(d_scales) // scale factors
            );

            convert_transpose_fp32_to_mxe4m3<<<(n*n + 255) / 256, 256, , stream>>>(
                d_A_sub, // Input matrix in FP32
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur + n*r*sizeof(__half)), // Output matrix in FP8
                n,       // rows
                r,       // cols
                reinterpret_cast<__nv_fp8_e8m0*>(d_scales + n*r*sizeof(__nv_fp8_e4m3)/32) // scale factors
            );


            LtMxfp8Matmul(
                cublasltH,
                d_A_sub, // A matrix
                n,       // m
                n,       // k
                n,       // lda
                d_A_sub, // B matrix
                n,       // k
                n,       // n
                n,       // ldb
                &one,    // alpha
                d_A_sub, // C matrix
                n,       // ldc
                updated_diag + start_row, // D matrix
                n        // ldd
            );

        } else if (to_ret[0] == 2) {
            // Use FP16 GEMM --  round to FP16 first
            convertFloattoHalf<<<(n*n + 255) / 256, 256, 0, stream>>>(
                d_A_sub, // Input matrix in FP32
                reinterpret_cast<__half*>(d_schur), // Output matrix in FP16
                n,       // rows
                r        // cols
            );
            LtHSgemm_fp16in_fp32out(
                cublasltH,
                reinterpret_cast<__half*>(d_schur), // A matrix in FP16
                n,       // m
                n,       // k
                n,       // lda
                reinterpret_cast<__half*>(d_schur + n*r*sizeof(__half)), // B matrix in FP16
                n,       // k
                n,       // n
                n,       // ldb
                &one,    // alpha
                d_A_sub, // C matrix in FP32
                n,       // ldc
                updated_diag + start_row, // D matrix in FP32
                n        // ldd
            );


        } else {
            //use FP32 GEMM
            LtSgemm(
                cublasltH,
                d_A_sub, // A matrix in FP32
                n,       // m
                n,       // k
                n,       // lda
                d_A_sub, // B matrix in FP32
                n,       // k
                n,       // n
                n,       // ldb
                &one,    // alpha
                d_A_sub, // C matrix in FP32
                n,       // ldc
                updated_diag + start_row, // D matrix in FP32
                n        // ldd
            );

        }
        
        
    }


}