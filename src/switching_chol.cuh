
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



void switching_chol(float* d_A, int n, int b,
                    float* d_diagVals, float* updated_diag, float eps_prime, bool perturb_diag, int microscal_size,
                    cusolverDnHandle_t cusolverH, cublasLtHandle_t cublasltH, cublasHandle_t cublasH,
                    cudaStream_t stream, uint8_t* scales) 
{

    float one = 1.0f;
    float neg_One = -1.0f;

    int devInfo_h = 0;
    int* devInfo  = nullptr;
    CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );
    void* d_workspace = nullptr;
    size_t workspaceSize = 16 * 1024 * 1024; // 16 MB
    cudaMalloc(&d_workspace, workspaceSize);

    __nv_fp8_e8m0* d_scales = reinterpret_cast<__nv_fp8_e8m0*>(scales);

    //allocate n*b buffer for schur update
    void* d_schur = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_schur, sizeof(float) * n * b));

    void* d_schur_8 = nullptr;
    void* d_schur_16 = nullptr;
    void* d_schur_32 = nullptr;

    d_schur_8 = reinterpret_cast<__nv_fp8_e4m3*>(d_schur);
    d_schur_16 = reinterpret_cast<__half*>(d_schur);
    d_schur_32 = reinterpret_cast<float*>(d_schur);

    

    // float* A_00 = nullptr;
    // cudaHostAlloc((void**)&A_00, sizeof(float) * r * r, cudaHostAllocDefault);
    int vecThreads = 256;
    int vecBlocks = (n * n + vecThreads - 1) / vecThreads;


    
    int num_blocks = (n + b - 1) / b;
    int lwork = 0;
        CUSOLVER_CHECK(
            cusolverDnSpotrf_bufferSize(
                cusolverH,
                CUBLAS_FILL_MODE_LOWER, // We'll store the factor in the "upper" part for row-major
                b,                      // max block size
                d_A,                    // just pass a valid device pointer
                n,                      //ld
                &lwork                   //pointer to int for size of buffer
            )
        );

    // Allocate workspace for Cholesky factorization

    float* panel_workspace = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&panel_workspace, sizeof(float)*lwork));

    for(int i = 0; i < num_blocks; ++i) {
        int start_row = i * b;
        int end_row = min(start_row + b, n);
        int block_size = end_row - start_row;

        // (A) Copy the submatrix to device
        float* d_A_sub = d_A + start_row * n + start_row;


        // (C) Perform Cholesky factorization on the submatrix
        CUSOLVER_CHECK(cusolverDnSpotrf(
            cusolverH,
            CUBLAS_FILL_MODE_LOWER,
            block_size,         // Size of submatrix (n x n)
            d_A_sub,            // Pointer to the submatrix
            n,                 // Leading dimension of d_A_sub
            panel_workspace,        // Workspace (must be allocated)
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
            n - start_row, block_size, // Size of the submatrix
            &one,                   // Alpha
            d_A_sub,                // Pointer to the submatrix (L)
            n,                     // Leading dimension of d_A_sub
            d_A_sub + block_size,                // Pointer to the submatrix (A)
            n                      // Leading dimension of d_A_sub
        ));

        //update the diagonal values
        update_diag<<<(n - start_row)/256, 256, 0, stream>>>(
            updated_diag + start_row, d_A_sub + block_size, n - start_row, block_size, n 
        );      //looks correct!

        //now decide which GEMM to use based on can_switch
        float prec_3[3] = {MACH_EPS_FP32, MACH_EPS_FP16 ,MACH_EPS_E4M3};
        int * to_ret = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&to_ret, sizeof(int)));
        can_switch<<<(n - start_row)/256, 256, 0, stream>>>(
            d_diagVals + start_row, updated_diag + start_row, prec_3 , 3, eps_prime, n - start_row, to_ret);    //fixed
        
        if (to_ret[0] == 0) {
            // Use FP8 GEMM -- for this first round to fp8
            convert_fp32_to_mxe4m3<<<(n*n + 255) / 256, 256, 0, stream>>>(
                d_A_sub + b, // Input matrix in FP32
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur), // Output matrix in FP8
                n - i*b,       // rows
                b,       // cols
                (d_scales), // scale factors
                n   //ld
            );

            convert_transpose_fp32_to_mxe4m3<<<(n*n + 255) / 256, 256,0 , stream>>>(
                d_A_sub + b, // Input matrix in FP32
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur + n*b*sizeof(__half)), // Output matrix in FP8
                n - i*b,       // rows
                b,       // cols
                (d_scales + n*b*sizeof(__nv_fp8_e4m3)/32), // scale factors
                n           //ld
            );


            LtMxfp8Matmul(
                cublasltH,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n - i*b, // m
                n - i*b, // n
                b,       // k
                &neg_One,    // alpha
                d_scales, // A scale factors
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur), //A
                n,       // lda
                reinterpret_cast<__nv_fp8_e8m0*>(d_scales + n*b*sizeof(__nv_fp8_e4m3)/32), // B scale factors
                reinterpret_cast<__nv_fp8_e4m3*>(d_schur + n*b*sizeof(__half)), // B
                n,       // ldb
                &one, // beta
                d_A + n*i*b + i*b, // C matrix
                n,       // ldc
                d_A + n*i*b + i*b, // D matrix
                n,       // ldd
                d_workspace,            // workspace
                workspaceSize,          // workspace size
                CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, // A scale mode
                CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 // B scale mode
            );

        } else if (to_ret[0] == 1) {
            // Use FP16 GEMM --  round to FP16 first
            convertFloatToHalf<<<(n*n + 255) / 256, 256, 0, stream>>>(
                d_A_sub, // Input matrix in FP32
                reinterpret_cast<__half*>(d_schur), // Output matrix in FP16
                n - i*b,       // rows
                b,        // cols
                n,          //ld_src
                n           //ld_dst       
            );
            LtHSgemm_fp16in_fp32out(
                cublasltH,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n - i*b,
                n - i*b,
                b,
                &neg_One,    // alpha
                reinterpret_cast<__half*>(d_schur), // A matrix in FP16
                n,       // lda
                reinterpret_cast<__half*>(d_schur + n*b*sizeof(__half)), // B matrix in FP16
                n,       // ldb
                &one,    // beta
                d_A + n*i*b + i*b, // C matrix in FP32
                n,       // ldc
                d_workspace, // workspace
                workspaceSize // workspace size
            );


        } else {
            //use FP32 GEMM
            LtSgemm(
                cublasltH,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n - i*b, // m
                n - i*b, // n
                b,       // k
                &neg_One,    // alpha
                d_A_sub + b, // A matrix in FP32
                n,       // lda
                d_A_sub + b, // B matrix in FP32
                n,       // ldb
                &one,    // beta
                d_A + n*i*b + i*b, // C matrix in FP32
                n,       // ldc
                d_workspace, // workspace
                workspaceSize // workspace size
            );

        }
        
        
    }


}