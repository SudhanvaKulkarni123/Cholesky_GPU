#include "kernels_macros.cuh"
#include "matgen.cuh"
#include "micro_chol.hpp"
#include <fstream>
#include <iostream>


using namespace std;

void host_isnan(float* A, int n)
{
    for(int i = 0; i < n*n; i++) {
        if(isnanf(A[i])){ std::cout << "A has a nan\n"; return;}

    }
}


// Uniform precision Cholesky factorizatison in single precision.
// This function factors the matrix (in-place) stored in d_A.
int uniform_precision_Cholesky(float* d_A, int ld, int r, float* diag_A, float* updated_diag, 
                               cublasHandle_t handle, cudaStream_t stream, int n) {
    float one    = 1.0f;
    float negOne = -1.0f;

    // Additional CUDA Stream for GEMM (SYRK)
    cudaStream_t gemmStream;
    cudaStreamCreate(&gemmStream);

    // CUDA Events for Synchronization
    cudaEvent_t start, stop, panel_done, trsm_done;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&panel_done);
    cudaEventCreate(&trsm_done);

    float time_factorize = 0, time_trsm = 0, time_gemm = 0;

    // Use pinned memory for faster host-device transfer
    float* A_00;
    cudaHostAlloc((void**)&A_00, sizeof(float) * r * r, cudaHostAllocDefault);

    for (int k = 0; k < n; k += r) {
        int r_block = (k + r < n) ? r : (n - k);

        // Start timing Panel Factorization
        cudaEventRecord(start, stream);

        // Asynchronous copy of panel to CPU
        size_t dstPitch = r_block * sizeof(float);
        size_t srcPitch = ld * sizeof(float);
        size_t widthInBytes = r_block * sizeof(float);

        CUDA_CHECK(cudaMemcpy2DAsync(A_00, dstPitch, 
                                     d_A + k + k * ld, srcPitch, 
                                     widthInBytes, r_block, 
                                     cudaMemcpyDeviceToHost, stream));

        // Factorization happens on CPU asynchronously
        cudaStreamSynchronize(stream);
        micro_cholesky(A_00, r_block, r_block);

        // Copy factorized block back asynchronously
        CUDA_CHECK(cudaMemcpy2DAsync(d_A + k + k * ld, srcPitch, 
                                     A_00, dstPitch, 
                                     widthInBytes, r_block, 
                                     cudaMemcpyHostToDevice, stream));

        // Signal that the panel factorization is done
        cudaEventRecord(panel_done, stream);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_factorize += elapsed;

        int sub = n - (k + r_block);
        if (sub > 0) {
            // **TRSM (Right Solve)**
            cudaEventRecord(start, stream);

            CUBLAS_CHECK(
                cublasStrsm(handle,
                            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                            sub, r_block, &one,
                            d_A + k + k * ld, ld,
                            d_A + (k + r_block) + k * ld, ld)
            );

            // Signal that TRSM is done
            cudaEventRecord(trsm_done, stream);

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_trsm += elapsed;

            // **Concurrent GEMM (SYRK) on Separate Stream**
            cudaStreamWaitEvent(gemmStream, trsm_done, 0);  // Ensure TRSM is done before GEMM

            cudaEventRecord(start, gemmStream);

            CUBLAS_CHECK(
                cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            sub, r_block, &negOne,
                            d_A + (k + r_block) + k * ld, ld,  // A_sub
                            &one, d_A + (k + r_block) + (k + r_block) * ld, ld) // A_trailing
            );

            cudaEventRecord(stop, gemmStream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_gemm += elapsed;
        }
    }

    cudaFreeHost(A_00);
    cudaStreamDestroy(gemmStream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(panel_done);
    cudaEventDestroy(trsm_done);

    // Print profiling results
    std::cout << "Profiling Results (ms):\n";
    std::cout << "Factorization (A00): " << time_factorize << " ms\n";
    std::cout << "TRSM: " << time_trsm << " ms\n";
    std::cout << "GEMM: " << time_gemm << " ms\n";

    return 0;
}

// #include <cuda_fp16.h>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <iostream>
int halfprec_mixed_precision_Cholesky(float* d_A, int ld, int r,
                                      float* diag_A, float* updated_diag,
                                      cublasHandle_t handle, cudaStream_t stream,
                                      int n) {
    float one = 1.0f;
    float negOne = -1.0f;

    // Allocate half-precision buffer for A_sub
    __half* d_A_sub_half;
    cudaMalloc((void**)&d_A_sub_half, sizeof(__half) * n * n);

    float* A_00;
    cudaHostAlloc((void**)&A_00, sizeof(float) * r * r, cudaHostAllocDefault);

    // CUDA Events for Profiling
    cudaEvent_t start, stop;
    float time_factorize = 0, time_trsm = 0, time_conversion = 0, time_gemm = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Variables for FLOP counts
    double flops_factorize = 0, flops_trsm = 0, flops_gemm = 0;

    // Create cublasLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // Setup matrix multiplication descriptors
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;

    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)
    );
    cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)
    );


    // Set up workspace
    size_t workspaceSize = 32 * 1024 * 1024; // 32MB workspace
    void* d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);

    // Preference for performance tuning
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)
    );

    for (int k = 0; k < n; k += r) {
        int r_block = (k + r < n) ? r : (n - k);

        // Start timing A00 factorization
        cudaEventRecord(start, stream);

        size_t dstPitch = r_block * sizeof(float);
        size_t srcPitch = ld * sizeof(float);
        size_t widthInBytes = r_block * sizeof(float);

        CUDA_CHECK(cudaMemcpy2D(A_00, dstPitch,
                                     d_A + k + k * ld, srcPitch, 
                                     widthInBytes, r_block, 
                                cudaMemcpyDeviceToHost));

        // CPU Cholesky factorization
        micro_cholesky(A_00, r_block, r_block);

        int sub = n - (k + r_block);

        CUDA_CHECK(cudaMemcpy2D(d_A + k + k * ld, srcPitch,
                                     A_00, dstPitch, 
                                     widthInBytes, r_block, 
                                cudaMemcpyHostToDevice));

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_factorize += elapsed;

        // Calculate FLOPs for Cholesky Factorization
        flops_factorize += (1.0 / 3.0) * r_block * r_block * r_block;

        if (sub > 0) {
            // Start timing TRSM
            cudaEventRecord(start, stream);

            CUBLAS_CHECK(
                cublasStrsm(
                    handle,
                    CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT,
                    sub, r_block, &one,
                    d_A + k + k * ld, ld,
                    d_A + (k + r_block) + k * ld, ld
                )
            );

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_trsm += elapsed;

            // Calculate FLOPs for TRSM
            flops_trsm += r_block * r_block * sub;

            // Start timing conversion
            cudaEventRecord(start, stream);

            dim3 blockSize(16, 16);
            dim3 gridSize((sub + blockSize.x - 1) / blockSize.x, (r_block + blockSize.y - 1) / blockSize.y);
            convertFloatToHalf<<<gridSize, blockSize, 0, stream>>>(
                d_A + (k + r_block) + k * ld,
                d_A_sub_half + (k + r_block) + k * ld,
                sub, r_block, ld, ld
            );

            cudaStreamSynchronize(stream);

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_conversion += elapsed;

            // Start timing GEMM
            cudaEventRecord(start, stream);

            // Setup matrix layouts
            cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, sub, r_block, ld);
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, sub, r_block, ld);
            cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, sub, sub, ld);

            // Get best algorithm heuristic
            cublasLtMatmulHeuristicResult_t heuristicResult;
            int returnedResults = 0;
            cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);

            if (returnedResults == 0) {
                std::cerr << "No suitable algorithm found for cublasLtMatmul!" << std::endl;
                return -1;
            }

            // Perform GEMM using cublasLtMatmul
            CUBLAS_CHECK(
                cublasLtMatmul(
                    ltHandle, matmulDesc, &negOne,
                    d_A_sub_half + (k + r_block) + k * ld, Adesc,
                    d_A_sub_half + (k + r_block) + k * ld, Bdesc,
                    &one,
                    d_A + (k + r_block) + (k + r_block) * ld, Cdesc,
                    d_A + (k + r_block) + (k + r_block) * ld, Cdesc,
                    &heuristicResult.algo, d_workspace, workspaceSize, stream
                )
            );

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_gemm += elapsed;

            // Calculate FLOPs for GEMM
            flops_gemm += 2.0 * sub * sub * r_block;

            // Cleanup matrix layouts
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
        }
    }

    // Free resources
    cudaFreeHost(A_00);
    cudaFree(d_A_sub_half);
    cudaFree(d_workspace);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(ltHandle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print profiling results
    std::cout << "Profiling Results (ms) & FLOPs:\n";
    std::cout << "Factorization (A00): " << time_factorize << " ms, "
              << "FLOPs: " << flops_factorize / 1e9 << " GFLOPs\n";
    std::cout << "TRSM: " << time_trsm << " ms, "
              << "FLOPs: " << flops_trsm / 1e9 << " GFLOPs\n";
    std::cout << "Conversion (Float -> Half): " << time_conversion << " ms\n";
    std::cout << "GEMM: " << time_gemm << " ms, "
              << "FLOPs: " << flops_gemm / 1e9 << " GFLOPs\n";

    return 0;
}



///fp8 version of mixed Cholesky-

// Preconditioned Conjugate Gradient (CG) solver (simplified skeleton)
// This function demonstrates the allocation, conversion, and Cholesky preconditioning.
// The actual CG loop here is simplified and uses dummy updates.
//-----------------------------
// precond_CG function
//-----------------------------
int precond_CG(double* A, double* x,
               double* b, int n, int r, double normA) {

    std::ofstream myfile("e5m2_error_f_cond.csv");

    // Create CUDA streams and cuBLAS handle.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));  // Still creating a stream for kernels/BLAS if we like
    cudaStream_t logging_stream;
    CUDA_CHECK(cudaStreamCreate(&logging_stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    double One = 1.0;
    float Onef = 1.0f;
    double Zero = 0.0;
    double NegOne = -1.0;

    

    // Allocate device memory for matrix A in double precision and single precision.
    size_t size_A_d = n * n * sizeof(double);
    size_t size_A_f = n * n * sizeof(float);
    double* d_A    = nullptr;
    float*  s_A    = nullptr;
    double* LL_copy= nullptr;

    // --- Synchronous cudaMalloc ---
    CUDA_CHECK(cudaMalloc((void**)&d_A,    size_A_d));
    CUDA_CHECK(cudaMalloc((void**)&s_A,    size_A_f));
    CUDA_CHECK(cudaMalloc((void**)&LL_copy,size_A_d));

    // Allocate device memory for vectors (synchronous).
    size_t vec_size = n * sizeof(double);
    double* dev_x = nullptr; 
    double* dev_b = nullptr; 
    double* dev_r = nullptr; 
    double* p     = nullptr; 
    double* Ap    = nullptr; 
    double* dev_z = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_z, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_x, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_r, vec_size));
    CUDA_CHECK(cudaMalloc((void**)&p,     vec_size));
    CUDA_CHECK(cudaMalloc((void**)&Ap,    vec_size));

    // Device mem for scalars (synchronous).
    size_t scal_size = sizeof(double);
    double* pAp       = nullptr;
    double* rz        = nullptr;
    double* rz_new    = nullptr;
    double* inf_norm_r= nullptr;
    double* d_One = nullptr;
    double* d_Zero = nullptr;
    double* d_NegOne = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&pAp,       scal_size));
    CUDA_CHECK(cudaMalloc((void**)&rz,        scal_size));
    CUDA_CHECK(cudaMalloc((void**)&rz_new,    scal_size));
    CUDA_CHECK(cudaMalloc((void**)&inf_norm_r,scal_size));

    CUDA_CHECK(cudaMalloc((void**)&d_One,     scal_size));
    CUDA_CHECK(cudaMalloc((void**)&d_Zero,    scal_size));
    CUDA_CHECK(cudaMalloc((void**)&d_NegOne,  scal_size));
    CUDA_CHECK(cudaMemcpy(d_One,    &One,    scal_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Zero,   &Zero,   scal_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NegOne, &NegOne, scal_size, cudaMemcpyHostToDevice));


    // Allocate device memory for preconditioner diagonal and permutations.
    double* dev_D       = nullptr;
    int* dev_left_perm  = nullptr;
    int* dev_right_perm = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_D,         vec_size));
    CUDA_CHECK(cudaMalloc((void**)&dev_left_perm,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_right_perm, n * sizeof(int)));

    // Host-side preconditioner data (already allocated).
    // For demonstration:
    double* D_host         = (double*) malloc(n * sizeof(double));
    int* left_perm_host    = (int*) malloc(n * sizeof(int));
    int* right_perm_host   = (int*) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++){
        left_perm_host[i]  = i;
        right_perm_host[i] = i;
        D_host[i] = 1.0;  // example
    }

    // Copy preconditioner data to device (synchronous).
    CUDA_CHECK(cudaMemcpy(dev_D, D_host, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_left_perm,  left_perm_host,  n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_right_perm, right_perm_host, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy matrix A and vector b from host to device (synchronous).
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A_d, cudaMemcpyHostToDevice));
    // If you really want to ensure the copy is finished here, do:
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dev_b, b, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());  // ensure dev_b is ready

    // Initialize search direction p with b (synchronous).
    CUDA_CHECK(cudaMemcpy(p, b, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert double-precision matrix to single precision (kernel uses stream).
    int totalElems = n * n;
    int threads = 256;
    int blocks = (totalElems + threads - 1) / threads;
    convertDoubleToFloat<<<blocks, threads, 0, stream>>>(d_A, s_A, totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float* diag_A = nullptr;
    float* updated_diag = nullptr;

    CUDA_CHECK(cudaMalloc((void**) & diag_A, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) & updated_diag, n*sizeof(float)));

    blocks = (n + threads - 1)/threads;
    copy_diag<<<blocks, threads, 0, stream>>>(diag_A, updated_diag, s_A, n);

    // Perform mixed-precision Cholesky factorization on s_A (with your function).
      cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    halfprec_mixed_precision_Cholesky(s_A, n, r, diag_A, updated_diag, handle, stream, n);
      CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_our = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_our, start, stop));
    printf("mixed precision cholesky took %.2f ms.\n", time_our);

    //perform trmm to see if we get back same matrix

    #ifdef DEBUG_MODE
    float* debug_mat = nullptr;
    float* host_debug_mat = (float*) malloc(n*n*sizeof(float));
    CUDA_CHECK(cudaMalloc((void**) &debug_mat, n*n*sizeof(float)));
    CUDA_CHECK(cudaMemset(debug_mat, 0.0f, n*n*sizeof(float)));
    set_identity<<<blocks, threads, 0, stream>>>(debug_mat, n);
    CUBLAS_CHECK(cublasStrmm(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        n,          // # of rows of debug_mat
        n,          // # of columns of debug_mat
        &Onef,
        s_A, n,     // pointer to L
        debug_mat, n,
        debug_mat, n
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUBLAS_CHECK(cublasStrmm(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,      // now use the transpose of L
        CUBLAS_DIAG_NON_UNIT,
        n,  // # rows
        n,  // # cols
        &Onef,
        s_A, n,
        debug_mat, n,
        debug_mat, n
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));



    #endif

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));


    // Convert factorized single-precision matrix back to double
    convertFloattoDouble<<<blocks, threads, 0, stream>>>(s_A, LL_copy, totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Solve L y = b then L^T x = y; use p as temp
    CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, p, 1));
    CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, p, 1));

    // Set dev_x = p (device to device copy - synchronous).
    CUDA_CHECK(cudaMemcpy(dev_x, p, vec_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // r = b - A*x
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, n, n, d_NegOne,
                             d_A, n, dev_x, 1, d_One, dev_b, 1));
    CUDA_CHECK(cudaMemcpy(dev_r, dev_b, vec_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));

     double cpu_inf_norm_r = 0.0;

    int vecThreads = 256;
    int vecBlocks = (n + vecThreads - 1) / vecThreads;

        // Allocate temporary device memory to store per-block maximums.
    double* dev_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_blockMax, vecBlocks * sizeof(double)));

    vanilla_Max<<<vecBlocks, vecThreads, vecThreads * sizeof(double), stream>>>(dev_r, dev_blockMax, n);

    int finalThreads = 256;

    final_max_reduce<<<1, finalThreads, finalThreads * sizeof(double), stream>>>(dev_blockMax, inf_norm_r, vecBlocks);

    // Wait for the kernels to finish.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free the temporary block maximum array.
    CUDA_CHECK(cudaFree(dev_blockMax));

    CUDA_CHECK(cudaStreamSynchronize(stream));

       
    // synchronous memcpy from device to host
    CUDA_CHECK(cudaMemcpy(&cpu_inf_norm_r, inf_norm_r, sizeof(double), cudaMemcpyDeviceToHost));

    double init_residual = cpu_inf_norm_r / normA;
    std::cout << "init residual is : " << init_residual << std::endl;


    // Compute r^T z
    CUDA_CHECK(cudaMemcpy(dev_z, dev_r, vec_size, cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, dev_z, 1));
    CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, dev_z, 1));

    CUBLAS_CHECK(cublasDdot(handle, n, dev_z, 1, dev_r, 1, rz));

    CUDA_CHECK(cudaMemcpy(p, dev_z, vec_size, cudaMemcpyDeviceToDevice));

    const double d_conv_bound = 1e-12;
    int count = 0;

    // CG loop
    const int max_CG_iter = 1000;
    for (int j = 0; j < max_CG_iter; j++) {
        // Ap = A * p
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, n, n,
                                 d_One, d_A, n, p, 1, d_Zero, Ap, 1));
        
        CUBLAS_CHECK(cublasDdot(handle, n, Ap, 1, p, 1, pAp));
        CUDA_CHECK(cudaStreamSynchronize(stream));


        // x_r_update kernel
  
        x_r_update<<<vecBlocks, vecThreads, sizeof(double), stream>>>(dev_x, dev_r, rz, pAp, p, Ap, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Compute infinity norm of residual
         vanilla_Max<<<vecBlocks, vecThreads, vecThreads * sizeof(double), stream>>>(dev_r, dev_blockMax, n);
         final_max_reduce<<<1, finalThreads, finalThreads * sizeof(double), stream>>>(dev_blockMax, inf_norm_r, vecBlocks);
        CUDA_CHECK(cudaStreamSynchronize(stream));

       
        // synchronous memcpy from device to host
        CUDA_CHECK(cudaMemcpy(&cpu_inf_norm_r, inf_norm_r, sizeof(double), cudaMemcpyDeviceToHost));

        double scaled_residual = cpu_inf_norm_r / normA;
        myfile << j << "," << scaled_residual << std::endl;
        if (scaled_residual < d_conv_bound) {
            count = j + 1;
            std::cout << "Converged in " << count << " iterations." << std::endl;

            break;
        }

        // Preconditioning
 
        CUDA_CHECK(cudaMemcpy(dev_z, dev_r, vec_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Scale dev_z by D
        //diag_scal<<<vecBlocks, vecThreads, 0, stream>>>(dev_z, dev_z, dev_D, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, dev_z, 1));
        CUBLAS_CHECK(cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, n, LL_copy, n, dev_z, 1));

        // Scale dev_z by D again
        // diag_scal<<<vecBlocks, vecThreads, 0, stream>>>(dev_z, dev_z, dev_D, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Compute r^T z
        CUBLAS_CHECK(cublasDdot(handle, n, dev_r, 1, dev_z, 1, rz_new));

        // p = z + beta * p
        update_search_dir<<<vecBlocks, vecThreads, sizeof(double), stream>>>(p, dev_z, rz_new, rz, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(rz, rz_new, sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        count++;
    }

    // Copy final solution to host
    CUDA_CHECK(cudaMemcpy(x, dev_x, vec_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(s_A));
    CUDA_CHECK(cudaFree(LL_copy));
    CUDA_CHECK(cudaFree(dev_x));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_r));
    CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(Ap));
    CUDA_CHECK(cudaFree(dev_D));
    CUDA_CHECK(cudaFree(dev_left_perm));
    CUDA_CHECK(cudaFree(dev_right_perm));
    CUDA_CHECK(cudaFree(dev_z));

    CUDA_CHECK(cudaFree(pAp));
    CUDA_CHECK(cudaFree(rz));
    CUDA_CHECK(cudaFree(rz_new));
    CUDA_CHECK(cudaFree(inf_norm_r));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(logging_stream));

    myfile.close();

    return count;
    } // End CG loop.

// Solve A*x = b using cuSOLVER's Cholesky factorization.
int solveWithCuSolver(double* A, double* x, const double* b, int n) {
    cusolverDnHandle_t cusolverH = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    
    double *d_A = nullptr;
    double *d_b = nullptr;
    int lda = n;
    size_t matrixSize = n * n * sizeof(double);
    size_t vecSize = n * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc((void**)&d_b, vecSize));
    
    // Copy A and b from host to device.
    CUDA_CHECK(cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, vecSize, cudaMemcpyHostToDevice));
    
    int work_size = 0;
    int *devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    
    // Query working space for Cholesky factorization.
    cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n, d_A, lda, &work_size);
    double *work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&work, work_size * sizeof(double)));
    
    // Cholesky factorization: A = L * L^T.
    cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, n, d_A, lda, work, work_size, devInfo);
    int devInfo_h = 0;
    CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        printf("Cholesky factorization failed with devInfo = %d\n", devInfo_h);
        exit(EXIT_FAILURE);
    }
    
    // Solve A*x = b.
    cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, n, 1, d_A, lda, d_b, n, devInfo);
    CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        printf("Cholesky solve failed with devInfo = %d\n", devInfo_h);
        exit(EXIT_FAILURE);
    }
    
    // Copy the solution from device to host.
    CUDA_CHECK(cudaMemcpy(x, d_b, vecSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(work));
    CUDA_CHECK(cudaFree(devInfo));
    cusolverDnDestroy(cusolverH);
    
    return 0;
}


//---------------------------------------------------------------------
// Main function: setup dummy problem and run preconditioned CG.
int main() {
        // Create cuSOLVER / cuBLAS handles and stream
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Problem size
    int n = 16384;
    double condVal = 100.0;
    int r = 64;

    // Allocate device memory for the SPD matrix
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, n*n*sizeof(double)));

    // Generate PSD with geometric distribution
    generatePSD(dA, n, condVal, DistType::Geometric, cublasH, cusolverH, stream);
    
    // (Alternatively, for arithmetic distribution, pass DistType::Arithmetic)

    // Copy back to host to inspect
    double* A = (double*) malloc(n*n*sizeof(double));
    CUDA_CHECK(cudaMemcpy(A, dA, n*n*sizeof(double), cudaMemcpyDeviceToHost));


    double inf_norm = 0.0;
    for (int i = 0; i < n; i++) {  // Loop over rows.
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {  // Loop over columns.
            // Since A is stored in column-major order, element (i,j) is A[i + j*n].
            row_sum += fabs(A[i + j * n]);
        }
        if (row_sum > inf_norm)
            inf_norm = row_sum;
    }
    
    // Right-hand side vector b.
    double* b = (double*) malloc(n*sizeof(double));

    for(int i = 0; i  <n; i++) b[i] = (double)rand()/(double)RAND_MAX;
    
    // Prepare containers for the solutions.
    double* x_our = (double *) malloc(n*sizeof(double));
    double* x_cusolver = (double *) malloc(n*sizeof(double));


    
    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------------------
    // Run our custom solver.
    CUDA_CHECK(cudaEventRecord(start));
    int cg_iters = precond_CG(A, x_our, b, n, r, inf_norm);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_our = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_our, start, stop));
    printf("Our solver completed in %d iterations and took %.2f ms.\n", cg_iters, time_our);
    
    // ----------------------------
    // Run cuSOLVER's solver.
    CUDA_CHECK(cudaEventRecord(start));
    solveWithCuSolver(A, x_cusolver, b, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_cusolver = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_cusolver, start, stop));
    printf("cuSOLVER completed in %.2f ms.\n", time_cusolver);

    // Compute residual of our solver: r_our = b - A*x_our
    double r_our_norm = 0.0;
    for (int i = 0; i < n; i++) {
        // Compute A[i,:] * x_our
        double Ai_x = 0.0;
        for (int j = 0; j < n; j++) {
            Ai_x += A[i + j * n] * x_our[j];
        }
        // residual component = b[i] - (A[i,:]*x_our)
        double ri = b[i] - Ai_x;
        r_our_norm += ri * ri;
    }
    r_our_norm = sqrt(r_our_norm);

    // Compute residual of cuSOLVER: r_cus = b - A*x_cusolver
    double r_cusolver_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double Ai_x = 0.0;
        for (int j = 0; j < n; j++) {
            Ai_x += A[i + j * n] * x_cusolver[j];
        }
        double ri = b[i] - Ai_x;
        r_cusolver_norm += ri * ri;
    }
    r_cusolver_norm = sqrt(r_cusolver_norm);

    // Print out the 2-norm of both residuals
    printf("Residual norm of our solver     : %e\n", r_our_norm);
    printf("Residual norm of cuSOLVER      : %e\n", r_cusolver_norm);

    
    // ----------------------------
    // Compare the solutions (compute relative L2 norm difference).
    double diff_norm = 0.0, sol_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x_our[i] - x_cusolver[i];
        diff_norm += diff * diff;
        sol_norm  += x_cusolver[i] * x_cusolver[i];
    }
    diff_norm = sqrt(diff_norm);
    sol_norm = sqrt(sol_norm);
    printf("Relative difference between our solver and cuSOLVER: %e\n", diff_norm / sol_norm);
    
    // Clean up CUDA events.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    
    return 0;
}