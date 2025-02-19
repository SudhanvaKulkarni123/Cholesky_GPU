int fp8_mixed_precision_Cholesky(float* d_A, int ld, int r,
                                      float* diag_A, float* updated_diag,
                                      cublasHandle_t handle,
                                      cublasLtHandle_t ltHandle, cudaStream_t stream,
                                      int n) {
    float one = 1.0f;
    float negOne = -1.0f;

    // Allocate FP8 buffer for A_sub
    uint8_t* d_A_sub_fp8;
    CUDA_CHECK(cudaMalloc((void**)&d_A_sub_fp8, sizeof(uint8_t) * n * r));

    //Allocate buffers to calculate max_L when rounbding L to fp8
    int vecThreads = 256;
    int vecBlocks = (n*n + vecThreads - 1) / vecThreads;

    // Allocate temporary device memory to store per-block maximums.
    float* dev_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_blockMax, vecBlocks * sizeof(float)));
    float* max_L = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &max_L, sizeof(float)));

    //A_00 for CPU factorization
    float* A_00 = (float*) malloc(sizeof(float) * r * r);

    // CUDA Events for Profiling
    cudaEvent_t start, stop;
    float time_factorize = 0, time_trsm = 0, time_conversion = 0, time_gemm = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Variables for FLOP counts
    double flops_factorize = 0, flops_trsm = 0, flops_gemm = 0;

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

            // Start timing conversion (Float -> FP8)
            cudaEventRecord(start, stream);



            vanilla_Max_2D_col_major<<<vecBlocks, vecThreads, vecThreads * sizeof(double), stream>>>(d_A + (k + r_block) + k * ld, dev_blockMax, r_block, sub, n);

            int finalThreads = 256;

            final_max_reduce<<<1, finalThreads, finalThreads * sizeof(double), stream>>>(dev_blockMax, max_L, vecBlocks);


            dim3 blockSize(16, 16);
            dim3 gridSize((sub + blockSize.x - 1) / blockSize.x, (r_block + blockSize.y - 1) / blockSize.y);
            floatToFp8E4M3Kernel<<<gridSize, blockSize, 0, stream>>>(
                d_A + (k + r_block) + k * ld,
                d_A_sub_fp8 + (k + r_block) + k * ld,
                sub, r_block, ld, ld
            );

            cudaStreamSynchronize(stream);

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_conversion += elapsed;

            // Start timing GEMM (FP8 MatMul using cuBLASLt)
            cudaEventRecord(start, stream);

            // cuBLASLt MatMul descriptors
            cublasLtMatmulDesc_t matmulDesc;
            cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
            cublasLtMatmulPreference_t preference;
            cublasLtMatmulAlgo_t algo;

            // Create cuBLASLt objects
            cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, sub, r_block, ld);
            cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, sub, r_block, ld);
            cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, sub, sub, ld);

            // Set preference
            cublasLtMatmulPreferenceCreate(&preference);
            int returnedResults = 0;
            cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC, preference, 1, &algo, &returnedResults);

            // Perform FP8 GEMM
            CUBLAS_CHECK(cublasLtMatmul(
                ltHandle,
                matmulDesc,
                &negOne,
                d_A_sub_fp8, layoutA,
                d_A_sub_fp8, layoutB,
                &one,
                d_A + (k + r_block) + (k + r_block) * ld, layoutC,
                d_A + (k + r_block) + (k + r_block) * ld, layoutC,
                &algo,
                nullptr, 0, 0));

            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_gemm += elapsed;

            // Calculate FLOPs for GEMM
            flops_gemm += 2.0 * sub * sub * r_block;

            // Cleanup cuBLASLt descriptors
            cublasLtMatmulDescDestroy(matmulDesc);
            cublasLtMatrixLayoutDestroy(layoutA);
            cublasLtMatrixLayoutDestroy(layoutB);
            cublasLtMatrixLayoutDestroy(layoutC);
            cublasLtMatmulPreferenceDestroy(preference);
        }
    }

    // Free resources
    free(A_00);
    cudaFree(d_A_sub_fp8);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print profiling results
    std::cout << "Profiling Results (ms) & FLOPs:\n";
    std::cout << "Factorization (A00): " << time_factorize << " ms, "
              << "FLOPs: " << flops_factorize / 1e9 << " GFLOPs\n";
    std::cout << "TRSM: " << time_trsm << " ms, "
              << "FLOPs: " << flops_trsm / 1e9 << " GFLOPs\n";
    std::cout << "Conversion (Float -> FP8): " << time_conversion << " ms\n";
    std::cout << "GEMM: " << time_gemm << " ms, "
              << "FLOPs: " << flops_gemm / 1e9 << " GFLOPs\n";

    return 0;
}
//---------------------------------------------------------------------