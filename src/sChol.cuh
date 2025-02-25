// void Ssyrk_SGemm(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
//                  int k, int n, const cutlass::half_t* alpha, const cutlass::half_t* A, int lda, 
//                  const float* beta, float* C, int ldc) 
// {
//     // Loop over the blocks of the symmetric rank-k update
//     for (int i = 0; i < n; i += k) 
//     {
//         for (int j = 0; j <= i; j += k) 
//         {
//             // Determine sub-matrix dimensions (for boundary handling)
//             int ib = min(k, n - i);  // Rows in this block
//             int jb = min(k, n - j);  // Columns in this block
            
//             // Compute C_ij = A_ik * A_jk^T (lower triangular update)
//             CHECK_CUBLAS(cublasGemmEx(handle,
//                                       trans, CUBLAS_OP_T,  // Multiply A_ik * A_jk^T
//                                       ib, jb, k,           // Matrix dimensions
//                                       alpha,               // Scaling factor
//                                       A + i * lda, CUDA_R_16F, lda,  // Matrix A_ik
//                                       A + j * lda, CUDA_R_16F, lda,  // Matrix A_jk
//                                       beta,                // Scaling factor for C
//                                       C + i * ldc + j, CUDA_R_32F, ldc,  // Output C_ij
//                                       CUBLAS_COMPUTE_32F_FAST_TF32,  // Compute in FP32
//                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Enable Tensor Cores
//             ));
//         }
//     }
// }


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

// CUBLAS_CHECK(
//     cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
//                 sub, r_block, &negOne,
//                 d_A + (k + r_block) + k * ld, ld,  // A_sub
//                 &one, d_A + (k + r_block) + (k + r_block) * ld, ld) // A_trailing
// );

CUBLAS_CHECK( 
cublasSgemm(handle,
CUBLAS_OP_N, CUBLAS_OP_T,   // A is not transposed; B is transposed (i.e. A^T)
sub, sub, r_block,          // dimensions: C (sub x sub) = A (sub x r_block) * A^T (r_block x sub)
&negOne,                    // alpha = -1
d_A + (k + r_block) + k * ld, ld, // A_sub pointer, leading dimension ld
d_A + (k + r_block) + k * ld, ld, // same pointer for A_sub, but transposed in GEMM call
&one,                       // beta = 1
d_A + (k + r_block) + (k + r_block) * ld, ld) // C pointer (A_trailing), leading dimension ld
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



int uniform_prec_GPU_cholesky(
    float* d_A,         // row-major NxN matrix on device
    int ld,             // leading dimension = number of columns = N
    int r,              // block size
    float* diag_A,      // (not used in this snippet)
    float* updated_diag,// (not used in this snippet)
    cublasHandle_t handle,
    cusolverDnHandle_t cusolverH,
    cudaStream_t stream,
    int n )
{
    float one    = 1.0f;
    float negOne = -1.0f;

    // We'll need devInfo for cuSOLVER
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    // Additional CUDA stream for GEMM (SYRK-like update)
    cudaStream_t gemmStream;
    cudaStreamCreate(&gemmStream);

    // CUDA events for timing/profiling
    cudaEvent_t start, stop, panel_done, trsm_done;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&panel_done);
    cudaEventCreate(&trsm_done);

    float time_factorize = 0.0f, time_trsm = 0.0f, time_gemm = 0.0f;

    // First, query workspace needed for Cholesky (panel_size <= r)
    int lwork = 0;
    CUSOLVER_CHECK(
        cusolverDnSpotrf_bufferSize(
            cusolverH,
            CUBLAS_FILL_MODE_LOWER, // We'll store the factor in the "upper" part for row-major
            r,                      // max block size
            d_A,                    // just pass a valid device pointer
            ld,
            &lwork
        )
    );

    // Allocate workspace for the panel factorization
    float* panel_workspace = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&panel_workspace, sizeof(float)*lwork));

    // Blocked factor
    for (int k = 0; k < n; k += r) 
    {
        int r_block = (k + r < n) ? r : (n - k);

        // Start timing the "panel factorization"
        cudaEventRecord(start, stream);

        // ---------------------------------------------------------
        //  1) Factor the diagonal block A[k..k+r_block, k..k+r_block]
        //     In row-major, that block starts at offset:
        //        row = k, col = k --> d_A + (k*ld + k)
        // ---------------------------------------------------------
        float* d_panel = d_A + (k * ld + k);

        // Spotrf in-place: upper triangle for row-major
        CUSOLVER_CHECK(
            cusolverDnSpotrf(
                cusolverH,
                CUBLAS_FILL_MODE_LOWER,  // store factor in 'upper' for row-major
                r_block,
                d_panel,
                ld,
                panel_workspace,
                lwork,
                devInfo
            )
        );

        // Check devInfo if factorization is successful
        {
            int info_cpu = 0;
            CUDA_CHECK(cudaMemcpyAsync(&info_cpu, devInfo, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
            if (info_cpu != 0) {
                std::cerr << "Cholesky factorization failed at block k=" 
                          << k << " with info=" << info_cpu << std::endl;
                // handle or exit...
            }
        }

        // Panel factorization done
        cudaEventRecord(panel_done, stream);

        // Stop timer for factorization
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_factorize += elapsed;

        // ---------------------------------------------------------
        //  2) TRSM to solve the off-diagonal blocks:
        //     A[k..k+r_block, k+r_block..n] gets used as the factor (U)
        //     We update the block B = A[k, k+r_block..n], shape sub x r_block
        //     but from the row-major viewpoint, the factor is 'upper'
        // ---------------------------------------------------------
        int sub = n - (k + r_block);
        if (sub > 0) {
            // Start TRSM timing
            cudaEventRecord(start, stream);

            // B <- B * U^{-1}, side=RIGHT, fill=UPPER, op=N
                        
            CUBLAS_CHECK(
                cublasStrsm(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                sub, r_block, &one,
                d_A + k + k * ld, ld,
                d_A + (k + r_block) + k * ld, ld)
                );
                
            // Signal TRSM done
            cudaEventRecord(trsm_done, stream);

            // Stop TRSM timing
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_trsm += elapsed;

            // -----------------------------------------------------
            //  3) GEMM-like update (SYRK): 
            //      A[(k+r_block)..n, (k+r_block)..n] -= B * B^T
            //     We'll do it asynchronously on gemmStream
            // -----------------------------------------------------
            cudaStreamWaitEvent(gemmStream, trsm_done, 0);  // wait for TRSM

            cudaEventRecord(start, gemmStream);

            // From row-major viewpoint, B is sub x r_block.
            // But cuBLAS is column-major. 
            // We fix that by telling cuBLAS to transpose the first operand:
            //   B is "CUBLAS_OP_T" => shape becomes (r_block x sub)
            //   B^T is "CUBLAS_OP_N" => shape (r_block x sub)
            // => result sub x sub
            CUBLAS_CHECK(
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,   // A is not transposed; B is transposed (i.e. A^T)
                sub, sub, r_block,          // dimensions: C (sub x sub) = A (sub x r_block) * A^T (r_block x sub)
                &negOne,                    // alpha = -1
                d_A + (k + r_block) + k * ld, ld, // A_sub pointer, leading dimension ld
                d_A + (k + r_block) + k * ld, ld, // same pointer for A_sub, but transposed in GEMM call
                &one,                       // beta = 1
                d_A + (k + r_block) + (k + r_block) * ld, ld) // C pointer (A_trailing), leading dimension ld
                );

            cudaEventRecord(stop, gemmStream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_gemm += elapsed;
        }
    }

    // Cleanup
    cudaFree(panel_workspace);
    cudaFree(devInfo);

    cudaStreamDestroy(gemmStream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(panel_done);
    cudaEventDestroy(trsm_done);

    // Print profiling results
    std::cout << "Profiling Results (ms):\n";
    std::cout << "  Factorization: " << time_factorize << " ms\n";
    std::cout << "  TRSM:         " << time_trsm       << " ms\n";
    std::cout << "  GEMM (SYRK):  " << time_gemm       << " ms\n";

    return 0;
}


int uniform_prec_fused_cholesky(   float* d_A,         // row-major NxN matrix on device
    int ld,             // leading dimension = number of columns = N
    int r,              // block size
    float* diag_A,      // (not used in this snippet)
    float* updated_diag,// (not used in this snippet)
    cublasHandle_t handle,
    cusolverDnHandle_t cusolverH,
    cudaStream_t stream,
    int n )
{

   // 1) For "getrfnp", we need a device workspace "d_work" and devInfo
   int devInfo_h = 0;
   int* devInfo  = nullptr;
   CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );

   float one = 1.0f;
   float negOne = -1.0f;

   // This will store the max workspace needed for factorizing a submatrix of size (sub x r_block)
   int max_lwork = 0;
   float panel_time = 0.0f;
   float GEMM_time = 0.0f;

   // Create events for optional timing (if desired)
   cudaEvent_t startEvt, stopEvt;
   cudaEventCreate(&startEvt);
   cudaEventCreate(&stopEvt);

   //allocate largest possible LU buffer-
   int lwork = 0;
   CUSOLVER_CHECK(
    cusolverDnSgetrf_bufferSize(cusolverH,
        n,
        r,
        d_A,
        ld,
        &lwork )
);


           // 2.2) Allocate device workspace
           float* d_work = nullptr;
           CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));


   // 2) Blocked factorization loop
   //    We'll factor in columns of width 'block_size' from left to right.
   for(int k = 0; k < n; k += r)
   {
       // Current block width
       int r_block = (k + r < n) ? r : (n - k);
       // Remaining submatrix height
       int sub = n - k;  // from row k..(n-1)

       // Panel pointer in row-major: row=k, col=k => d_A + (k*ld + k)
       float* d_panel = d_A + (k * ld + k);

       // ---------------------------------------------------------
       // (A) Panel factorization (getrfnp) on sub x r_block block
       //     i.e. factor A[k..n-1, k..k+r_block-1] in place
       // ---------------------------------------------------------
       {
           
           // 2.3) Perform factorization
           CUDA_CHECK(cudaEventRecord(startEvt, stream));
           CUSOLVER_CHECK(
            cusolverDnSgetrf(cusolverH,
                sub,
                r_block,
                d_panel,
                n,
                d_work,
                (int*) nullptr,
                devInfo )
           );
           CUDA_CHECK(cudaEventRecord(stopEvt, stream));
           CUDA_CHECK(cudaEventSynchronize(stopEvt));

           float ms = 0.0f;
           cudaEventElapsedTime(&ms, startEvt, stopEvt);
           panel_time += ms;

           // 2.4) Check devInfo
           CUDA_CHECK(cudaMemcpyAsync(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
           cudaStreamSynchronize(stream);
           if(devInfo_h != 0) {
               std::cerr << "Error: getrfnp failed at block k=" << k 
                         << " with devInfo=" << devInfo_h << "\n";
               // Possibly handle or return error code
           }

      
       }

       // ---------------------------------------------------------
       // (B) Trailing submatrix update
       //     We want to apply the block factor to A[k..n-1, k+r_block..n-1].
       //     That is typically "U" in the top r_block rows of d_panel,
       //     plus "L" in the bottom part. However, getrfnp solves
       //     that panel in place. We just do the standard block-LU
       //     trailing update: A_trailing -= L_panel * U_panel
       // ---------------------------------------------------------
       int trailing_cols = n - (k + r_block);
       if(trailing_cols > 0)
       {
           // L_panel is sub x r_block, U_panel is r_block x trailing_cols (in row-major).
           // We need to multiply L (sub x r_block) times U (r_block x trailing_cols)
           // and subtract from the trailing submatrix A[(k)..(n-1), (k+r_block)..(n-1)].

          
           cudaEventRecord(startEvt, stream);

           
           CUBLAS_CHECK(
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,   // A is not transposed; B is transposed (i.e. A^T)
                sub, sub, r_block,          // dimensions: C (sub x sub) = A (sub x r_block) * A^T (r_block x sub)
                &negOne,                    // alpha = -1
                d_A + (k + r_block) + k * ld, ld, // A_sub pointer, leading dimension ld
                d_A + (k + r_block) + k * ld, ld, // same pointer for A_sub, but transposed in GEMM call
                &one,                       // beta = 1
                d_A + (k + r_block) + (k + r_block) * ld, ld) // C pointer (A_trailing), leading dimension ld
                );

           cudaEventRecord(stopEvt, stream);
           cudaEventSynchronize(stopEvt);

           float ms;
           cudaEventElapsedTime(&ms, startEvt, stopEvt);
           GEMM_time += ms;
           // Optional: std::cout << "[Trailing update] took " << ms << " ms\n";
       }

   } // end for k

   cudaFree(devInfo);
   cudaEventDestroy(startEvt);
   cudaEventDestroy(stopEvt);
        // Cleanup panel workspace
        cudaFree(d_work);

   std::cout << "Blocked LU (no pivot) completed.\n"
             << "   Max workspace needed = " << max_lwork << " floats\n";
   std::cout << " time taken for panel factrization = " << panel_time << "ms\n";
   std::cout << "time taken for GEMM = " << GEMM_time << "ms\n";

   return 0;
}