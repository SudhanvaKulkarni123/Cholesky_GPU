#include "kernels_macros.cuh"
#include "matgen.cuh"

using namespace std;


// Uniform precision Cholesky factorizatison in single precision.
// This function factors the matrix (in-place) stored in d_A.
int uniform_precision_Cholesky(float* d_A, int ld, int r, cublasHandle_t handle, cudaStream_t stream, int n) {
    float one    = 1.0f;
    float negOne = -1.0f;
    // Blocked Cholesky factorization
    for (int k = 0; k < n; k += r) {
        int r_block = (k + r < n) ? r : (n - k);
        // Factorize the diagonal block
        Cholesky_kernel<<<1, 256, 0, stream>>>(d_A + k + k * ld, ld, r_block);
        int sub = n - (k + r_block);
        if (sub > 0) {
            // Solve triangular system: L_{kk}^{-1} * A_{sub} = A_{sub}
            CUBLAS_CHECK(
                cublasStrsm(
                    handle,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT,
                    sub,            // M
                    r_block,        // N
                    &one,
                    d_A + k + k * ld, // L_{kk}
                    ld,
                    d_A + (k + r_block) + k * ld, // block below diagonal
                    ld
                )
            );
            // Update trailing submatrix: A = A - A_{sub} * A_{sub}^T
            CUBLAS_CHECK(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    sub,                    // M
                    sub,                    // N
                    r_block,                // K
                    &negOne,
                    d_A + (k + r_block) + k * ld, // A_{sub}
                    ld,
                    d_A + (k + r_block) + k * ld, // A_{sub}
                    ld,
                    &one,
                    d_A + (k + r_block) + (k + r_block) * ld, // trailing submatrix
                    ld
                )
            );
        }
    }
    return 0;
}

//---------------------------------------------------------------------
// Preconditioned Conjugate Gradient (CG) solver (simplified skeleton)
// This function demonstrates the allocation, conversion, and Cholesky preconditioning.
// The actual CG loop here is simplified and uses dummy updates.
int precond_CG(std::vector<double>& A, std::vector<double>& x, std::vector<double>& b, int n, int r) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Allocate GPU memory for A (double) and its single-precision version.
    size_t size_A_d = n * n * sizeof(double);
    size_t size_A_f = n * n * sizeof(float);
    double* d_A = nullptr;
    float* s_A = nullptr;
    double* D = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&d_A, size_A_d, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&s_A, size_A_f, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&D, n * sizeof(double), stream));

    // Copy matrix A (double) to device.
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), size_A_d, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Convert double matrix to single precision.
    int totalElems = n * n;
    int threads = 256;
    int blocks = (totalElems + threads - 1) / threads;
    convertDoubleToFloat<<<blocks, threads, 0, stream>>>(d_A, s_A, totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // For this demonstration, we perform a (dummy) Cholesky factorization on the single precision matrix.
    uniform_precision_Cholesky(s_A, n, r, handle, stream, n);       //change this function to mixed prec

    // --- Begin simplified CG loop ---
    const int max_CG_iter = 1000;
    const double conv_bound = 1e-6;
    int p_iter = 0;
    double rz = 1.0; // dummy initial residual dot product

    for (int j = 0; j < max_CG_iter; j++) {
        // In a real implementation, you would:
        //   - Compute Ap = A * p using cublasDgemv
        //   - Compute dot products, update x, r, and p using cublas routines
        // Here we simulate a reduction in the residual.
        double scaled_residual = 1.0 / (j + 1);
        printf("Iteration %d: Residual = %e\n", j, scaled_residual);
        if (scaled_residual < conv_bound) {
            p_iter = j + 1;
            printf("Converged in %d iterations.\n", p_iter);
            break;
        }
        // Dummy update: reduce residual value.
        rz *= 0.9;
    }
    // --- End simplified CG loop ---

    // Cleanup GPU resources.
    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(s_A, stream));
    CUDA_CHECK(cudaFreeAsync(D, stream));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return p_iter;
}

// Solve A*x = b using cuSOLVER's Cholesky factorization.
int solveWithCuSolver(const vector<double>& A, vector<double>& x, const vector<double>& b, int n) {
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
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vecSize, cudaMemcpyHostToDevice));
    
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
    x.resize(n);
    CUDA_CHECK(cudaMemcpy(x.data(), d_b, vecSize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(work));
    CUDA_CHECK(cudaFree(devInfo));
    cusolverDnDestroy(cusolverH);
    
    return 0;
}


//---------------------------------------------------------------------
// Main function: setup dummy problem and run preconditioned CG.
int main(int argc, char** argv) {
    // Problem setup.
    int n = 1024;       // Dimension of the SPD system.
    int r = 64;         // Block size for our custom solver (dummy value).

    // Build a symmetric positive definite (SPD) matrix A.
    // For simplicity, we form A = I + J (I: identity, J: ones off-diagonals).
    vector<double> A(n * n, 1.0);
    for (int i = 0; i < n; i++) {
        A[i + i * n] = 2.0;
    }
    
    // Right-hand side vector b.
    vector<double> b(n, 1.0);
    
    // Prepare containers for the solutions.
    vector<double> x_our(n, 0.0), x_cusolver;
    
    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------------------
    // Run our custom solver.
    CUDA_CHECK(cudaEventRecord(start));
    int cg_iters = precond_CG(A, x_our, b, n, r);
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