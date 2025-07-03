#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "kernels_macros.cuh"

double estimateConditionNumber(double* dA,
                               int n,
                               cusolverDnHandle_t cusolverH,
                               cudaStream_t stream)
{
    // 1. Copy dA -> d_workA (so we don't destroy dA)
    double* d_workA = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_workA, n*n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_workA, dA, n*n*sizeof(double), cudaMemcpyDeviceToDevice));

    // 2. Allocate space for eigenvalues
    double* d_W = nullptr;  // holds eigenvalues
    CUDA_CHECK(cudaMalloc((void**)&d_W, n*sizeof(double)));

    // 3. Create devInfo and query workspace size
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    
    int lwork = 0;
    // We only need eigenvalues, but must choose EIG_MODE_VECTOR or NOVECTOR
    // Either is fine if we only care about the eigenvalues, but 'VECTOR' can help
    // preserve the full result if needed. We'll do EIG_MODE_NOVECTOR for minimal overhead.
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;

    // We'll assume the matrix is stored in lower or upper. Typically, for SPD, we use
    // the lower triangular part. But syevd can handle either. Let's pick 'LOWER'.
    CUSOLVER_CHECK( cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_workA,
        n,
        d_W,
        &lwork
    ));

    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork*sizeof(double)));

    // 4. Compute eigenvalues in d_W (d_workA gets overwritten)
    CUSOLVER_CHECK( cusolverDnDsyevd(
        cusolverH,
        jobz,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_workA, // overwritten
        n,
        d_W,
        d_work,
        lwork,
        devInfo
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 5. Check devInfo
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(info != 0) {
        std::cerr << "dsyevd failed with devInfo = " << info << std::endl;
        // Clean up and return something safe
        cudaFree(d_workA);
        cudaFree(d_W);
        cudaFree(d_work);
        cudaFree(devInfo);
        return -1.0;
    }

    // 6. Copy eigenvalues back to host
    std::vector<double> h_W(n);
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W, n*sizeof(double), cudaMemcpyDeviceToHost));

    // Eigenvalues are returned in ascending order => W[0]..W[n-1]
    double lambda_min = h_W[0];
    double lambda_max = h_W[n-1];

    // Condition number ~ (lambda_max / lambda_min)
    double condNum = (lambda_min > 0.0) ? (lambda_max / lambda_min) : -1.0;

    // Cleanup
    CUDA_CHECK(cudaFree(d_workA));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(devInfo));

    return condNum;
}


void randomOrthonormalMatrix(double* d_Q, int n, 
                             cusolverDnHandle_t cusolverH, 
                             cudaStream_t stream)
{
    // (A) Allocate host memory for M, fill with randoms
    std::vector<double> h_M(n*n);
    std::srand(static_cast<unsigned>(123444));
    for(int i = 0; i < n*n; ++i){
        h_M[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // (B) Copy M to device
    double* d_M = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_M, n*n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));

    // (C) QR factorization via cuSOLVER
    int work_size = 0;
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    // 1. Query buffer size
    CUSOLVER_CHECK( cusolverDnDgeqrf_bufferSize(cusolverH, n, n, d_M, n, &work_size) );
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work, work_size * sizeof(double)));

    // ================================
    // Use DEVICE memory for tau
    double* d_tau = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(double)));
    // ================================

    // 2. Factor M in-place => M = Q * R  (geqrf)
    CUSOLVER_CHECK(cusolverDnDgeqrf(
        cusolverH, 
        n, n, 
        d_M, n, 
        d_tau,         // <--- DEVICE pointer for tau
        d_work, work_size, 
        devInfo
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // check info
    int infoQR = 0;
    CUDA_CHECK(cudaMemcpy(&infoQR, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(infoQR != 0){
        std::cerr << "QR factorization failed. devInfo=" << infoQR << std::endl;
    }

    // 3. Extract Q (orgqr)
    CUSOLVER_CHECK(cusolverDnDorgqr(
        cusolverH,
        n,  // m
        n,  // n
        n,  // k
        d_M, n,
        d_tau,       // <--- DEVICE pointer for tau
        d_work, work_size,
        devInfo
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    int infoOrgqr = 0;
    CUDA_CHECK(cudaMemcpy(&infoOrgqr, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(infoOrgqr != 0){
        std::cerr << "orgqr failed. devInfo=" << infoOrgqr << std::endl;
    }

    // Now d_M holds Q
    CUDA_CHECK(cudaMemcpy(d_Q, d_M, n*n*sizeof(double), cudaMemcpyDeviceToDevice));

    // cleanup
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_tau));   // free device tau
    CUDA_CHECK(cudaFree(devInfo));
}


// ---------------------------------------------------------------------------
// Distribution type
enum class DistType {
    Geometric,
    Arithmetic
};

// ---------------------------------------------------------------------------
// 5. Generate a PSD matrix A = Q * Lambda * Q^T on the device.
//
// dA:      (output) device pointer for NxN matrix
// n:       dimension
// condVal: desired condition number
// dist:    pick geometric or arithmetic distribution
//
void generatePSD(double* dA, 
                 int n, 
                 double condVal, 
                 DistType dist,
                 cublasHandle_t cublasH,
                 cusolverDnHandle_t cusolverH,
                 cudaStream_t stream)
{
    // 0. constants
    const double alpha = 1.0;
    const double beta  = 0.0;

    // 1. Allocate device memory for Q
    double* d_Q = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, n*n*sizeof(double)));

    double* d_buff = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buff, n*n*sizeof(double)));

    double* debug_arr = (double*) malloc(n*sizeof(double));

    // 2. Generate random orthonormal Q
    randomOrthonormalMatrix(d_Q, n, cusolverH, stream);

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    //check if mat is orth
    double* debug_mat = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &debug_mat, n*n*sizeof(double)));
    CUBLAS_CHECK(cublasDgemm(
        cublasH,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        d_Q, n,
        d_Q, n, 
        &beta,
        debug_mat, n
    ));


    // 3. Allocate diagonal array d_diagVals (size n)
    double* d_diagVals = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_diagVals, n*n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_diagVals, 0, n * n * sizeof(double)));

    // 3a. Copy condVal to device
    double* d_cond = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_cond, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cond, &condVal, sizeof(double), cudaMemcpyHostToDevice));

    // 4. Fill diagVals with chosen distribution
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    unsigned long long seed = 123456789; // Random seed for reproducibility
    if (dist == DistType::RandomLogUniform) {
        construct_diag_logrand<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n, seed);
    } else if (dist == DistType::Clustered) {
        construct_diag_clustered<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
    } else if (dist == DistType::Arithmetic) {
        construct_diag_arith<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
    } else if (dist == DistType::Geometric) {
        construct_diag_geom<<<gridSize, blockSize, 0, stream>>>(d_diagVals, d_cond, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
 
    CUBLAS_CHECK(cublasDgemm( 
        cublasH, 
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        d_diagVals, n,
        d_Q, n,
        &beta,
        d_buff, n
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 6. Now compute A = (Q * diag) * Q^T
    //    i.e. gemm( Q, Q^T ) with cublas

    CUBLAS_CHECK(cublasDgemm(
        cublasH,
        CUBLAS_OP_T,   // Q not transposed
        CUBLAS_OP_N,   // Q^T
        n,             // m
        n,             // n
        n,             // k
        &alpha,
        d_Q, n,
        d_buff, n,
        &beta,
        dA, n
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_diagVals));
    CUDA_CHECK(cudaFree(d_cond));
    CUDA_CHECK(cudaFree(d_buff));
}
