// generate_matrix.cu
#include "kernels_macros.cuh"


void generateMatrix(int n, double** d_A_out) {
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    // Allocate host memory for a random n-by-n matrix.
    double* h_A = (double*)malloc(n * n * sizeof(double));
    if (h_A == nullptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }


    
    // Allocate device memory for the input matrix.
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
        // Fill with random values.
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));


    CURAND_CHECK(curandGenerateNormalDouble(gen, d_A, n, 0.0, 1.0));

    // --- Compute QR factorization to obtain Q ---
    // Allocate space for tau vector (length = n).
    double* d_tau = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(double)));

    // Query workspace size for geqrf.
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, n, n, d_A, n, &work_size));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work, work_size * sizeof(double)));

    // Allocate device memory for devInfo.
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    // Compute QR factorization: d_A will contain R in the upper triangle and Householder vectors in the lower triangle.
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, n, n, d_A, n, d_tau, d_work, work_size, devInfo));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        fprintf(stderr, "QR factorization failed, info = %d\n", info);
        exit(EXIT_FAILURE);
    }
    
    // Generate Q explicitly from the output of geqrf.
    CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, n, n, n, d_A, n, d_tau, d_work, work_size, devInfo));
    CUDA_CHECK(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        fprintf(stderr, "Generating Q failed, info = %d\n", info);
        exit(EXIT_FAILURE);
    }
    // Now, d_A contains the orthonormal matrix Q.

    // --- Create diagonal Sigma ---
    // On the host, form sigma[i] = 1/(i+1).
    double* h_sigma = (double*)malloc(n * sizeof(double));
    if (h_sigma == nullptr) {
        fprintf(stderr, "Failed to allocate h_sigma.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        h_sigma[i] = 1.0 / (i + 1);
    }
    double* d_sigma = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_sigma, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, n * sizeof(double), cudaMemcpyHostToDevice));

    // --- Form B = Q * Sigma ---
    // We want to multiply each column j of Q by sigma[j]. Since Q is in d_A,
    // we allocate a new matrix d_Q to store the scaled Q.
    double* d_Q = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Q, d_A, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
    // For each column j, scale d_Q(:,j) by sigma[j].
    for (int j = 0; j < n; j++) {
        // Pointer to the j-th column is d_Q + j*n.
        // Note: cublasDscal scales a vector of length n.
        CUBLAS_CHECK(cublasDscal(cublasH, n, &h_sigma[j], d_Q + j * n, 1));
    }
    // At this point, d_Q = Q * diag(sigma).

    // --- Compute A = B * Q^T, where B = d_Q and Q is still in d_A ---
    double* d_A_out_local = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A_out_local, n * n * sizeof(double)));
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, n, n,
                             &alpha,
                             d_Q, n,
                             d_A, n,
                             &beta,
                             d_A_out_local, n));
    // Now, d_A_out_local contains A = Q * diag(sigma) * Q^T.

    // Return the result.
    *d_A_out = d_A_out_local;

    // --- Clean up ---
    free(h_A);
    free(h_sigma);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_sigma));
    CUDA_CHECK(cudaFree(d_Q));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

// Test main to call generateMatrix and print a few entries.
int main() {
    int n = 8;  // For demonstration, use a small matrix.
    double* d_A = nullptr;
    generateMatrix(n, &d_A);

    // Copy the generated matrix back to host and print it.
    double* h_A = (double*)malloc(n * n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Generated matrix A = Q * Sigma * Q^T:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", h_A[i + j * n]);  // Column-major print.
        }
        printf("\n");
    }

    free(h_A);
    CUDA_CHECK(cudaFree(d_A));
    return 0;
}
