#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define M 4096  // Number of rows (tall)
#define N 512    // Number of columns (skinny)
#define NUM_RUNS 10  // Number of iterations for averaging performance

#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#define CUSOLVER_CHECK(call)                                                \
    {                                                                       \
        cusolverStatus_t err = call;                                        \
        if (err != CUSOLVER_STATUS_SUCCESS) {                               \
            std::cerr << "cuSOLVER Error at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

int main() {
    cusolverDnHandle_t cusolverH;
    (cusolverDnCreate(&cusolverH));

    float *d_A;
    int *d_pivot, *d_info;
    int lwork = 0;
    float *d_work;

    // Allocate matrix (row-major layout: MxN)
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_pivot, N * sizeof(int)));  // Only N pivots
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));       // For error check

    // Query workspace size for GETRF
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolverH, M, N, d_A, M, &lwork));

    // Allocate workspace
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark cuSOLVER GETRF over multiple runs
    float total_time_ms = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        // Perform LU factorization (in-place)
        CUSOLVER_CHECK(cusolverDnSgetrf(cusolverH, M, N, d_A, M, d_work, d_pivot, d_info));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Compute average time and GFLOPS
    float avg_time_ms = total_time_ms / NUM_RUNS;
    double gflops = (2.0 * M * N * N) / (3.0 * avg_time_ms * 1e6); // Approximate GFLOPS formula

    std::cout << "cuSOLVER GETRF Performance for " << M << "x" << N << " matrix:\n";
    std::cout << "  Avg Time: " << avg_time_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << " GFLOP/s\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
 (cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
