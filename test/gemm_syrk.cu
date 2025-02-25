#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <nvToolsExt.h>
#include <cstdlib>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUBLAS_CHECK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLASassert: %d %s %d\n", status, file, line);
        exit(status);
    }
}

// Random matrix fill function
void randomFill(float* arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main() {
    // --------------------------------------------------
    // 1. Large matrix example: 2048x2048x2048
    // --------------------------------------------------
    const int N = 2048, K = 2048;
    const float alpha = 1.0f, beta = 0.0f;
    float *h_A = new float[N * K];
    float *h_C_sgemm = new float[N * N];
    float *h_C_syrk = new float[N * N];
    randomFill(h_A, N * K);

    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // --------------------------------------------------
    // 2. Smaller examples: 128x128x128, 64x64x64, 144x144x144,
    //    160x160x160, 256x256x256, 240x240x240, and 272x272x272
    // --------------------------------------------------
    // 128 example
    const int N1 = 128, K1 = 128;
    float *h_A1 = new float[N1 * K1];
    float *h_C1_sgemm = new float[N1 * N1];
    float *h_C1_syrk = new float[N1 * N1];
    randomFill(h_A1, N1 * K1);
    float *d_A1, *d_C1;
    CUDA_CHECK(cudaMalloc((void**)&d_A1, N1 * K1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C1, N1 * N1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A1, h_A1, N1 * K1 * sizeof(float), cudaMemcpyHostToDevice));

    // 64 example
    const int N2 = 64, K2 = 64;
    float *h_A2 = new float[N2 * K2];
    float *h_C2_sgemm = new float[N2 * N2];
    float *h_C2_syrk = new float[N2 * N2];
    randomFill(h_A2, N2 * K2);
    float *d_A2, *d_C2;
    CUDA_CHECK(cudaMalloc((void**)&d_A2, N2 * K2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C2, N2 * N2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A2, h_A2, N2 * K2 * sizeof(float), cudaMemcpyHostToDevice));

    // 144 example (144 = 9 * 16)
    const int N3 = 144, K3 = 144;
    float *h_A3 = new float[N3 * K3];
    float *h_C3_sgemm = new float[N3 * N3];
    float *h_C3_syrk = new float[N3 * N3];
    randomFill(h_A3, N3 * K3);
    float *d_A3, *d_C3;
    CUDA_CHECK(cudaMalloc((void**)&d_A3, N3 * K3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C3, N3 * N3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A3, h_A3, N3 * K3 * sizeof(float), cudaMemcpyHostToDevice));

    // 160 example (160 = 10 * 16)
    const int N4 = 160, K4 = 160;
    float *h_A4 = new float[N4 * K4];
    float *h_C4_sgemm = new float[N4 * N4];
    float *h_C4_syrk = new float[N4 * N4];
    randomFill(h_A4, N4 * K4);
    float *d_A4, *d_C4;
    CUDA_CHECK(cudaMalloc((void**)&d_A4, N4 * K4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C4, N4 * N4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A4, h_A4, N4 * K4 * sizeof(float), cudaMemcpyHostToDevice));

    // 256 example
    const int N5 = 256, K5 = 256;
    float *h_A5 = new float[N5 * K5];
    float *h_C5_sgemm = new float[N5 * N5];
    float *h_C5_syrk = new float[N5 * N5];
    randomFill(h_A5, N5 * K5);
    float *d_A5, *d_C5;
    CUDA_CHECK(cudaMalloc((void**)&d_A5, N5 * K5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C5, N5 * N5 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A5, h_A5, N5 * K5 * sizeof(float), cudaMemcpyHostToDevice));

    // New: 240 example (240 = 15 * 16)
    const int N6 = 240, K6 = 240;
    float *h_A6 = new float[N6 * K6];
    float *h_C6_sgemm = new float[N6 * N6];
    float *h_C6_syrk = new float[N6 * N6];
    randomFill(h_A6, N6 * K6);
    float *d_A6, *d_C6;
    CUDA_CHECK(cudaMalloc((void**)&d_A6, N6 * K6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C6, N6 * N6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A6, h_A6, N6 * K6 * sizeof(float), cudaMemcpyHostToDevice));

    // New: 272 example (272 = 17 * 16)
    const int N7 = 272, K7 = 272;
    float *h_A7 = new float[N7 * K7];
    float *h_C7_sgemm = new float[N7 * N7];
    float *h_C7_syrk = new float[N7 * N7];
    randomFill(h_A7, N7 * K7);
    float *d_A7, *d_C7;
    CUDA_CHECK(cudaMalloc((void**)&d_A7, N7 * K7 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C7, N7 * N7 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A7, h_A7, N7 * K7 * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle and CUDA events for timing
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timing variables
    float time_sgemm = 0.0f, time_syrk = 0.0f;
    float time_sgemm_128 = 0.0f, time_syrk_128 = 0.0f;
    float time_sgemm_64 = 0.0f, time_syrk_64 = 0.0f;
    float time_sgemm_144 = 0.0f, time_syrk_144 = 0.0f;
    float time_sgemm_160 = 0.0f, time_syrk_160 = 0.0f;
    float time_sgemm_256 = 0.0f, time_syrk_256 = 0.0f;
    float time_sgemm_240 = 0.0f, time_syrk_240 = 0.0f;
    float time_sgemm_272 = 0.0f, time_syrk_272 = 0.0f;

    // --------------------------------------------------
    // 1. SGEMM and SSYRK for 2048x2048x2048
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
    nvtxRangePushA("SGEMM_2048");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, N, K,
                             &alpha, d_A, N, d_A, N,
                             &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
    nvtxRangePushA("SSYRK_2048");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N, K,
                             &alpha, d_A, N,
                             &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 2. SGEMM and SSYRK for 128x128x128
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C1, 0, N1 * N1 * sizeof(float)));
    nvtxRangePushA("SGEMM_128");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N1, N1, K1,
                             &alpha, d_A1, N1, d_A1, N1,
                             &beta, d_C1, N1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_128, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C1, 0, N1 * N1 * sizeof(float)));
    nvtxRangePushA("SSYRK_128");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N1, K1,
                             &alpha, d_A1, N1,
                             &beta, d_C1, N1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_128, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 3. SGEMM and SSYRK for 64x64x64
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C2, 0, N2 * N2 * sizeof(float)));
    nvtxRangePushA("SGEMM_64");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N2, N2, K2,
                             &alpha, d_A2, N2, d_A2, N2,
                             &beta, d_C2, N2));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_64, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C2, 0, N2 * N2 * sizeof(float)));
    nvtxRangePushA("SSYRK_64");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N2, K2,
                             &alpha, d_A2, N2,
                             &beta, d_C2, N2));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_64, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 4. SGEMM and SSYRK for 144x144x144
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C3, 0, N3 * N3 * sizeof(float)));
    nvtxRangePushA("SGEMM_144");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N3, N3, K3,
                             &alpha, d_A3, N3, d_A3, N3,
                             &beta, d_C3, N3));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_144, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C3, 0, N3 * N3 * sizeof(float)));
    nvtxRangePushA("SSYRK_144");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N3, K3,
                             &alpha, d_A3, N3,
                             &beta, d_C3, N3));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_144, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 5. SGEMM and SSYRK for 160x160x160
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C4, 0, N4 * N4 * sizeof(float)));
    nvtxRangePushA("SGEMM_160");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N4, N4, K4,
                             &alpha, d_A4, N4, d_A4, N4,
                             &beta, d_C4, N4));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_160, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C4, 0, N4 * N4 * sizeof(float)));
    nvtxRangePushA("SSYRK_160");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N4, K4,
                             &alpha, d_A4, N4,
                             &beta, d_C4, N4));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_160, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 6. SGEMM and SSYRK for 256x256x256
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C5, 0, N5 * N5 * sizeof(float)));
    nvtxRangePushA("SGEMM_256");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N5, N5, K5,
                             &alpha, d_A5, N5, d_A5, N5,
                             &beta, d_C5, N5));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_256, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C5, 0, N5 * N5 * sizeof(float)));
    nvtxRangePushA("SSYRK_256");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N5, K5,
                             &alpha, d_A5, N5,
                             &beta, d_C5, N5));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_256, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 7. SGEMM and SSYRK for 240x240x240
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C6, 0, N6 * N6 * sizeof(float)));
    nvtxRangePushA("SGEMM_240");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N6, N6, K6,
                             &alpha, d_A6, N6, d_A6, N6,
                             &beta, d_C6, N6));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_240, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C6, 0, N6 * N6 * sizeof(float)));
    nvtxRangePushA("SSYRK_240");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N6, K6,
                             &alpha, d_A6, N6,
                             &beta, d_C6, N6));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_240, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // 8. SGEMM and SSYRK for 272x272x272
    // --------------------------------------------------
    CUDA_CHECK(cudaMemset(d_C7, 0, N7 * N7 * sizeof(float)));
    nvtxRangePushA("SGEMM_272");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N7, N7, K7,
                             &alpha, d_A7, N7, d_A7, N7,
                             &beta, d_C7, N7));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_sgemm_272, start, stop));
    nvtxRangePop();

    CUDA_CHECK(cudaMemset(d_C7, 0, N7 * N7 * sizeof(float)));
    nvtxRangePushA("SSYRK_272");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             N7, K7,
                             &alpha, d_A7, N7,
                             &beta, d_C7, N7));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_syrk_272, start, stop));
    nvtxRangePop();

    // --------------------------------------------------
    // Print all profiling results
    // --------------------------------------------------
    std::cout << "Profiling Results (ms):" << std::endl;
    std::cout << "2048x2048x2048 SGEMM: " << time_sgemm << " ms, SSYRK: " << time_syrk << " ms" << std::endl;
    std::cout << "128x128x128 SGEMM: " << time_sgemm_128 << " ms, SSYRK: " << time_syrk_128 << " ms" << std::endl;
    std::cout << "64x64x64 SGEMM: " << time_sgemm_64 << " ms, SSYRK: " << time_syrk_64 << " ms" << std::endl;
    std::cout << "144x144x144 SGEMM: " << time_sgemm_144 << " ms, SSYRK: " << time_syrk_144 << " ms" << std::endl;
    std::cout << "160x160x160 SGEMM: " << time_sgemm_160 << " ms, SSYRK: " << time_syrk_160 << " ms" << std::endl;
    std::cout << "256x256x256 SGEMM: " << time_sgemm_256 << " ms, SSYRK: " << time_syrk_256 << " ms" << std::endl;
    std::cout << "240x240x240 SGEMM: " << time_sgemm_240 << " ms, SSYRK: " << time_syrk_240 << " ms" << std::endl;
    std::cout << "272x272x272 SGEMM: " << time_sgemm_272 << " ms, SSYRK: " << time_syrk_272 << " ms" << std::endl;

    // --------------------------------------------------
    // Clean up host memory
    // --------------------------------------------------
    delete[] h_A;      delete[] h_C_sgemm;    delete[] h_C_syrk;
    delete[] h_A1;     delete[] h_C1_sgemm;   delete[] h_C1_syrk;
    delete[] h_A2;     delete[] h_C2_sgemm;   delete[] h_C2_syrk;
    delete[] h_A3;     delete[] h_C3_sgemm;   delete[] h_C3_syrk;
    delete[] h_A4;     delete[] h_C4_sgemm;   delete[] h_C4_syrk;
    delete[] h_A5;     delete[] h_C5_sgemm;   delete[] h_C5_syrk;
    delete[] h_A6;     delete[] h_C6_sgemm;   delete[] h_C6_syrk;
    delete[] h_A7;     delete[] h_C7_sgemm;   delete[] h_C7_syrk;

    // --------------------------------------------------
    // Clean up device memory
    // --------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));  CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A1)); CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_A2)); CUDA_CHECK(cudaFree(d_C2));
    CUDA_CHECK(cudaFree(d_A3)); CUDA_CHECK(cudaFree(d_C3));
    CUDA_CHECK(cudaFree(d_A4)); CUDA_CHECK(cudaFree(d_C4));
    CUDA_CHECK(cudaFree(d_A5)); CUDA_CHECK(cudaFree(d_C5));
    CUDA_CHECK(cudaFree(d_A6)); CUDA_CHECK(cudaFree(d_C6));
    CUDA_CHECK(cudaFree(d_A7)); CUDA_CHECK(cudaFree(d_C7));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
