#include "kernels_macros.cuh"
#include "matgen.cuh"
#include "micro_chol.hpp"
#include <fstream>
#include <iostream>

#include <limits>

//#include "sChol.cuh"
//#include "hChol.cuh"
//#include "fp8Chol.cuh"
#include "switching_chol.cuh"



#define MAX_THREADS 1024
#define MAX_NUM_BLOCKS(N) ((N + MAX_THREADS - 1) / MAX_THREADS)

int selectBestGPU() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if (num_devices == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }

    size_t max_free_mem = 0;
    int best_device = 0;

    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "GPU " << i << ": Free = " 
                  << free_mem / (1024 * 1024) << " MB, Total = " 
                  << total_mem / (1024 * 1024) << " MB\n";

        if (free_mem > max_free_mem) {
            max_free_mem = free_mem;
            best_device = i;
        }
    }

    std::cout << "Selecting GPU " << best_device << " with max free memory: " 
              << max_free_mem / (1024 * 1024) << " MB\n";

    return best_device;
}



void host_isnan(float* A, int n)
{
    for(int i = 0; i < n*n; i++) {
        if(isnanf(A[i])){ std::cout << "A has a nan\n"; return;}

    }
}


// vanilla CG
int vanilla_CG(double* A, double* x, double* b, int n, double normA)
{

}


// Preconditioned Conjugate Gradient (CG) solver (simplified skeleton)
// This function demonstrates the allocation, conversion, and Cholesky preconditioning.
// The actual CG loop here is simplified and uses dummy updates.
//-----------------------------
// precond_CG function
//-----------------------------
int precond_CG(double* A, double* x,
               double* b, int n, int r, double normA, cublasHandle_t& cublasH, cusolverDnHandle_t& cusolverH,
                cudaStream_t& stream ,float eps_prime = 0.0f, float flr = 0.0f, bool perturb_diag = false, int microscal_size = 32) {


                double rel_tol = 0.0;
                int max_it = 200;
                //start with memory allocation -- first calculate total needed memory so that we makee only one syscall
size_t total_memory = 0;

    size_t bytes_LL_cpy     = n * n * sizeof(double);        // LL^T copy
    size_t bytes_A          = n * n * sizeof(float);         // For computing LL^T
    size_t bytes_diag_a     = n * sizeof(float);             // diag_a
    size_t bytes_updated    = n * sizeof(float);             // updated_diag
    size_t bytes_scales     = (2*n*n / microscal_size) * sizeof(uint8_t);  // scale factors
    size_t equilibration_arr = n * sizeof(int); // For equilibration
    size_t bytes_vec        = n * sizeof(double); 

    total_memory = bytes_LL_cpy + bytes_A + bytes_diag_a + bytes_updated + bytes_scales + equilibration_arr + 4*bytes_vec;

    // Allocate all at once
    void* base_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&base_ptr, total_memory));

    uint8_t* raw = static_cast<uint8_t*>(base_ptr);

    double* d_LL_cpy       = reinterpret_cast<double*>(raw);          raw += bytes_LL_cpy;
    float*  d_A      = reinterpret_cast<float* >(raw);          raw += bytes_A;
    float*  d_diag_a       = reinterpret_cast<float* >(raw);          raw += bytes_diag_a;
    float*  d_updated_diag = reinterpret_cast<float* >(raw);          raw += bytes_updated;
    uint8_t* d_scales      = reinterpret_cast<uint8_t*>(raw);         raw += bytes_scales;
    double* d_r            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_p            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_z            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_Ap           = reinterpret_cast<double*>(raw);          /*done*/


  
    cublasLtHandle_t cublasLtH;
    CUBLAS_CHECK(cublasLtCreate(&cublasLtH));


    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    // copy A to bytes_A, while rounding to single prec
    convertDoubleToFloat<<<MAX_NUM_BLOCKS(n*n), MAX_THREADS, 0>>>(A, d_A, n * n);

    //equilibrate A if needed here


    //extract diagonal of A to d_diag_a
    extractDiagonal<<<MAX_NUM_BLOCKS(n), MAX_THREADS, 0>>>(d_A, d_diag_a, n, n);

    //prepare updated_diag
    CUDA_CHECK(cudaMemcpy(d_updated_diag, d_diag_a, bytes_diag_a, cudaMemcpyDeviceToDevice));

    // you can now call switching_cholesky
    switching_chol(d_A, n, r, d_diag_a, d_updated_diag, eps_prime, perturb_diag, microscal_size,
                   cusolverH, cublasLtH ,cublasH, stream, d_scales);
    
    // Copy LL^T to d_LL_cpy
    // This assumes that switching_chol has filled d_A with LL^T.
    
    int tmp = (n * n + 1024 - 1) / 1024;
    convertFloattoDouble<<<tmp, 1024, 0>>>(d_A, d_LL_cpy, n * n);

    //now guess initial solution to Ax = b. Use potrs

    CUDA_CHECK(cudaMemcpyAsync(x, b, n*sizeof(double), cudaMemcpyDeviceToDevice, stream));
    int info;
    CUSOLVER_CHECK(
        cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, n, 1, d_LL_cpy, n, x, n, &info));
    if (info) return info;          // preconditioner failed

    const double d_negone = -1.0, d_one = 1.0, d_zero = 0.0;

    CUBLAS_CHECK(
        cublasDgemv(cublasH, CUBLAS_OP_N,
                    n, n, &d_negone,
                    A, n,            // << use original double-precision A
                    x, 1,
                    &d_zero,
                    d_r, 1));        // d_r = −A x

    CUBLAS_CHECK(cublasDaxpy(cublasH, n, &d_one, b, 1, d_r, 1));   // r += b

    // ───────────── 5. z₀ = M⁻¹ r₀ ;  p₀ = z₀ ───────────────────────
    CUDA_CHECK(cudaMemcpyAsync(d_z, d_r, n*sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUSOLVER_CHECK(
        cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER,
                         n, 1, d_LL_cpy, n, d_z, n, &info));
    if (info) return info;

    CUBLAS_CHECK(cublasDcopy(cublasH, n, d_z, 1, d_p, 1));

    double rho, rho_old, resid, normb;
    CUBLAS_CHECK(cublasDdot(cublasH, n, d_r, 1, d_z, 1, &rho));
    CUBLAS_CHECK(cublasDnrm2(cublasH, n, b, 1, &normb));
    resid = sqrt(rho) / normb;

    // ───────────── 6. PCG loop (all double) ────────────────────────
    int k = 0;
    while (resid > rel_tol && k < max_it)
    {
        ++k;

        // Ap = A p
        CUBLAS_CHECK(
            cublasDgemv(cublasH, CUBLAS_OP_N,
                        n, n, &d_one,
                        A, n,
                        d_p, 1,
                        &d_zero,
                        d_Ap, 1));

        double alpha_den;
        CUBLAS_CHECK(cublasDdot(cublasH, n, d_p, 1, d_Ap, 1, &alpha_den));
        double alpha = rho / alpha_den;

        CUBLAS_CHECK(cublasDaxpy(cublasH, n, &alpha, d_p , 1, x , 1));  // x ← x+αp
        double neg_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(cublasH, n, &neg_alpha, d_Ap, 1, d_r, 1)); // r ← r-αAp

        // z = M⁻¹ r
        CUDA_CHECK(cudaMemcpyAsync(d_z, d_r, n*sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
        CUSOLVER_CHECK(
            cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER,
                             n, 1, d_LL_cpy, n, d_z, n, &info));
        if (info) return info;

        rho_old = rho;
        CUBLAS_CHECK(cublasDdot(cublasH, n, d_r, 1, d_z, 1, &rho));

        double beta = rho / rho_old;
        CUBLAS_CHECK(cublasDscal(cublasH, n, &beta, d_p, 1));      // p = βp
        CUBLAS_CHECK(cublasDaxpy(cublasH, n, &d_one, d_z, 1, d_p, 1)); // p = z+βp

        resid = sqrt(rho) / normb;
    }

    // ───────────── done – x already holds solution ─────────────────
    cudaFree(base_ptr);
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);

    return (resid <= rel_tol) ? 0 : -1;   // 0 = converged


    
    } // End CG loop.


    int vanilla_CG(double* d_A, double* d_x, double* d_b, int n, double tol = 1e-6, int max_iter = 1000) {
        cublasHandle_t handle;
        cublasCreate(&handle);
    
        // Allocate GPU memory
        double *d_r, *d_p, *d_Ap;
        cudaMalloc((void**)&d_r, n * sizeof(double));
        cudaMalloc((void**)&d_p, n * sizeof(double));
        cudaMalloc((void**)&d_Ap, n * sizeof(double));
    
        double alpha, beta, r_dot, r_dot_new, pAp;
        double neg_one = -1.0, one = 1.0, zero = 0.0;
    
        // Initialize r = b - Ax
        cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, d_x, 1, &zero, d_r, 1);
        cublasDaxpy(handle, n, &neg_one, d_r, 1, d_b, 1);  // r = b - Ax
        cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice); // p = r
    
        // Compute initial residual norm r_dot = r^T r
        cublasDdot(handle, n, d_r, 1, d_r, 1, &r_dot);
        double b_norm;
        cublasDdot(handle, n, d_b, 1, d_b, 1, &b_norm);
        b_norm = sqrt(b_norm);
    
        int k = 0;
        while (sqrt(r_dot) / b_norm > tol && k < max_iter) {
            // Ap = A * p
            cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, d_p, 1, &zero, d_Ap, 1);
            
            // Compute alpha = r^T r / (p^T A p)
            cublasDdot(handle, n, d_p, 1, d_Ap, 1, &pAp);
            alpha = r_dot / pAp;
    
            // x = x + alpha * p
            cublasDaxpy(handle, n, &alpha, d_p, 1, d_x, 1);
    
            // r = r - alpha * Ap
            double negalpha = -alpha;
            cublasDaxpy(handle, n, &(negalpha), d_Ap, 1, d_r, 1);
    
            // Compute new r_dot = r^T r
            cublasDdot(handle, n, d_r, 1, d_r, 1, &r_dot_new);
    
            // Compute beta = (new r^T r) / (old r^T r)
            beta = r_dot_new / r_dot;
    
            // p = r + beta * p
            cublasDscal(handle, n, &beta, d_p, 1);
            cublasDaxpy(handle, n, &one, d_r, 1, d_p, 1);
    
            // Update r_dot for next iteration
            r_dot = r_dot_new;
            k++;
        }
    
        // Cleanup
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_Ap);
        cublasDestroy(handle);
    
        return k; // Return number of iterations
    }

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
int main(int argc, char* argv[]) {
        // Create cuSOLVER / cuBLAS handles and stream

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "CUDA devices available: " << deviceCount << std::endl;
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found! Exiting...\n";
            exit(EXIT_FAILURE);
        }

          // Initialize cuBLAS and cuSOLVER handles
            cublasHandle_t cublasH;
            cusolverDnHandle_t cusolverH;
            CUBLAS_CHECK(cublasCreate(&cublasH));
            CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
        int N = 128; //change N to be some multiple of 128 below 40K based on GPU id
        int r = 64; //this stays the same across GPUs
        double* d_A = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_A, N*N));
        double condition_num = 100.0; //pick condiiton num to be anwhere between 10 and 10^6 based on gpu_id (make sure there are good number of samples for any given <N, cond> combo)
        int gpu_id = -1;
        CUDA_CHECK(cudaGetDevice(&gpu_id));
        DistType sigma_distr = DistType::RandomLogUniform;
        switch(gpu_id%4){
            case 0:
            sigma_distr = DistType::RandomLogUniform;
            break;
            case 1:
            sigma_distr = DistType::Clustered;
            break;
            case 2:
            sigma_distr = DistType::Arithmetic;
            break;
            case 3:
            sigma_distr = DistType::Geometric;
        } 
        generatePSD(d_A, N, condition_num, sigma_distr, cublasH, cusolverH, stream);

        double* b = nullptr;
        double* x = nullptr;
        double* inf_norm_A = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&b, N));
        CUDA_CHECK(cudaMalloc((void**)&x, N));

        //compute inf norm of d_A
        int num_threads = 256;
        int num_blocks = (N + 255)/256;
        infNormKernel<<<num_blocks, num_threads, 0>>>(d_A, b, N, N);

        maxReduceKernel<<<>>>
        float eps_primes[7] = {0.1f, 0.05f, 0.01f, 0.005f, 0.001f, 0.0005f, 0.0001f};    //one each GPU (so for each condition number, N combo, test )
        float timesA[7];
        std::fill_n(timesA, 7, std::numeric_limits<float>::infinity());
        float timesB[7];
        std::fill_n(timesB, 7, std::numeric_limits<float>::infinity());

        int success = 0;
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float bestTimeA = std::numeric_limits<float>::infinity();
        float bestEpsA = 0.0f;
        float bestTimeB = std::numeric_limits<float>::infinity();
        float bestEpsB = 0.0f;

        int i = 0;
        for (auto eps_prime : eps_primes) {
            cudaEventRecord(start, nullptr);
            cudaDeviceSynchronize();
            
            success = precond_CG(d_A, x, b, N, r, *inf_norm_A, cublasH, cusolverH,
                                stream, eps_prime, 0.0f, true);

            cudaDeviceSynchronize();
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);

            if (success == 0) {
                timesA[i] = elapsedTime;
                if (elapsedTime < bestTimeA) {
                    bestTimeA = elapsedTime;
                    bestEpsA = eps_prime;
                }
            }
            ++i;
        }

        // Second loop
        i = 0;
        for (auto eps_prime : eps_primes) {
            cudaEventRecord(start, nullptr);
            cudaDeviceSynchronize();
            
            success = precond_CG(d_A, x, b, N, r, *inf_norm_A, cublasH, cusolverH,
                                stream, eps_prime, 0.0f, false);

            cudaDeviceSynchronize();
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);

            if (success == 0) {
                timesB[i] = elapsedTime;
                if (elapsedTime < bestTimeB) {
                    bestTimeB = elapsedTime;
                    bestEpsB = eps_prime;
                        }
                    }
                    ++i;
                }

                // Write results to per-GPU file
                int deviceId = 0;
                cudaGetDevice(&deviceId);

                std::ofstream out("timing_gpu_" + std::to_string(deviceId) + ".txt", std::ios::app);
                if (out.is_open()) {
                    out << "GPU " << deviceId << ":\n";
                    out << "  Best epsilon (precond=true):  " << bestEpsA << ", time: " << bestTimeA << " ms\n";
                    out << "  Best epsilon (precond=false): " << bestEpsB << ", time: " << bestTimeB << " ms\n";
                    out.close();
                }

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                cudaFree(d_A);
                cudaFree(b);
                cudaFree(x);
                





        
        
    return 0;
}