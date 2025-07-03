#include "kernels_macros.cuh"
#include "matgen.cuh"
#include "micro_chol.hpp"
#include <fstream>
#include <iostream>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <json.hpp>
//#include "sChol.cuh"
//#include "hChol.cuh"
//#include "fp8Chol.cuh"
#include "switching_chol.cuh"
#include <mkl.h>


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
               double* b, int n, int r, double normA, float eps_prime = 0.0f, float flr = 0.0f, bool perturb_diag = false, int microscal_size = 32) {


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
    float*  d_A_float      = reinterpret_cast<float* >(raw);          raw += bytes_A_float;
    float*  d_diag_a       = reinterpret_cast<float* >(raw);          raw += bytes_diag_a;
    float*  d_updated_diag = reinterpret_cast<float* >(raw);          raw += bytes_updated;
    uint8_t* d_scales      = reinterpret_cast<uint8_t*>(raw);         raw += bytes_scales;
    int*    d_equil        = reinterpret_cast<int*   >(raw);          raw += bytes_equil;
    double* d_r            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_p            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_z            = reinterpret_cast<double*>(raw);          raw += bytes_vec;
    double* d_Ap           = reinterpret_cast<double*>(raw);          /*done*/


    // Initialize cuBLAS and cuSOLVER handles
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    // copy A to bytes_A, while rounding to single prec
    convertDoubleToFloat<<<MAX_NUM_BLOCKS(n*n), MAX_THREADS, 0>>>(A, d_A, n * n);

    //equilibrate A if needed here


    //extract diagonal of A to d_diag_a
    extractDiagonal<<<MAX_NUM_BLOCKS(n), MAX_THREADS, 0>>>(d_A, d_diag_a, n);

    //prepare updated_diag
    CUDA_CHECK(cudaMemcpy(d_updated_diag, d_diag_a, bytes_diag_a, cudaMemcpyDeviceToDevice));

    // you can now call switching_cholesky
    switching_chol(d_A, n, r, d_diag_a, d_updated_diag, d_scales, eps_prime, perturb_diag, microscal_size,
                   cusolverH, cublasH, stream, d_scales);
    
    // Copy LL^T to d_LL_cpy
    // This assumes that switching_chol has filled d_A with LL^T.
    convertFloatToDouble<<<MAX_NUM_BLOCKS(n*n), MAX_THREADS, 0>>>(d_A, d_LL_cpy, n * n);

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


// Solve A*x = b using Intel MKL's Cholesky factorization.
int solveWithMKL(double* A, double* x, const double* b, int n) {
    // Copy b to x since LAPACK overwrites b with the solution.
    memcpy(x, b, n * sizeof(double));

    // Perform Cholesky factorization (A = L * L^T)
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A, n);
    if (info != 0) {
        printf("MKL Cholesky factorization failed with info = %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Solve A*x = b using the factorized matrix
    info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', n, 1, A, n, x, n);
    if (info != 0) {
        printf("MKL Cholesky solve failed with info = %d\n", info);
        exit(EXIT_FAILURE);
    }

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
        
        int num_devices;
cudaGetDeviceCount(&num_devices);


        // Try selecting device 1 explicitly
        int best_gpu = selectBestGPU();
        if (best_gpu >= 0) {
            cudaSetDevice(best_gpu);  // Set the selected GPU
            std::cout << "Using GPU " << best_gpu << std::endl;
        }
        cudaDeviceSynchronize();

        
        
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;


    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreate(&stream));

    //set problem params and factorization stuff using settings.json
    std::ifstream settings_file("settings.json", std::ifstream::binary);
    nlohmann::json settings = nlohmann::json::parse(settings_file);

   
    
    auto mat_set = settings["matrix_settings"];
    auto fact_set = settings["factorization_settings"];
    string tmp;
    tmp = mat_set["n"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    int n = stoi(tmp);
    tmp = mat_set["condition_number"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    double condVal = stod(tmp);
    //TODO - add code for different distributions
    tmp = fact_set["block_size"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    int r = stoi(tmp);
    tmp = fact_set["eps_prime"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    float eps_prime = stof(tmp);
    tmp = fact_set["floor"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    float flr = stof(tmp);
    tmp = fact_set["diag_pert"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    bool perturb_diag = stoi(tmp) != 0;
    tmp = fact_set["left"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    bool left_looking = stoi(tmp) != 0;

    



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
    int cg_iters = precond_CG(A, x_our, b, n, r, inf_norm, eps_prime, flr, perturb_diag, left_looking);
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
    // // Run MKL's solver.
    // printf("Running MKL solver...\n");
    // CUDA_CHECK(cudaEventRecord(start));
    // solveWithMKL(A, x_cusolver, b, n);
    // CUDA_CHECK(cudaEventRecord(stop));
    // CUDA_CHECK(cudaEventSynchronize(stop));
    // float time_mkl = 0.0f;
    // CUDA_CHECK(cudaEventElapsedTime(&time_mkl, start, stop));
    // printf("MKL completed in %.2f ms.\n", time_mkl);


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

    CUDA_CHECK(cudaEventRecord(start));

    double *d_x_vanilla, *d_b;
    cudaMalloc((void**)&d_x_vanilla, n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    double* x_vanilla = (double*)malloc(n * sizeof(double)); // Fixed sizeof typo

    // Run Vanilla CG
    CUDA_CHECK(cudaEventRecord(start));
    int vanilla_cg_iters = vanilla_CG(dA, d_x_vanilla, d_b, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_vanilla_cg = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_vanilla_cg, start, stop));
    printf("Vanilla CG completed in %d iterations and took %.2f ms.\n", vanilla_cg_iters, time_vanilla_cg);

    // ----------------------------
    // Compute residual norm for Vanilla CG
    double r_vanilla_cg_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double Ai_x = 0.0;
        for (int j = 0; j < n; j++) {
            Ai_x += A[i + j * n] * x_vanilla[j];
        }
        double ri = b[i] - Ai_x;
        r_vanilla_cg_norm += ri * ri;
    }
    r_vanilla_cg_norm = sqrt(r_vanilla_cg_norm);

    // Print out the residual norm for Vanilla CG
    printf("Residual norm of Vanilla CG    : %e\n", r_vanilla_cg_norm);


    
    // ----------------------------
    // Compare the solutions (compute relative L2 norm difference).
    double diff_norm = 0.0, sol_norm = 0.0, x_max = 0.0, x_nrm = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x_our[i] - x_cusolver[i];
        diff_norm += diff * diff;
        sol_norm  += x_cusolver[i] * x_cusolver[i];
        x_nrm += x_cusolver[i]*x_cusolver[i];
        x_max = max(x_max, abs(x_cusolver[i]));
    }
    diff_norm = sqrt(diff_norm);
    sol_norm = sqrt(sol_norm);
    x_nrm = sqrt(x_nrm);
    printf("norm difference between our solver and cuSOLVER: %e\n", diff_norm/x_nrm );
    printf("inf norm of x is : %e\n", x_max);
    
    // Clean up CUDA events.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_x_vanilla);
    cudaFree(d_b);
    free(x_vanilla);

    
    return 0;
}