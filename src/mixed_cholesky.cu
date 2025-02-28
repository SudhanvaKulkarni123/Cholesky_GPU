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


using namespace std;

enum distr_type : uint8_t {
    Arith, Geom, Rand
};

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
               double* b, int n, int r, double normA, float eps_prime = 0.0f, float flr = 0.0f) {


    
    std::ofstream myfile("e5m2_error_f_cond.csv");
    cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));  // Still creating a stream for kernels/BLAS if we like
    cudaStream_t logging_stream;
    CUDA_CHECK(cudaStreamCreate(&logging_stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    cusolverDnHandle_t CuHandle_t;
    CUSOLVER_CHECK(cusolverDnCreate(&CuHandle_t));
    CUSOLVER_CHECK(cusolverDnSetStream(CuHandle_t, stream));

    


    double One = 1.0;
    float Onef = 1.0f;
    double Zero = 0.0;
    double NegOne = -1.0;

    
    cudaEvent_t init_start, init_stop;
     CUDA_CHECK(cudaEventCreate(&init_start));
    CUDA_CHECK(cudaEventCreate(&init_stop));
    CUDA_CHECK(cudaEventRecord(init_start, stream)); 
// Compute total memory required
size_t size_A_d = n * n * sizeof(double);
size_t size_A_f = n * n * sizeof(float);
size_t max_buff_size = n * r * sizeof(float);  // Matul buffer size
size_t vec_size = n * sizeof(double);
size_t half_vec_size = n * sizeof(float);  // For float vectors
size_t scal_size = sizeof(double);
size_t perm_size = n * sizeof(int);



// Total size to allocate in a single call
size_t workspace_size = n * n * sizeof(float);  // New workspace buffer

size_t total_size = 
    size_A_d +  // d_A
    size_A_f +  // s_A
    workspace_size + // New workspace buffer
    size_A_d +  // LL_copy
    7 * vec_size +  // dev_x, dev_b, dev_r, p, Ap, dev_z, dev_D (all double)
    2 * half_vec_size +  // diag_A, updated_diag (float)
    4 * scal_size +  // pAp, rz, rz_new, inf_norm_r (double)
    3 * scal_size +  // d_One, d_Zero, d_NegOne (double)
    2 * perm_size + sizeof(int);  // dev_left_perm, dev_right_perm (int)

// Allocate memory
void* d_mem = nullptr;
CUDA_CHECK(cudaMalloc(&d_mem, total_size));

char* d_mem_char = reinterpret_cast<char*>(d_mem);

double* d_A        = reinterpret_cast<double*>(d_mem_char);
double* LL_copy    = reinterpret_cast<double*>(d_mem_char + size_A_d);  // LL_copy right after d_A
float* s_A         = reinterpret_cast<float*>(d_mem_char + size_A_d + size_A_d);
float* workspace   = reinterpret_cast<float*>(d_mem_char + size_A_d + size_A_d + size_A_f);  // Workspace right after s_A

double* dev_x      = reinterpret_cast<double*>(d_mem_char + size_A_d + size_A_d + size_A_f + workspace_size);
double* dev_b      = dev_x + n;
double* dev_r      = dev_b + n;
double* p          = dev_r + n;
double* Ap         = p + n;
double* dev_z      = Ap + n;
double* dev_D      = dev_z + n;

float* diag_A      = reinterpret_cast<float*>(dev_D + n);  
float* updated_diag= diag_A + n;

double* pAp        = reinterpret_cast<double*>(updated_diag + n);
double* rz         = pAp + 1;
double* rz_new     = rz + 1;
double* inf_norm_r = rz_new + 1;

double* d_One      = inf_norm_r + 1;
double* d_Zero     = d_One + 1;
double* d_NegOne   = d_Zero + 1;

int* dev_left_perm  = reinterpret_cast<int*>(d_NegOne + 1);
int* dev_right_perm = dev_left_perm + n;
int* devInfo = dev_right_perm + n;


// Copy scalar values asynchronously
CUDA_CHECK(cudaMemcpyAsync(d_One,    &One,    scal_size, cudaMemcpyHostToDevice, stream));
CUDA_CHECK(cudaMemcpyAsync(d_Zero,   &Zero,   scal_size, cudaMemcpyHostToDevice, stream));
CUDA_CHECK(cudaMemcpyAsync(d_NegOne, &NegOne, scal_size, cudaMemcpyHostToDevice, stream));

// Allocate host memory for preconditioner
double* D_host       = (double*)malloc(n * sizeof(double));
int* left_perm_host  = (int*)malloc(n * sizeof(int));
int* right_perm_host = (int*)malloc(n * sizeof(int));

// Initialize preconditioner data
for (int i = 0; i < n; i++) {
    left_perm_host[i] = i;
    right_perm_host[i] = i;
    D_host[i] = 1.0;  // Example
}

// Copy permutation and preconditioner data asynchronously
CUDA_CHECK(cudaMemcpyAsync(dev_left_perm, left_perm_host, perm_size, cudaMemcpyHostToDevice, stream));
CUDA_CHECK(cudaMemcpyAsync(dev_right_perm, right_perm_host, perm_size, cudaMemcpyHostToDevice, stream));
CUDA_CHECK(cudaMemcpyAsync(dev_D, D_host, vec_size, cudaMemcpyHostToDevice, stream));

    // Copy matrix A and vector b from host to device (synchronous).
    CUDA_CHECK(cudaMemcpyAsync(d_A, A, size_A_d, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(dev_b, b, vec_size, cudaMemcpyHostToDevice, stream));

    // Initialize search direction p with b (synchronous).
    CUDA_CHECK(cudaMemcpyAsync(p, b, vec_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Convert double-precision matrix to single precision (kernel uses stream).
    int totalElems = n * n;
    int threads = 256;
    int blocks = (totalElems + threads - 1) / threads;
    convertDoubleToFloat<<<blocks, threads, 0, stream>>>(d_A, s_A, totalElems);
    CUDA_CHECK(cudaStreamSynchronize(stream));


    blocks = (n + threads - 1)/threads;
    copy_diag<<<blocks, threads, 0, stream>>>(diag_A, updated_diag, s_A, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(init_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(init_stop));
    float time_init = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_init, init_start, init_stop));
    printf("initializartion took %.2f ms.\n", time_init);
    // cudaEventRecord(stop);
    // CUDA_CHECK(cudaEventSynchronize(stop));
    // float time_init = 0.0f;
    // CUDA_CHECK(cudaEventElapsedTime(&time_init, start, stop));
    // printf("initialization took %.2f ms.\n", time_init);



    // Perform mixed-precision Cholesky factorization on s_A (with your function).



    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    //uniform_prec_GPU_cholesky(s_A, n, r, diag_A, updated_diag, handle, CuHandle_t ,stream, n);
    //uniform_prec_fused_cholesky(s_A, n, r, diag_A, updated_diag, handle, CuHandle_t ,stream, n);
    //uniform_prec_GPU_cholesky(s_A, n, r, diag_A, updated_diag, handle, CuHandle_t ,stream, n);
    //uniform_precision_Cholesky(s_A, n, r, diag_A, updated_diag, handle, stream, n);
   // halfprec_mixed_precision_Cholesky(s_A, n, r, diag_A, updated_diag, handle, CuHandle_t, stream, n, reinterpret_cast<cutlass::half_t*>(workspace));
    //fp8_mixed_precision_Cholesky(s_A, n, r, diag_A, updated_diag, handle, CuHandle_t, stream, n, reinterpret_cast<cutlass::float_e4m3_t*>(workspace), flr);

    switching_precision_Cholesky(s_A, n, r, workspace, diag_A, updated_diag, handle, CuHandle_t, stream, n, eps_prime, flr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaEventRecord(stop, stream));
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
    float time_init_soln;
    cudaEventRecord(start, stream);
    CUSOLVER_CHECK(cusolverDnDpotrs(CuHandle_t, CUBLAS_FILL_MODE_LOWER, n, 1, LL_copy, n, dev_z, n, devInfo));

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init_soln, start, stop);


    // Set dev_x = p (device to device copy - synchronous).
    CUDA_CHECK(cudaMemcpy(dev_x, p, vec_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // r = b - A*x
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, n, n, d_NegOne,
                             d_A, n, dev_x, 1, d_One, dev_b, 1));           //doesnt matter if its row or col major
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


    float time_init_precond;
    // Compute r^T z
    cudaEventRecord(start, stream);
    CUDA_CHECK(cudaMemcpy(dev_z, dev_r, vec_size, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnDpotrs(CuHandle_t, CUBLAS_FILL_MODE_LOWER, n, 1, LL_copy, n, dev_z, n, devInfo));


    CUBLAS_CHECK(cublasDdot(handle, n, dev_z, 1, dev_r, 1, rz));

    CUDA_CHECK(cudaMemcpy(p, dev_z, vec_size, cudaMemcpyDeviceToDevice));
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init_precond, start, stop);

    const double d_conv_bound = 1e-14;
    int count = 0;

    float gemv_CG_time = 0;
    float trsv_CG_time = 0;

    float update_search_time = 0;

    // CG loop
    const int max_CG_iter = 1000;
    for (int j = 0; j < max_CG_iter; j++) {
        // Ap = A * p
        cudaEventRecord(start, stream);
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, n, n,
                                 d_One, d_A, n, p, 1, d_Zero, Ap, 1));          //doesnt matter if its row or col major
        
        CUBLAS_CHECK(cublasDdot(handle, n, Ap, 1, p, 1, pAp));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        gemv_CG_time += elapsed_time;



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

        cudaEventRecord(start, stream);
        CUSOLVER_CHECK(cusolverDnDpotrs(CuHandle_t, CUBLAS_FILL_MODE_LOWER, n, 1, LL_copy, n, dev_z, n, devInfo));
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_time2;
        cudaEventElapsedTime(&elapsed_time2, start, stop);
        trsv_CG_time += elapsed_time2;

        // Scale dev_z by D again
        // diag_scal<<<vecBlocks, vecThreads, 0, stream>>>(dev_z, dev_z, dev_D, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Compute r^T z
        CUBLAS_CHECK(cublasDdot(handle, n, dev_r, 1, dev_z, 1, rz_new));

        // p = z + beta * p
        cudaEventRecord(start, stream);
        update_search_dir<<<vecBlocks, vecThreads, sizeof(double), stream>>>(p, dev_z, rz_new, rz, n);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_time3;
        cudaEventElapsedTime(&elapsed_time3, start, stop);
        update_search_time += elapsed_time3;

        CUDA_CHECK(cudaMemcpy(rz, rz_new, sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        count++;
    }


    printf("GEMVs in CG took %.2f ms.\n", gemv_CG_time);
    printf("trsv in CG took %.2f ms.\n", trsv_CG_time);
    printf("search dir update in CG took %.2f ms.\n", update_search_time);




    float epilogue_time;
    // Copy final solution to host
    CUDA_CHECK(cudaMemcpy(x, dev_x, vec_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
   CUDA_CHECK(cudaFree(d_mem));

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
        
        // Try selecting device 0 explicitly
        int device = 0;
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device 0: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        
        cudaDeviceSynchronize();
        std::cout << "Using CUDA device: " << device << std::endl;
        
        
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
    int cg_iters = precond_CG(A, x_our, b, n, r, inf_norm, eps_prime, flr);
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
    // Run MKL's solver.
    printf("Running MKL solver...\n");
    CUDA_CHECK(cudaEventRecord(start));
    solveWithMKL(A, x_cusolver, b, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_mkl = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_mkl, start, stop));
    printf("MKL completed in %.2f ms.\n", time_mkl);


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

    
    return 0;
}