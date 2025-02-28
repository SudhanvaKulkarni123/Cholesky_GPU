int halfprec_mixed_precision_Cholesky(float* d_A, int ld, int r,
    float* diag_A, float* updated_diag,
    cublasHandle_t handle,  cusolverDnHandle_t cusolverH, cudaStream_t stream,
    int n, cutlass::half_t* d_A_sub_half) {
float one = 1.0f;
float negOne = -1.0f;

int devInfo_h = 0;
int* devInfo  = nullptr;
CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );

// float* A_00 = nullptr;
// cudaHostAlloc((void**)&A_00, sizeof(float) * r * r, cudaHostAllocDefault);
int vecThreads = 256;
int vecBlocks = (n * n + vecThreads - 1) / vecThreads;


float* dev_blockMax = nullptr;
CUDA_CHECK(cudaMalloc((void**)&dev_blockMax, vecBlocks * sizeof(float)));

// CUDA Events for Profiling
cudaEvent_t start, stop;
float time_factorize = 0, time_trsm = 0, time_conversion = 0, time_gemm = 0;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Variables for FLOP counts
double flops_factorize = 0, flops_trsm = 0, flops_gemm = 0;
float alpha = -1.0f;
float beta = 1.0f;
float eps = 0.03125f;


using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
float,  
8
>;

//declare GEMM functor-
using GemmFp16Fp32 = cutlass::gemm::device::Gemm<
cutlass::half_t,            // ElementA
cutlass::layout::ColumnMajor,  // LayoutA
cutlass::half_t,            // ElementB
cutlass::layout::RowMajor,  // LayoutB
float,                      // ElementC (accumulation)
cutlass::layout::ColumnMajor,  // LayoutC
float                       // Epilogue scalar type
>;
GemmFp16Fp32 lo_gemm_op;

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

// Allocate half-precision buffer for A_sub


for (int k = 0; k < n; k += r) {
int r_block = (k + r < n) ? r : (n - k);
// Start timing A00 factorization
cudaEventRecord(start, stream);

size_t dstPitch = r_block * sizeof(float);
size_t srcPitch = ld * sizeof(float);
size_t widthInBytes = r_block * sizeof(float);


// CUDA_CHECK(cudaMemcpy2D(A_00, dstPitch,
//    d_A + k + k * ld, srcPitch, 
//    widthInBytes, r_block, 
// cudaMemcpyDeviceToHost));

// // CPU Cholesky factorization
// micro_cholesky(A_00, r_block, r_block);

int sub = n - (k + r_block);

// CUDA_CHECK(cudaMemcpy2D(d_A + k + k * ld, srcPitch,
//    A_00, dstPitch, 
//    widthInBytes, r_block, 
// cudaMemcpyHostToDevice));

//factorizer on GPU 
float* d_panel = d_A + (k * ld + k);
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

// Start timing conversion
cudaEventRecord(start, stream);


//  vanilla_Max_2D_col_major<<<vecBlocks, vecThreads, vecThreads * sizeof(double), stream>>>(
//     d_A + (k + r_block) + k * ld, dev_blockMax, r_block, sub, n);
// int finalThreads = 256;
// final_max_reduce<<<1, finalThreads, finalThreads * sizeof(double), stream>>>(
dim3 blockSize(16, 16);
dim3 gridSize((sub + blockSize.x - 1) / blockSize.x, (r_block + blockSize.y - 1) / blockSize.y);
convertFloatToHalf<<<gridSize, blockSize, 0, stream>>>(
d_A + (k + r_block) + k * ld, 
d_A_sub_half,
sub, r_block, ld, ld);



cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
time_conversion += elapsed;

//now compute GEMM with CUTLASS
// Now, define the GEMM operator using these tuning parameters.

// Set up arguments for the CUTLASS FP8 GEMM
typename GemmFp16Fp32::Arguments fp16_arguments{
{sub, sub, r_block},                    // GEMM problem size
{d_A_sub_half, ld},            // Tensor A: pointer and leading dimension M
{d_A_sub_half, ld},            // Tensor B: pointer and leading dimension K
{d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor C: pointer and leading dimension M (input/output)
{d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor D: pointer and leading dimension M (output)
{negOne, one}                 // Epilogue parameters
};





cudaEventRecord(start, stream);
cutlass::Status status = lo_gemm_op(fp16_arguments, stream);


cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
time_gemm += elapsed;


}
}

// Free resources

cudaFree(d_A_sub_half);
cudaFree(dev_blockMax);


cudaEventDestroy(start);
cudaEventDestroy(stop);

// Print profiling results
std::cout << "Profiling Results (ms) & FLOPs:\n";
std::cout << "Factorization (A00): " << time_factorize << " ms, "
<< "FLOPs: " << flops_factorize / 1e9 << " GFLOPs\n";
std::cout << "TRSM: " << time_trsm << " ms, "
<< "FLOPs: " << flops_trsm / 1e9 << " GFLOPs\n";
std::cout << "Conversion (Float -> Half): " << time_conversion << " ms\n";
std::cout << "GEMM: " << time_gemm << " ms, "
<< "FLOPs: " << flops_gemm / 1e9 << " GFLOPs\n";

return 0;
}

int left_fp16_cholesky(float* d_A, int ld, int r,
float* diag_A, float* updated_diag,
cublasHandle_t handle, cudaStream_t stream, int n)
{
float one = 1.0f;
float negOne = -1.0f;

cutlass::half_t* d_A_half;
cudaMalloc((void**)&d_A_half, sizeof(cutlass::half_t) * r * n);

float* A00;
cudaHostAlloc((void**)&A00, sizeof(float) * r * r, cudaHostAllocDefault);

for(int i = 0; i < n; i+= r)
{

}

return 0;

}

