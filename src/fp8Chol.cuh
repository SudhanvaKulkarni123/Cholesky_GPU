#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>         // ✅ Include for thrust::sequence
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <cutlass/array.h>

// Functor to map submatrix indices
struct SubmatrixIndexFunctor {
    const float* matrix;
    int ld, rows, cols;

    __host__ __device__
    SubmatrixIndexFunctor(const float* matrix, int ld, int rows, int cols)
        : matrix(matrix), ld(ld), rows(rows), cols(cols) {}

    __host__ __device__
    float operator()(int idx) const {
        int row = idx / cols;  // Compute row within submatrix
        int col = idx % cols;  // Compute col within submatrix
        return fabsf(matrix[(row) + (col)*ld]);  // Access full matrix
    }
};

// Function to find max of submatrix
float find_submatrix_max(const float* d_matrix, int ld, int rows, int cols) {
    thrust::device_vector<int> indices(rows * cols);
    thrust::sequence(indices.begin(), indices.end());  // Create index range

    // Use thrust::transform to map indices to values, then reduce to find max
    float max_val = thrust::transform_reduce(
        indices.begin(), indices.end(),
        SubmatrixIndexFunctor(d_matrix, ld, rows, cols),
        0.0,  // Init value
        thrust::maximum<float>()  // Reduction operation
    );

    return max_val;
}

template <typename T, int N, float eps>
struct CustomEpilogue {
    T alpha, beta;

    CustomEpilogue(T alpha, T beta) : alpha(alpha), beta(beta) {}

    CUTLASS_HOST_DEVICE
    void operator()(T& out, const T& C, const T& D, int row, int col) const {
        // Compute alpha * C + beta * D
        out = alpha * C + beta * D;

        // Apply ReLU only on diagonal of D
        if (row == col) {
            out = max(out, T(eps));
        }
    }
};



// ///fp8 version of mixed Cholesky-
int fp8_mixed_precision_Cholesky(float* d_A, int ld, int r,
    float* diag_A, float* updated_diag,
    cublasHandle_t handle, cusolverDnHandle_t cusolverH, cudaStream_t stream,
    int n, cutlass::float_e4m3_t* d_A_sub_fp8, float eps = 0.0f) {


        int deviceCount;
cudaGetDeviceCount(&deviceCount);
std::cout << "CUDA devices available: " << deviceCount << std::endl;

        float one = 1.0f;
float negOne = -1.0f;


// Perturb the diagonal by eps * I




int devInfo_h = 0;
int* devInfo  = nullptr;
CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );


int vecThreads = 256;
int vecBlocks = (n * n + vecThreads - 1) / vecThreads;

float* max_L = nullptr;
CUDA_CHECK(cudaMallocManaged(&max_L, sizeof(float), cudaMemAttachGlobal));

float h_max_L;

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



using EpilogueOutputOp =  cutlass::epilogue::thread::LinearCombination<
float, 8>;

//declare GEMM functor-
using GemmFp8Fp32 = cutlass::gemm::device::Gemm<
cutlass::float_e4m3_t,           // ElementA (FP8)
cutlass::layout::RowMajor,       // LayoutA
cutlass::float_e4m3_t,           // ElementB (FP8)
cutlass::layout::ColumnMajor,       // LayoutA^T
float,                           // ElementC (output)
cutlass::layout::RowMajor,       // LayoutC
float,                           // ElementAccumulator (accumulation type: FP32)
cutlass::arch::OpClassTensorOp,  // Operator class (Tensor Ops)
cutlass::arch::Sm89,             // Architecture (Sm89 for FP8 support)
cutlass::gemm::GemmShape<128, 64, 128>,  // Threadblock shape
cutlass::gemm::GemmShape<64, 32, 128>,   // Warp shape
cutlass::gemm::GemmShape<16, 8, 32>,     // Instruction shape
EpilogueOutputOp,                      // **Custom No-Op Epilogue**
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock swizzle
4,    // Number of stages
16,   // Alignment A
16   // Alignment B
//cutlass::arch::OpMultiplyAddFastAccum<> // Operator
>;

GemmFp8Fp32 lo_gemm_op;

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


for (int k = 0; k < n; k += r) {
int r_block = (k + r < n) ? r : (n - k);
// Start timing A00 factorization
cudaEventRecord(start, stream);

size_t dstPitch = r_block * sizeof(float);
size_t srcPitch = ld * sizeof(float);
size_t widthInBytes = r_block * sizeof(float);

int sub = n - (k + r_block);

// CUDA_CHECK(cudaMemcpy2D(A_00, dstPitch,
//    d_A + k + k * ld, srcPitch, 
//    widthInBytes, r_block, 
// cudaMemcpyDeviceToHost));

// // CPU Cholesky factorization
// micro_cholesky(A_00, r_block, r_block);


// CUDA_CHECK(cudaMemcpy2D(d_A + k + k * ld, srcPitch,
//    A_00, dstPitch, 
//    widthInBytes, r_block, 
// cudaMemcpyHostToDevice));

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


// vanilla_Max_2D_col_major<<<vecBlocks, vecThreads, vecThreads * sizeof(float), stream>>>(d_A + k + r_block + k*ld, dev_blockMax, sub, r_block, ld);

// int finalThreads = 256;

// final_max_reduce<<<1, finalThreads, finalThreads * sizeof(float), stream>>>(dev_blockMax, max_L, vecBlocks);
*max_L = find_submatrix_max(d_A + k + r_block + k*ld, ld,  sub, r_block);


dim3 blockSize(16, 16);
dim3 gridSize((sub + blockSize.x - 1) / blockSize.x, (r_block + blockSize.y - 1) / blockSize.y);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
time_conversion += elapsed;

transposeScaleCastFp8<<<gridSize, blockSize, 0, stream>>>(
d_A + (k + r_block) + k * ld, max_L, 
d_A_sub_fp8,
sub, r_block, ld, ld);
cudaStreamSynchronize(stream);



//now compute GEMM with CUTLASS
// Now, define the GEMM operator using these tuning parameters.

// Set up arguments for the CUTLASS FP8 GEMM
//
float to_put = -1.0*(*max_L)*(*max_L);

typename GemmFp8Fp32::Arguments fp8_arguments{
{sub, sub, r_block},                    // GEMM problem size
{d_A_sub_fp8, ld},            // Tensor A: pointer and leading dimension M
{d_A_sub_fp8, ld},            // Tensor B: pointer and leading dimension K
{d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor C: pointer and leading dimension M (input/output)
{d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor D: pointer and leading dimension M (output)
{to_put, one}                 // Epilogue parameters
};


cudaEventRecord(start, stream);
cutlass::Status status = lo_gemm_op(fp8_arguments);
fixDiag<<<gridSize, blockSize, 0, stream>>>(
    d_A + k + k*ld, n, 
    eps);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
time_gemm += elapsed;

}
}

// Free resources
cudaFree(d_A_sub_fp8);
cudaFree(max_L);
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

//we only need to show the worst case, so we'll
int left_fp8(cutlass::float_e4m3_t* d_A, int ld, int r, float* scalings,
    float* diag_A, float* updated_diag,
    cublasHandle_t handle, cusolverDnHandle_t cusolverH, cudaStream_t stream,
    int n, float eps = 0.0f)
{

int devInfo_h = 0;
int* devInfo  = nullptr;
CUDA_CHECK( cudaMalloc((void**)&devInfo, sizeof(int)) );


int vecThreads = 64;
int vecBlocks = (n * n + vecThreads - 1) / vecThreads;

float* max_L = nullptr;
CUDA_CHECK(cudaMallocManaged(&max_L, sizeof(float), cudaMemAttachGlobal));

float h_max_L;

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



using EpilogueOutputOp =  cutlass::epilogue::thread::LinearCombination<
float, 8>;

//declare GEMM functor-
using GemmFp8Fp32 = cutlass::gemm::device::Gemm<
cutlass::float_e4m3_t,           // ElementA (FP8)
cutlass::layout::RowMajor,       // LayoutA
cutlass::float_e4m3_t,           // ElementB (FP8)
cutlass::layout::ColumnMajor,       // LayoutA^T
float,                           // ElementC (output)
cutlass::layout::RowMajor,       // LayoutC
float,                           // ElementAccumulator (accumulation type: FP32)
cutlass::arch::OpClassTensorOp,  // Operator class (Tensor Ops)
cutlass::arch::Sm89,             // Architecture (Sm89 for FP8 support)
cutlass::gemm::GemmShape<128, 64, 128>,  // Threadblock shape
cutlass::gemm::GemmShape<64, 32, 128>,   // Warp shape
cutlass::gemm::GemmShape<16, 8, 32>,     // Instruction shape
EpilogueOutputOp,                      // **Custom No-Op Epilogue**
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock swizzle
4,    // Number of stages
16,   // Alignment A
16   // Alignment B
//cutlass::arch::OpMultiplyAddFastAccum<> // Operator
>;

GemmFp8Fp32 lo_gemm_op;


// typename GemmFp8Fp32::Arguments fp8_arguments{
//     {sub, r_block, r_block},                    // GEMM problem size
//     {d_A_sub_fp8, ld},            // Tensor A: pointer and leading dimension M
//     {d_A_sub_fp8, ld},            // Tensor B: pointer and leading dimension K
//     {d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor C: pointer and leading dimension M (input/output)
//     {d_A + k + r_block + (k+r_block)*ld, ld},       // Tensor D: pointer and leading dimension M (output)
//     {to_put, one}                 // Epilogue parameters
//     };
    
    

    
}

