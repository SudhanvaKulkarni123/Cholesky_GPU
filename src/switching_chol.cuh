#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>         // ✅ Include for thrust::sequence
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <cutlass/array.h>



#define MACH_EPS_E4M3 0.125f
#define MACH_EPS_FP16 0.0009765625f
#define MACH_EPS_FP32 1.1920929e-7f

enum class precisions : uint8_t {
    fp8 = 0,
    fp16 = 1,
    bf16 = 2,
    fp32 = 3
};

std::ofstream gemms_file("gemmers.csv");
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
int switching_precision_Cholesky(float* d_A, int ld, int r, float* d_A_sub,
    float* diag_A, float* updated_diag,
    cublasHandle_t handle, cusolverDnHandle_t cusolverH, cudaStream_t stream,
    int n, float eps_prime = 0.0f, float eps = 0.0f, bool modify_diag = false) {



float one = 1.0f;
float negOne = -1.0f;


float* debug_arr = (float*) malloc(10*sizeof(float));


// Perturb the diagonal by eps * I

auto d_A_sub_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(d_A_sub);
auto d_A_sub_half = reinterpret_cast<cutlass::half_t*>(d_A_sub);

int devInfo_h = 0;
int* devInfo  = nullptr;
CUDA_CHECK(cudaMallocManaged((void**)&devInfo, sizeof(int) + sizeof(int)));
int* just_switch = (int*)(devInfo + 1);
*just_switch = 1;


int vecThreads = 256;
int vecBlocks = (n * n + vecThreads - 1) / vecThreads;
int one_threads = (n + vecThreads - 1) / vecThreads;

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
long long flops_factorize = 0, flops_trsm = 0, flops_gemm = 0;
float alpha = -1.0f;
float beta = 1.0f;



using EpilogueOutputOp =  cutlass::epilogue::thread::LinearCombination<
float, 8>;

//declare GEMM functors-
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

using GemmFp16Fp32 = cutlass::gemm::device::Gemm<
cutlass::half_t,            // ElementA
cutlass::layout::ColumnMajor,  // LayoutA
cutlass::half_t,            // ElementB
cutlass::layout::RowMajor,  // LayoutB
float,                      // ElementC (accumulation)
cutlass::layout::ColumnMajor,  // LayoutC
float                       // Epilogue scalar type
>;
GemmFp16Fp32 half_gemm_op;

using GemmFp32Fp32 = cutlass::gemm::device::Gemm<
float,            // ElementA
cutlass::layout::ColumnMajor,  // LayoutA
float,            // ElementB
cutlass::layout::RowMajor,  // LayoutB
float,                      // ElementC (accumulation)
cutlass::layout::ColumnMajor,  // LayoutC
float                       // Epilogue scalar type
>;

GemmFp32Fp32 gemm_op_32;

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

// fixDiag<<<r*r/256, 256, 0, stream>>>(
//     d_A + k + k*ld, r, eee
//     eps);

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
*just_switch = 1;
update_diag<<<n/one_threads + 1, one_threads, 0, stream>>>(diag_A + k + r, updated_diag + k + r, d_A + (k + r_block) + k * ld, n-k - r, r_block, n);



can_switch<<<n/one_threads + 1, one_threads, 0, stream>>>(diag_A + k + r, updated_diag + k + r, MACH_EPS_E4M3, eps_prime, sub - r, just_switch);




dim3 blockSize(16, 16);
dim3 gridSize((sub + blockSize.x - 1) / blockSize.x, (r_block + blockSize.y - 1) / blockSize.y);
cudaStreamSynchronize(stream);

flops_gemm += 2*sub*sub*r_block;
if(*just_switch != 0)
{

    gemms_file << "performing " << sub << " by " << r_block << "size SYRK in fp8\n"; 

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
    d_A + k + r + (k + r)*ld, n, 
    eps);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed, start, stop);
time_gemm += elapsed;


} else {
    *just_switch = 1;
    can_switch<<<n/one_threads + 1, one_threads, 0, stream>>>(diag_A + k + r, updated_diag + k + r, MACH_EPS_FP16, eps_prime, sub - r, just_switch);
    cudaStreamSynchronize(stream);

    if(*just_switch != 0) {

        gemms_file << "performing " << sub << " by " << r_block << "size SYRK in fp16\n"; 
        cutlass::half_t* d_A_sub_half = reinterpret_cast<cutlass::half_t*>(d_A_sub);
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
        cutlass::Status status = half_gemm_op(fp16_arguments, stream);
        fixDiag<<<gridSize, blockSize, 0, stream>>>(
            d_A + k + r + (k + r)*ld, n, 
            eps);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        time_gemm += elapsed;

    } else {
        // FP32 gemm fallback

        gemms_file << "performing " << sub << " by " << r_block << "size SYRK in fp32\n"; 
            cudaEventRecord(start, stream);
            GemmFp32Fp32 gemm_op_32;
            // **Row-major GEMM fix: Adjust layouts**
            typename GemmFp32Fp32::Arguments arguments_32{
                {sub, sub, r_block},                   // GEMM problem size
                { d_A + (k + r_block) + k * ld, int(ld)},          // Tensor A: pointer and leading dimension M
                { d_A + (k + r_block) + k * ld, int(ld)},          // Tensor B: pointer and leading dimension K
                {d_A + (k + r_block) + (k + r_block)*ld, int(ld)},        // Tensor C: pointer and leading dimension M (input/output)
                {d_A + (k + r_block) + (k + r_block)*ld, int(ld)},        // Tensor D: pointer and leading dimension M (output)
                {negOne, one}                // Epilogue parameters
            };

            gemm_op_32(arguments_32);
            fixDiag<<<gridSize, blockSize, 0, stream>>>(
                d_A + (k + r) + (k + r)*ld, n, 
                eps);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            time_gemm += elapsed;

    }
}

}
}


cudaFree(max_L);
cudaFree(dev_blockMax);
cudaFree(devInfo);


cudaEventDestroy(start);
cudaEventDestroy(stop);

// Print profiling results
std::cout << "Profiling Results (ms) & FLOPs:\n";
std::cout << "Factorization (A00): " << time_factorize << " ms, "
<< "FLOPs: " << (double)flops_factorize / 1e9 << " GFLOPs\n";
std::cout << "TRSM: " << time_trsm << " ms, "
<< "FLOPs: " << (double)flops_trsm / 1e9 << " GFLOPs\n";
std::cout << "Conversion (Float -> Half): " << time_conversion << " ms\n";
std::cout << "GEMM: " << time_gemm << " ms, "
<< "FLOPs: " << (double)flops_gemm / 1e9 << " GFLOPs\n";

return 0;
}

//we only need to show the worst case, so we'll store in fp8
int left_fp8(cutlass::float_e4m3_t* d_A, int ld, int r, float* d_A_sup, int8_t* scalings,
    float* diag_A, float* updated_diag,
    cublasHandle_t handle, cusolverDnHandle_t cusolverH, cudaStream_t stream,
    int n, float eps = 0.0f, int num_streams = 1)
{

        //declare GEMM functors-
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


    float one = 1.0f;
    float negOne = -1.0f;

    precisions* prec_list = (precisions*) malloc(sizeof(uint8_t) * ((int) n / r));
    float* debug_arr = (float*) malloc(10 * sizeof(float));

    int devInfo_h = 0;
    int* devInfo  = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&devInfo, sizeof(int) + sizeof(int)));
    int* just_switch = (int*)(devInfo + 1);
    *just_switch = 1;

    // Create GEMM streams for concurrent schur update
    cudaStream_t gemm_streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&gemm_streams[i]));
    }

    int lwork = 0;
    CUSOLVER_CHECK(
        cusolverDnSpotrf_bufferSize(
            cusolverH,
            CUBLAS_FILL_MODE_LOWER, 
            r, 
            d_A_sup, 
            ld,
            &lwork
        )
    );

    float* panel_workspace;
    CUDA_CHECK(cudaMalloc((void**)&panel_workspace, lwork * sizeof(float)));

    for (int k = 0; k < n; k += r) {
        // Convert FP8 to float for factorization
        dim3 vecThreads(32,32); //blocksize
        dim3 vecBlocks((n-k + 31)/32, (r + 31)/32) ; //gridSize 
        Fp8toFloat<<<vecBlocks, vecThreads, 0, stream>>>(d_A + k + k * ld, scalings + (k + k * ld) / r, d_A_sup, n - k, r, n, r);

        if (k != 0) {
            //first loop to go down the panel
            for (int j = k; j < n; j += r) {
                
                for(int i = 0; i < k; i += r) {

                        // Scaling factor
                float to_put = -1.0 * scalings[(j) / r] * scalings[(i) / r];

                typename GemmFp8Fp32::Arguments fp8_arguments{
                    {k, k, k},                              // GEMM problem size
                    {d_A + k*ld + j, ld},             // Tensor A
                    {d_A + k*ld + i, ld},             // Tensor B
                    {d_A_sup, ld},                          // Tensor C (input/output)
                    {d_A_sup, ld},             // Tensor D
                    {to_put, one}                           // Epilogue parameters
                };

                lo_gemm_op(fp8_arguments, gemm_streams[(int)j/r]);
                }
                
            }     
        }

        // Panel factorization
        float* d_panel = d_A_sup;
        CUSOLVER_CHECK(
            cusolverDnSpotrf(
                cusolverH,
                CUBLAS_FILL_MODE_LOWER, 
                r,
                d_panel,
                ld,
                panel_workspace,
                lwork,
                devInfo
            )
        );

        // TRSM before rounding down
        CUBLAS_CHECK( 
            cublasStrsm(
                handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                r, k - r, &one,
                d_A_sup,
                ld,
                d_A_sup + k + r,
                ld
            )
        );

        // Round down after TRSM
        FloatToMXFP8<<<vecBlocks, vecThreads, sizeof(float)*1024, stream>>>(d_A_sup, scalings + (int)((k + k*ld)/r) ,d_A + k*ld + k, k - r, r, ld, r);

    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(gemm_streams[i]));
    }
    CUDA_CHECK(cudaFree(panel_workspace));
    CUDA_CHECK(cudaFree(devInfo));
    free(debug_arr);
    free(prec_list);

    return 0;
}










