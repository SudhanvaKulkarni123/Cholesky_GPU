#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cublasLt.h>

__inline__ __device__ __nv_fp8_e8m0 encode_e8m0_scale(float scale) {
    int exp;
    std::frexp(scale, &exp);         // scale = mantissa * 2^exp
    int biased_exp = exp + 127;      // E8M0 bias
    biased_exp = max(0, min(255, biased_exp));
    return static_cast<__nv_fp8_e8m0>(biased_exp);
}


__inline__ __device__
float warpReduceMax(float val) {
    // full mask for warp
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}



// Convert FP32 → FP8 with a shared scaling factor, note -> CUDA does rowmajor, so I need to warp reduce on y
__global__ void convert_fp32_to_mxe4m3(
    const float* input, __nv_fp8_e4m3* output,
    int rows, int cols, __nv_fp8_e8m0* encoded_scale, int ld)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    //use tid_x for row and tid_y for col
    for(int i = tid_x; i < rows; i += blockDim.x*gridDim.x) {

        for(int j = tid_y; j < cols; j += blockDim.y*gridDim.y) {

            __nv_fp8_e8m0 exp;
            
            float val;
            //warp reeduction
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fabs(input[ld*i + j]);
                float neighbor = __shfl_down_sync(0xffffffff, val, offset);
                val = max(val, neighbor);
            }

            val = __shfl_sync(0xffffffff, val, 0);  //thread 0 in the warp has max exp
            exp = __nv_fp8_e8m0(val);

            //computes max over 32 elem!
            
            if((ld*i + j) % 32 == 0) {
                encoded_scale[(ld*i + j) / 32] = exp;
            }


            output[(ld*i + j)] = __nv_fp8_e4m3(input[ld*i + j] / (float)exp);
            }

        }
    

    return;
    
}







#define TILE_DIM 32
#define BLOCK_ROWS 8

// Convert FP32 → FP8 with a shared scaling factor

__global__ void convert_transpose_fp32_to_mxe4m3(
    const float* input, __nv_fp8_e4m3* output,
    int rows, int cols, __nv_fp8_e8m0* encoded_scale, int ld)
{

    /*
    ---------------
    |   A00    A01  |``access A(i,j) with tid_x and tid_y - set TILE_DIM = block_dim. In this case, block of thread also means panel
    |               |   But 1D grid, so the grid ordering is technically A00, A01, A10, A11
    |    A10   A11  |
    |               |
    ----------------
    */
    

    extern __shared__ float shared_data[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * ld;

    __nv_fp8_e8m0 exp;

    //algo is simple. Tile transpose and before write back use warp reduction to find the max exp. 
    //TBH don;t really need warp reduction since it it in shared data, but ist fine
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     shared_data[threadIdx.y+j][threadIdx.x] = input[(y+j)*width + x];

    __syncthreads();
    float buf;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
     for (int offset = 16; offset > 0; offset /= 2) {
        //warp reduction
        buf = fabs(shared_data[threadIdx.y+j][threadIdx.x]) ;
        buf = max(buf, 
            __shfl_down_sync(0xffffffff, buf, offset));
        }

        buf = __shfl_sync(0xffffffff, buf, 0);

        exp = __nv_fp8_e8m0(buf);
        if (((y+j)*width + x) % 32 == 0) {
            encoded_scale[((y+j)*width + x)/32] = exp;
        }
       output[(y+j)*width + x] = __nv_fp8_e4m3(
            shared_data[threadIdx.x][threadIdx.y + j] / (float)exp);
       }
     


    return;
    
}




//lauuncher for fp8 gemm
void LtMxfp8Matmul(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const __nv_fp8_e8m0 *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const __nv_fp8_e8m0 *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *beta, /* host pointer */
                 float *C,
                 int ldc,
                 float *D,
                 int ldd,
                 void *workspace,
                 size_t workspaceSize,
                 cublasLtMatmulMatrixScale_t AScaleMode,
                 cublasLtMatmulMatrixScale_t BScaleMode
                 ) {

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));


    // set block scaling mode
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode)));

    // set scaling factors
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, ldd));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     D,
                                     Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
}



void LtSgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const float *A,
             int lda,
             const float *B,
             int ldb,
             const float *beta, /* host pointer */
             float *C,
             int ldc,
             void *workspace,
             size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
}

void LtHSgemm_fp16in_fp32out(cublasLtHandle_t      ltHandle,
                             cublasOperation_t     transa,
                             cublasOperation_t     transb,
                             int                   m,
                             int                   n,
                             int                   k,
                             const float          *alpha,
                             const __half         *A,
                             int                   lda,
                             const __half         *B,
                             int                   ldb,
                             const float          *beta,
                             float                *C,
                             int                   ldc,
                             void                 *workspace,
                             size_t                workspaceSize)
{
    cublasLtMatmulDesc_t        opDesc  = nullptr;
    cublasLtMatrixLayout_t      Adesc   = nullptr,
                                Bdesc   = nullptr,
                                Cdesc   = nullptr;
    cublasLtMatmulPreference_t  pref    = nullptr;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heur = {};

    /* -------------------------------------------------- *
     * 1. Operation descriptor – accumulate in FP32       *
     * -------------------------------------------------- */
    CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&opDesc,
                                 CUBLAS_COMPUTE_32F,   // ⬅ Accumulation = FP32
                                 CUDA_R_32F));         // ⬅ Scale type   = FP32
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(opDesc,
                                       CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transa,
                                       sizeof(transa)));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(opDesc,
                                       CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transb,
                                       sizeof(transb)));


    CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&Adesc,
                                   CUDA_R_16F,          // ⬅ CHANGED: input in FP16
                                   (transa == CUBLAS_OP_N) ? m : k,
                                   (transa == CUBLAS_OP_N) ? k : m,
                                   lda));
    CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&Bdesc,
                                   CUDA_R_16F,          // ⬅ CHANGED: input in FP16
                                   (transb == CUBLAS_OP_N) ? k : n,
                                   (transb == CUBLAS_OP_N) ? n : k,
                                   ldb));
    CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&Cdesc,
                                   CUDA_R_32F,          // ⬅ output kept in FP32
                                   m,
                                   n,
                                   ldc));


    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)));

    CUBLAS_CHECK(
        cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                       opDesc,
                                       Adesc,
                                       Bdesc,
                                       Cdesc,
                                       Cdesc,
                                       pref,
                                       1,
                                       &heur,
                                       &returnedResults));

    if (returnedResults == 0) {
        CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }


    CUBLAS_CHECK(
        cublasLtMatmul(ltHandle,
                       opDesc,
                       alpha,
                       A,  Adesc,
                       B,  Bdesc,
                       beta,
                       C,  Cdesc,   // C (beta scaling)
                       C,  Cdesc,   // D (output)
                       &heur.algo,
                       workspace,
                       workspaceSize,
                       /* stream */ 0));

    if (pref)  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (opDesc)CUBLAS_CHECK(cublasLtMatmulDescDestroy(opDesc));
}