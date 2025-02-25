#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/rank_k_universal.h>
#include <cutlass/gemm/device/rank_k.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Error-checking macros
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

#define CUBLAS_CHECK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << status
                  << " " << file << " " << line << std::endl;
        exit(status);
    }
}

// Utility: Fill an array with random floats
void randomFill(float* arr, int numElements) {
    for (int i = 0; i < numElements; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Matrix dimensions
const int N = 2048, K = 512;
const float alpha = 1.0f, beta = 0.0f;
const int numElementsA = N * K;
const int numElementsC = N * N;


cudaError_t CutlassSsyrkNN(
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  // Define type definition for double-precision CUTLASS SYRK with column-major
  // input matrices and 16x32x16 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for double-precision SYRK. Typical values are used as
  // default template arguments.
  //
  // To view the full syrk device API interface, see `cutlass/gemm/device/syrk.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassSyrk = cutlass::gemm::device::RankK<
    float,
    ColumnMajor,
    float,
    ColumnMajor,
    cutlass::FillMode::kLower,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<16, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,     // Stages
    16,     // AlignmentA
    false, // SplitKSerail
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
  >;

  // Define a CUTLASS SYRK type
  CutlassSyrk syrk_operator;

  // Construct the CUTLASS SYRK arguments object.
  //
  // One of CUTLASS's design patterns is to define syrk argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Syrk and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassSyrk::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                              {N, N, K}, // Syrk Problem dimensions
                              1, // batch_count,
                              {alpha, beta}, // Scalars used in the Epilogue
                              reinterpret_cast<void const *>(A),
                              const_cast<void *>(reinterpret_cast<void *>(C)),
                              reinterpret_cast<void *>(C), // destination matrix D (may be different memory than source C matrix)
                              (int64_t)N*K, // Batch strides
                              (int64_t)N*N,
                              (int64_t)N*N,
                              lda,
                              ldc,
                              ldc);

  //
  // Launch the CUTLASS SYRK kernel.
  //

  cutlass::Status status = syrk_operator(args);

  //
  // Return a cudaError_t if the CUTLASS SYRK operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}