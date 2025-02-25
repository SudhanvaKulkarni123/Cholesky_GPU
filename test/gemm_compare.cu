#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_conversion.h>
//#include <cutlass/linear_combination_clamp.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>

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

int main() {
  // Matrix dimensions (for a square GEMM)
  const int M = 1024, N = 1024, K = 512;
  const float alpha = 1.0f, beta = 0.0f;
  const int numElementsA = M * K;
  const int numElementsB = K * N;
  const int numElementsC = M * N;

  // ----------------------
  // cuBLAS fp32 GEMM Setup
  // ----------------------
  float *h_A = new float[numElementsA];
  float *h_B = new float[numElementsB];
  float *h_C = new float[numElementsC];  // output
  randomFill(h_A, numElementsA);
  randomFill(h_B, numElementsB);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, numElementsA * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_B, numElementsB * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_C, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, numElementsA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, numElementsB * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, numElementsC * sizeof(float)));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  cudaEvent_t start_cublas_real, stop_cublas_real;
  CUDA_CHECK(cudaEventCreate(&start_cublas_real));
  CUDA_CHECK(cudaEventCreate(&stop_cublas_real));
  float time_cublas_real = 0.0f;

  // Create CUDA events for cuBLAS timing
  cudaEvent_t start_cublas, stop_cublas;
  CUDA_CHECK(cudaEventCreate(&start_cublas));
  CUDA_CHECK(cudaEventCreate(&stop_cublas));
  float time_cublas = 0.0f;

  //lets start with cuBLAS Fp32
  CUDA_CHECK(cudaEventRecord(start_cublas_real));
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           M, N, K,
                           &alpha,
                           d_A, M,
                           d_B, K,
                           &beta,
                           d_C, M));
  CUDA_CHECK(cudaEventRecord(stop_cublas_real));
  CUDA_CHECK(cudaEventSynchronize(stop_cublas_real));
  CUDA_CHECK(cudaEventElapsedTime(&time_cublas_real, start_cublas_real, stop_cublas_real));



  using GemmFp32Fp32 = cutlass::gemm::device::Gemm<
      float,            // ElementA
      cutlass::layout::ColumnMajor,  // LayoutA
      float,            // ElementB
      cutlass::layout::ColumnMajor,  // LayoutB
      float,                      // ElementC (accumulation)
      cutlass::layout::ColumnMajor,  // LayoutC
      float                       // Epilogue scalar type
  >;

  GemmFp32Fp32 gemm_op_32;

  // Set up arguments for the CUTLASS GEMM
  typename GemmFp32Fp32::Arguments arguments_32{
      {M, N, K},                   // GEMM problem size
      {d_A, int(M)},          // Tensor A: pointer and leading dimension M
      {d_B, int(K)},          // Tensor B: pointer and leading dimension K
      {d_C, int(M)},        // Tensor C: pointer and leading dimension M (input/output)
      {d_C, int(M)},        // Tensor D: pointer and leading dimension M (output)
      {alpha, beta}                // Epilogue parameters
  };

  // CYTLASS FP32
  CUDA_CHECK(cudaMemset(d_C, 0, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(start_cublas));
  cutlass::Status status_32 = gemm_op_32(arguments_32);
  CUDA_CHECK(cudaEventRecord(stop_cublas));
  CUDA_CHECK(cudaEventSynchronize(stop_cublas));
  CUDA_CHECK(cudaEventElapsedTime(&time_cublas, start_cublas, stop_cublas));

  // ----------------------------
  // CUTLASS fp16 GEMM Setup
  // ----------------------------
  // Allocate host memory for fp16 (CUTLASS uses cutlass::half_t)
  cutlass::half_t *h_A_half = new cutlass::half_t[numElementsA];
  cutlass::half_t *h_B_half = new cutlass::half_t[numElementsB];
  // Convert h_A and h_B from float to cutlass::half_t using __float2half
  for (int i = 0; i < numElementsA; i++) {
    h_A_half[i] = __float2half(h_A[i]);
  }
  for (int i = 0; i < numElementsB; i++) {
    h_B_half[i] = __float2half(h_B[i]);
  }

  // Allocate device memory for fp16 inputs and fp32 output
  cutlass::half_t *d_A_half, *d_B_half;
  float *d_C_tensor;
  CUDA_CHECK(cudaMalloc((void**)&d_A_half, numElementsA * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_B_half, numElementsB * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_C_tensor, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A_half, h_A_half, numElementsA * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_half, h_B_half, numElementsB * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C_tensor, 0, numElementsC * sizeof(float)));

  // Define CUTLASS GEMM operator for fp16 inputs with fp32 accumulation.
  using GemmFp16Fp32 = cutlass::gemm::device::Gemm<
      cutlass::half_t,            // ElementA
      cutlass::layout::RowMajor,  // LayoutA
      cutlass::half_t,            // ElementB
      cutlass::layout::ColumnMajor,  // LayoutB
      float,                      // ElementC (accumulation)
      cutlass::layout::RowMajor,  // LayoutC
      float                       // Epilogue scalar type
  >;

  GemmFp16Fp32 gemm_op;

  // Set up arguments for the CUTLASS GEMM
  typename GemmFp16Fp32::Arguments arguments{
      {M, N, K},                   // GEMM problem size
      {d_A_half, int(M)},          // Tensor A: pointer and leading dimension M
      {d_A_half, int(K)},          // Tensor B: pointer and leading dimension K
      {d_C_tensor, int(M)},        // Tensor C: pointer and leading dimension M (input/output)
      {d_C_tensor, int(M)},        // Tensor D: pointer and leading dimension M (output)
      {alpha, beta}                // Epilogue parameters
  };

  // Create CUDA events for CUTLASS timing
  cudaEvent_t start_cutlass, stop_cutlass;
  CUDA_CHECK(cudaEventCreate(&start_cutlass));
  CUDA_CHECK(cudaEventCreate(&stop_cutlass));
  float time_cutlass = 0.0f;

  // Launch and time the CUTLASS GEMM kernel
  CUDA_CHECK(cudaMemset(d_C_tensor, 0, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(start_cutlass));
  cutlass::Status status = gemm_op(arguments);
  CUDA_CHECK(cudaEventRecord(stop_cutlass));
  CUDA_CHECK(cudaEventSynchronize(stop_cutlass));
  CUDA_CHECK(cudaEventElapsedTime(&time_cutlass, start_cutlass, stop_cutlass));

  // ----------------------------
  // CUTLASS FP8 GEMM Setup
  // ----------------------------
  // Allocate host memory for fp8 inputs (using cutlass::float_e4m3_t)
  cutlass::float_e4m3_t *h_A_fp8 = new cutlass::float_e4m3_t[numElementsA];
  cutlass::float_e4m3_t *h_B_fp8 = new cutlass::float_e4m3_t[numElementsB];

  // Convert h_A and h_B from float to fp8 using NumericConverter
  for (int i = 0; i < numElementsA; i++) {
    h_A_fp8[i] = cutlass::NumericConverter<cutlass::float_e4m3_t, float>()(h_A[i]);
  }
  for (int i = 0; i < numElementsB; i++) {
    h_B_fp8[i] = cutlass::NumericConverter<cutlass::float_e4m3_t, float>()(h_B[i]);
  }

  // Allocate device memory for fp8 inputs and fp32 output
  cutlass::float_e4m3_t *d_A_fp8, *d_B_fp8;
  float *d_C_tensor_8;
  CUDA_CHECK(cudaMalloc((void**)&d_A_fp8, numElementsA * sizeof(cutlass::float_e4m3_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_B_fp8, numElementsB * sizeof(cutlass::float_e4m3_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_C_tensor_8, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A_fp8, h_A_fp8, numElementsA * sizeof(cutlass::float_e4m3_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_fp8, h_B_fp8, numElementsB * sizeof(cutlass::float_e4m3_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C_tensor_8, 0, numElementsC * sizeof(float)));

  // Define CUTLASS GEMM operator for fp8 inputs with fp32 accumulation.
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Tune these numbers.
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;     // Tune these numbers.
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;       // Usually fixed for tensor ops.




using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    float,  
    8
>;


    // Now, define the GEMM operator using these tuning parameters.
using GemmFp8Fp32 = cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t,           // ElementA (FP8)
    cutlass::layout::RowMajor,       // LayoutA
    cutlass::float_e4m3_t,           // ElementB (FP8)
    cutlass::layout::ColumnMajor,       // LayoutB
    float,                           // ElementC (output)
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator (accumulation type: FP32)
    cutlass::arch::OpClassTensorOp,  // Operator class (Tensor Ops)
    cutlass::arch::Sm89,             // Architecture (Sm89 for FP8 support)
    cutlass::gemm::GemmShape<128, 64, 128>,  // Threadblock shape
    cutlass::gemm::GemmShape<64, 32, 128>,   // Warp shape
    cutlass::gemm::GemmShape<16, 8, 32> ,    // Instruction shape
    EpilogueOutputOp,                        // **Custom No-Op Epilogue**
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock swizzle
    3,    // Number of stages
    16,   // Alignment A
    16   // Alignment B
    // cutlass::arch::OpMultiplyAddFastAccum  // Operator
>;

GemmFp8Fp32 lo_gemm_op;



  // Set up arguments for the CUTLASS FP8 GEMM
  typename GemmFp8Fp32::Arguments fp8_arguments{
      {M, N, K},                    // GEMM problem size
      {d_A_fp8, int(M)},            // Tensor A: pointer and leading dimension M
      {d_A_fp8, int(K)},            // Tensor B: pointer and leading dimension K
      {d_C_tensor_8, int(M)},       // Tensor C: pointer and leading dimension M (input/output)
      {d_C_tensor_8, int(M)},       // Tensor D: pointer and leading dimension M (output)
      {alpha, beta}                 // Epilogue parameters
  };

  // Create CUDA events for FP8 timing
  cudaEvent_t fp8_start, fp8_stop;
  CUDA_CHECK(cudaEventCreate(&fp8_start));
  CUDA_CHECK(cudaEventCreate(&fp8_stop));
  float time_fp8 = 0.0f;

  // Launch and time the CUTLASS FP8 GEMM kernel
  CUDA_CHECK(cudaMemset(d_C_tensor_8, 0, numElementsC * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(fp8_start));
  status = lo_gemm_op(fp8_arguments);
  CUDA_CHECK(cudaEventRecord(fp8_stop));
  CUDA_CHECK(cudaEventSynchronize(fp8_stop));
  CUDA_CHECK(cudaEventElapsedTime(&time_fp8, fp8_start, fp8_stop));

  // Print performance results
  std::cout << "Performance Comparison:" << std::endl;
  std::cout << "cuBLAS fp32 GEMM time : " << time_cublas_real << "ms" << std::endl;
  std::cout << "CUTLASS tf32 GEMM (fp32 accumulation) time: " << time_cublas << " ms" << std::endl;
  std::cout << "CUTLASS fp16 GEMM (fp32 accumulation) time: " << time_cutlass << " ms" << std::endl;
  std::cout << "CUTLASS fp8 GEMM (fp32 accumulation) time: " << time_fp8 << " ms" << std::endl;

  double total_flops = 2.0 * M * N * K;

// Convert execution time from milliseconds to seconds
double time_cublas_real_sec = time_cublas_real / 1000.0;
double time_cublas_sec = time_cublas / 1000.0;
double time_cutlass_sec = time_cutlass / 1000.0;
double time_fp8_sec = time_fp8 / 1000.0;

// Compute performance (TFLOPS)
double tflops_cublas_real = total_flops / (time_cublas_real_sec * 1.0e12);
double tflops_cutlass_fp32 = total_flops / (time_cublas_sec * 1.0e12);
double tflops_cutlass_fp16 = total_flops / (time_cutlass_sec * 1.0e12);
double tflops_cutlass_fp8 = total_flops / (time_fp8_sec * 1.0e12);

// Print performance results
std::cout << "Performance Comparison:\n";
std::cout << "---------------------------------------------\n";
std::cout << "cuBLAS fp32 GEMM time      : " << time_cublas_real << " ms | " 
          << tflops_cublas_real << " TFLOPS\n";
std::cout << "CUTLASS fp32 GEMM time     : " << time_cublas << " ms | " 
          << tflops_cutlass_fp32 << " TFLOPS\n";
std::cout << "CUTLASS fp16 GEMM time     : " << time_cutlass << " ms | " 
          << tflops_cutlass_fp16 << " TFLOPS\n";
std::cout << "CUTLASS fp8 GEMM time      : " << time_fp8 << " ms | " 
          << tflops_cutlass_fp8 << " TFLOPS\n";


  // Cleanup host memory
  delete[] h_A; delete[] h_B; delete[] h_C;
  delete[] h_A_half; delete[] h_B_half;
  delete[] h_A_fp8; delete[] h_B_fp8;

  // Cleanup device memory
  CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_A_half)); CUDA_CHECK(cudaFree(d_B_half)); CUDA_CHECK(cudaFree(d_C_tensor));
  CUDA_CHECK(cudaFree(d_A_fp8)); CUDA_CHECK(cudaFree(d_B_fp8)); CUDA_CHECK(cudaFree(d_C_tensor_8));

  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaEventDestroy(start_cublas)); CUDA_CHECK(cudaEventDestroy(stop_cublas));
  CUDA_CHECK(cudaEventDestroy(start_cutlass)); CUDA_CHECK(cudaEventDestroy(stop_cutlass));
  CUDA_CHECK(cudaEventDestroy(fp8_start)); CUDA_CHECK(cudaEventDestroy(fp8_stop));

  std::cout << "CUTLASS GEMM completed successfully.\n";
  return 0;
}
