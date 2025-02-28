NVCC = nvcc

# Target executable name
TARGET = mixed_cholesky

# Source files
SRCS = mixed_cholesky.cu

# Compilation flags
CFLAGS = -arch=sm_89 -std=c++20 -O3 -use_fast_math -w

# MKL Paths
MKLROOT = /ecad/tools/intel/oneapi-2024.0.1/mkl/2024.0
MKL_INCLUDE = -I$(MKLROOT)/include
MKL_LIB = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl


# CUDA Libraries
LDFLAGS = -lcublas -lcusolver -lcurand -lcublasLt $(MKL_LIB)

# Include Paths
INCLUDE_PATH = -I$(HOME)/cutlass/include -I$(HOME)/LoFloat/tools -I$(HOME)/thrust/include $(MKL_INCLUDE)

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDE_PATH) $(SRCS) $(LDFLAGS) -o $(TARGET)

# Linter target
lint:
	clang-tidy $(SRCS) -- -x cuda --cuda-gpu-arch=sm_89 $(CFLAGS) $(INCLUDE_PATH)

# Static analysis for CUDA
cuda-check:
	cuda-memcheck --tool racecheck ./$(TARGET)

# Clean up build artifacts
clean:
	rm -f $(TARGET)
