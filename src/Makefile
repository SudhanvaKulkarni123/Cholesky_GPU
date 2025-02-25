
NVCC = nvcc

# Target executable name
TARGET = mixed_cholesky

# Source files
SRCS = mixed_cholesky.cu

# Compilation flags (adjust as needed)
CFLAGS = -arch=sm_89 -std=c++20 -O3 -use_fast_math

# Libraries to link
LDFLAGS = -lcublas -lcusolver -lcurand -lcublasLt

INCLUDE_PATH = -I$(HOME)/cutlass/include -I$(HOME)/LoFloat/tools

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDE_PATH) $(SRCS) $(LDFLAGS) -o $(TARGET)

# Clean up build artifacts
clean:
	rm -f $(TARGET)
