
NVCC = nvcc

# Target executable name
TARGET = mixed_cholesky

# Source files
SRCS = mixed_cholesky.cu

# Compilation flags (adjust as needed)
CFLAGS = -O3

# Libraries to link
LDFLAGS = -lcublas -lcusolver

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

# Clean up build artifacts
clean:
	rm -f $(TARGET)
