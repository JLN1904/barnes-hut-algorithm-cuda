NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -I/usr/local/cuda/include -rdc=true
CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart
TARGET = nbody_cuda

SRCS = main.cu nbody_cuda_kernels.cu plummer.cpp quadtree_gpu.cu

HEADERS = common_structs.h nbody_cuda_kernels.h plummer.h quadtree_gpu.h

CU_SRCS := $(filter %.cu, $(SRCS))
CPP_SRCS := $(filter %.cpp, $(SRCS))
OBJS := $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -arch=sm_86 $(OBJS) -o $@ $(CUDA_LIB)

# Regla para .cu
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Regla para .cpp 
%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o
