# Makefile to compile multiple CUDA programs

# Compiler
NVCC = nvcc

# Targets
all: tiled_matrix_multiplication prefix_sum_1 prefix_sum_2 convolution_1 convolution_2

tiled_matrix_multiplication:
	$(NVCC) -o tiled_matrix_multiplication matrix_multiplication.cu

prefix_sum_1:
	$(NVCC) -o prefix_sum_1 prefix_sum_1.cu

prefix_sum_2:
	$(NVCC) -o prefix_sum_2 prefix_sum_2.cu

convolution_1:
	$(NVCC) -o convolution_1 1D_convolution.cu

convolution_2:
	$(NVCC) -o convolution_2 2D_convolution.cu

# Clean
clean:
	rm -f tiled_matrix_multiplication prefix_sum_1 prefix_sum_2 convolution_1 convolution_2
