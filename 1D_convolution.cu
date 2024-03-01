#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 1024
#define MASK_WIDTH 5

// CUDA kernel for basic convolution
__global__ void BasicConvolutionKernel(float *input, float *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread index

    if (tid < INPUT_SIZE) {
        float result = 0.0f;
        int halfMaskWidth = MASK_WIDTH / 2; // Calculate half width of the mask

        // Apply convolution filter
        for (int j = -halfMaskWidth; j <= halfMaskWidth; j++) {
            int idx = tid + j; // Index with offset for convolution
            if (idx >= 0 && idx < INPUT_SIZE) {
                result += input[idx]; // Accumulate the convolution result
            }
        }

        output[tid] = result; // Store the result in the output array
    }
}

// Function to perform basic convolution on GPU
void BasicConvolution(float *input, float *output) {
    float *d_input, *d_output;
    const int arraySize = INPUT_SIZE * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_input, arraySize);
    cudaMalloc((void **)&d_output, arraySize);

    // Copy input from host to device
    cudaMemcpy(d_input, input, arraySize, cudaMemcpyHostToDevice);

    // Set grid and block dimensions for kernel launch
    int threadsPerBlock = 64; // Adjust as needed
    int blocksPerGrid = (INPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Create timer for measuring GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time for GPU execution
    cudaEventRecord(start);

    // Launch the convolution kernel
    BasicConvolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);

    // Record stop time for GPU execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, arraySize, cudaMemcpyDeviceToHost);

    // Calculate and print GPU execution time
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU execution time: %.3f ms\n", gpuTime);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Allocate host input and output arrays
    float *h_input = (float *)malloc(INPUT_SIZE * sizeof(float));
    float *h_output = (float *)malloc(INPUT_SIZE * sizeof(float));
    float *cpu_output = (float *)malloc(INPUT_SIZE * sizeof(float)); // For CPU reference

    // Initialize host input with random data
    for (int i = 0; i < INPUT_SIZE; ++i) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    // Create timer for measuring CPU execution time
    struct timeval c_start, c_end;

    // Record start time for CPU execution
    gettimeofday(&c_start, NULL);

    // Compute convolution on CPU for reference
    for (int i = 0; i < INPUT_SIZE; ++i) {
        float result = 0.0f;
        int halfMaskWidth = MASK_WIDTH / 2;

        // Apply convolution filter
        for (int j = -halfMaskWidth; j <= halfMaskWidth; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < INPUT_SIZE) {
                result += h_input[idx];
            }
        }

        cpu_output[i] = result;
    }

    // Record stop time for CPU execution
    gettimeofday(&c_end, NULL);

    // Calculate and print CPU execution time
    float cpuTime = (c_end.tv_sec - c_start.tv_sec) * 1000 + (c_end.tv_usec - c_start.tv_usec) * 0.001;
    printf("CPU execution time: %.3f ms\n", cpuTime);

    // Call CUDA convolution function
    BasicConvolution(h_input, h_output);

    // Compare GPU result with CPU reference for verification
    bool success = true;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (fabs(cpu_output[i] - h_output[i]) > 1e-4) {
            printf("Result verification failed at element [%d]!\n", i);
            success = false;
        }
    }

    if (success) {
        printf("Result verification succeeded!\n");
    } else {
        printf("Result verification failed!\n");
    }

    // Free host memory
    free(h_input);
    free(h_output);
    free(cpu_output);

    return 0;
}
