#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define B 1     // Batch size
#define C 4     // Input channels
#define M 16    // Output features
#define H 256   // Input height
#define W 256   // Input width
#define K 3     // Filter size

// CUDA kernel for 2D convolution
__global__ void ConvolutionKernel(float* input, float* output, float* filter, int Z) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int ox = bx * blockDim.x + tx;
    int oy = by * blockDim.y + ty;

    // Check if the thread is within the image boundaries
    if (ox < W && oy < H) {
        for (int m = 0; m < Z; ++m) {
            float result = 0.0f;
            // Loop over all input channels
            for (int c = 0; c < C; ++c) {
                // Loop over the convolution kernel
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int ix = ox - K / 2 + kx;
                        int iy = oy - K / 2 + ky;
                        // Check for valid index and accumulate the convolution result
                        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                            result += input[c * H * W + iy * W + ix] * filter[m * C * K * K + c * K * K + ky * K + kx];
                        }
                    }
                }
            }
            // Write the convolution result to the output
            output[m * H * W + oy * W + ox] = result;
        }
    }
}

// CPU implementation of convolution
void ConvolutionCPU(float* input, float* output, float* filter) {
    // Loop over batch, height, width, and output channels
    for (int by = 0; by < B; ++by) {
        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {
                for (int m = 0; m < M; ++m) {
                    float result = 0.0f;
                    // Loop over input channels and convolution kernel
                    for (int c = 0; c < C; ++c) {
                        for (int ky = 0; ky < K; ++ky) {
                            for (int kx = 0; kx < K; ++kx) {
                                int ix = ox - K / 2 + kx;
                                int iy = oy - K / 2 + ky;
                                // Check for valid index and accumulate the convolution result
                                if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                                    result += input[by * C * H * W + c * H * W + iy * W + ix] * filter[m * C * K * K + c * K * K + ky * K + kx];
                                }
                            }
                        }
                    }
                    // Write the convolution result to the output
                    output[by * M * H * W + m * H * W + oy * W + ox] = result;
                }
            }
        }
    }
}

int main() {
    // Allocate memory for input and output data
    size_t inputSize = B * C * H * W * sizeof(float);
    size_t outputSize = B * M * H * W * sizeof(float);
    size_t filterSize = M * C * K * K * sizeof(float);

    // Setup CUDA memory and copy data
    float* h_input = (float*)malloc(inputSize);
    float* h_output = (float*)malloc(outputSize);
    float* h_gpu_output = (float*)malloc(outputSize);
    float* h_filter = (float*)malloc(filterSize);

    // Initialize input data and filter
    for (int i = 0; i < B * C * H * W; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * C * K * K; ++i) {
        h_filter[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_input, * d_output, * d_filter;
    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);
    cudaMalloc((void**)&d_filter, filterSize);

    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);

    // Timer setup for CPU
    struct timeval cpu_start, cpu_end;
    gettimeofday(&cpu_start, NULL);

    // CPU convolution
    ConvolutionCPU(h_input, h_output, h_filter);
    gettimeofday(&cpu_end, NULL);
    double cpuTime = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;
    printf("CPU execution time: %.3f ms\n", cpuTime);

    // Timer setup for GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the convolution kernel
    ConvolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_filter, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU execution time: %.3f ms\n", gpuTime);

    cudaMemcpy(h_gpu_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Verify results
    bool resultMatch = true;
    for (int i = 0; i < B * M * H * W; ++i) {
        if (fabs(h_output[i] - h_gpu_output[i]) > 1e-5) {
            printf("Result verification failed at element [%d]!\n", i);
            printf("CPU result: %.6f\n", h_output[i]);
            printf("GPU result: %.6f\n", h_gpu_output[i]);
            resultMatch = false;
            break;
        }
    }
    if (resultMatch) {
        printf("Result verification succeeded!\n");
    } else {
        printf("Result verification failed!\n");
    }
    
    // Free memory and clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_output);
    free(h_gpu_output);
    free(h_filter);

    return 0;
}
