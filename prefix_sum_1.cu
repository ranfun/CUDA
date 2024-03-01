#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define nIter 1 // Setup num of iterations
#define N 1024
#define blockSize 1024

__global__ void PrefixSumSmallCUDA(
    float *X,
    float *Y,
    int length)
{
    // Shared memory to store temporary results
    extern __shared__ float temp[];

    // Thread index within the block
    int tid = threadIdx.x;
    // Overall index in the array
    int idx = blockIdx.x * blockDim.x + tid;

    // Load elements from global memory to shared memory
    if (idx < length) {
        temp[tid] = X[idx];
    }
    else {
        temp[tid] = 0.0;    // Zero padding for out-of-bound threads
    }

    __syncthreads();    // Synchronize to ensure all loads are completed

    // Perform parallel reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * (tid + 1) - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];    // Add elements at stride distance
        }
        __syncthreads();    // Synchronize at each reduction step
    }

    // Perform parallel post-reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = 2 * stride * (tid + 1) - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();    // Synchronize at each post-reduction step
    }

    // Write results from shared memory back to global memory
    if (idx < length) {
        Y[idx] = temp[tid];
    }
}

void PrefixSumSmall(
    float *input,
    float *output,
    int length
)
{
    float *d_out, *d_in;
    const int arraySize = length * sizeof(float);
    // Allocate device memory
    cudaMalloc((void **)&d_out, arraySize);
    cudaMalloc((void **)&d_in, arraySize);

    // Copy input from host to device
    cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

    // Grid dimension based on input size and block size
    dim3 grid((length + blockSize - 1) / blockSize, 1, 1);

    // Block dimension - each block has 'blockSize' threads
    dim3 threads(blockSize, 1, 1);

    // Create timer
    float total_time = 0;
    struct timeval start, end;
    for (int j = 0; j < nIter; j++) {
        cudaDeviceSynchronize();
        gettimeofday(&start, NULL);
        // Call the kernel
        PrefixSumSmallCUDA<<<grid, threads, blockSize * sizeof(float)>>>(d_in, d_out, length);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }

    // Print kernel latency
    printf("GPU execution time: %.3f ms\n", total_time);
    // Copy the result back to the host
    cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_out);
    cudaFree(d_in);
}

int main()
{
    size_t mem_size = sizeof(float) * N;
    // Allocate the host input
    float *h_in = (float *)malloc(mem_size);
    // Allocate the host output
    float *h_out = (float *)malloc(mem_size);
    // Allocate CPU input and output
    float *cpu_in = (float *)malloc(mem_size);
    float *cpu_out = (float *)malloc(mem_size);
    // Initialize the host input vectors with random data
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() / (float)RAND_MAX;
        cpu_in[i] = h_in[i];
    }
    // Call CUDA top function here
    PrefixSumSmall(h_in, h_out, N); // Pass pointers to arrays here
    // CPU implementation here
    float cpu_time = 0;
    struct timeval c_start, c_end;
    gettimeofday(&c_start, NULL);
    cpu_out[0] = cpu_in[0];
    for (int j = 1; j < N; j++) {
        cpu_out[j] = cpu_out[j - 1] + cpu_in[j];
    }
    gettimeofday(&c_end, NULL);
    cpu_time = (c_end.tv_sec - c_start.tv_sec) * 1000 + (c_end.tv_usec - c_start.tv_usec) * 0.001;
    printf("CPU execution time: %.3f ms\n", cpu_time);
    bool Success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(cpu_out[i] - h_out[i]) > 1e-3) {
            printf("CPU result: %.3f \n", cpu_out[i]);
            printf("GPU result: %.3f \n", h_out[i]);
            fprintf(stderr, "Result verification failed at element [%d]!\n", i);
            Success = false;
        }
    }
    if (Success == true) {
        printf("Result verification Success\n");
    }
    free(h_in);
    free(h_out);
    free(cpu_in);
    free(cpu_out);
    return 0;
}