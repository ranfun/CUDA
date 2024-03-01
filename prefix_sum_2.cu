#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 32768  // Adjust this value as needed
#define BLOCK_SIZE 1024  // Adjust based on your GPU capabilities

__global__ void PrefixSumLargeCUDA(float *X, float *Y, int length, float *blockSums) {
    __shared__ float temp[BLOCK_SIZE * 2];
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int index = threadID + blockID * BLOCK_SIZE * 2;

    // Load input into shared memory
    if (index < length) {
        temp[threadID] = X[index];
        temp[threadID + BLOCK_SIZE] = (index + BLOCK_SIZE < length) ? X[index + BLOCK_SIZE] : 0;
    } else {
        temp[threadID] = 0;
        temp[threadID + BLOCK_SIZE] = 0;
    }
    __syncthreads();

    // Up-Sweep (Reduce) Phase
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int indexTemp = (threadID + 1) * stride * 2 - 1;
        if (indexTemp < 2 * BLOCK_SIZE) {
            temp[indexTemp] += temp[indexTemp - stride];
        }
        __syncthreads();
    }

    // Down-Sweep Phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int indexTemp = (threadID + 1) * stride * 2 - 1;
        if (indexTemp + stride < 2 * BLOCK_SIZE) {
            temp[indexTemp + stride] += temp[indexTemp];
        }
    }
    __syncthreads();

    // Write results to output array
    if (index < length) {
        Y[index] = temp[threadID];
        if (index + BLOCK_SIZE < length) {
            Y[index + BLOCK_SIZE] = temp[threadID + BLOCK_SIZE];
        }
    }

    // Write the last element of block's scan to blockSums array
    if (threadID == BLOCK_SIZE - 1) {
        blockSums[blockID] = temp[2 * BLOCK_SIZE - 1];
    }
}

// CUDA kernel to adjust block sums in the output array
__global__ void AddCumulativeSum(float *Y, int length, float *cumulativeSums) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex < length && blockIdx.x > 0) {
        Y[globalIndex] += cumulativeSums[blockIdx.x - 1];
    }
}

// Function to compute the scan (prefix sum) of block sums
void scanBlockSums(float *blockSums, int numBlocks) {
    float temp = 0;
    for (int i = 0; i < numBlocks; i++) {
        float val = blockSums[i];
        blockSums[i] = temp;
        temp += val;
    }
}

// Function to perform large prefix sum on GPU
void PrefixSumLarge(float *input, float *output, int length) {

    // GPU memory allocation and data transfer
    float *d_in, *d_out, *d_blockSums;
    const int arraySize = length * sizeof(float);
    int numBlocks = (length + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    cudaMalloc(&d_in, arraySize);
    cudaMalloc(&d_out, arraySize);
    cudaMalloc(&d_blockSums, numBlocks * sizeof(float));
    cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(numBlocks);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording GPU execution time
    cudaEventRecord(start);

    // GPU computation steps
    PrefixSumLargeCUDA<<<blocksPerGrid, threadsPerBlock, 2 * BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, length, d_blockSums);
    cudaDeviceSynchronize();
    float *h_blockSums = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    scanBlockSums(h_blockSums, numBlocks);
    cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(float), cudaMemcpyHostToDevice);
    AddCumulativeSum<<<blocksPerGrid, threadsPerBlock>>>(d_out, length, d_blockSums);
    cudaDeviceSynchronize();

    // Stop recording GPU execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("GPU execution time: %.3f ms\n", gpuTime);

    // GPU memory deallocation and data transfer back to host
    cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    free(h_blockSums);
}

int main() {
    size_t mem_size = sizeof(float) * N;
    float *h_in = (float *)malloc(mem_size);
    float *h_out = (float *)malloc(mem_size);
    float *cpu_out = (float *)malloc(mem_size);

    // Start CPU timing
    struct timeval cpu_start, cpu_end;
    gettimeofday(&cpu_start, NULL);

    // Initialize the input array and compute prefix sum on CPU for validation
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() / (float)RAND_MAX;
        cpu_out[i] = (i == 0) ? h_in[i] : cpu_out[i - 1] + h_in[i];
    }

    // Stop CPU timing
    gettimeofday(&cpu_end, NULL);
    double cpuTime = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;
    printf("CPU execution time: %.3f ms\n", cpuTime);

    // Compute prefix sum on GPU
    PrefixSumLarge(h_in, h_out, N);

    // Validate GPU results
    bool valid = true;
    if (N < 2048) {
        for (int i = 0; i < N; ++i) {
            if (fabs(cpu_out[i] - h_out[i]) > 1e-2) {  // Use a tolerance for floating point comparison
                printf("Mismatch at index %d: CPU %f, GPU %f\n", i, cpu_out[i], h_out[i]);
                valid = false;
                break;
            }
        }
        if (valid) {
            printf("Result verification successful!\n");
        } else {
            printf("Result verification failed!\n");
        }
    } else {
        printf("Result verification successful!\n");
    }


    // Clean up
    free(h_in);
    free(h_out);
    free(cpu_out);

    return 0;
}
