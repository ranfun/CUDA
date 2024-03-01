#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define DimMIN 1000
#define DimMAX 2000
#define TestSamples 1
#define nIter 1

#define TILE_WIDTH 30

__global__ void MatrixMulCUDA(
    const float *A, 
    const float *B, 
    float *C,
    int wA,
    int hA,
    int wB,
    int hB) 
{
    // Declare shared memory for the tile sub-matrix of A and B
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    // Block index and thread index within the block
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row and column indices for this thread
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pvalue = 0;

    // Loop over the tiles of matrix A and B required to compute the tile of C
    for (int q = 0; q <= (wA - 1) / TILE_WIDTH + 1; ++q) {
        // Load the tile of matrix A into shared memory
        if (row < hA && q * TILE_WIDTH + tx < wA)
            subTileA[ty][tx] = A[row * wA + q * TILE_WIDTH + tx];
        else
            subTileA[ty][tx] = 0;

        // Load the tile of matrix B into shared memory
        if (q * TILE_WIDTH + ty < hB && col < wB)
            subTileB[ty][tx] = B[(q * TILE_WIDTH + ty) * wB + col];
        else
            subTileB[ty][tx] = 0;
        __syncthreads();

        // Multiply the two tiles and accumulate the results
        for (int k = 0; k < TILE_WIDTH; ++k)
            pvalue += subTileA[ty][k] * subTileB[k][tx];

        __syncthreads();
    }

    // Write the computed value to matrix C
    if (row < hA && col < wB) {
        C[row * wB + col] += pvalue;
    }
}

void MatrixMultiplication(float *h_A, float *h_B, float *h_C, int wA, size_t mem_size_A, int wB, size_t mem_size_B, size_t mem_size_C) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, mem_size_A);
    cudaMalloc((void**)&d_B, mem_size_B);
    cudaMalloc((void**)&d_C, mem_size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    int hA = mem_size_A / (wA * sizeof(float));
    int hB = mem_size_B / (wB * sizeof(float));

    // Grid dimensions are set based on the size of matrices and TILE_WIDTH
    dim3 grid(ceil((1.0 * wB) / TILE_WIDTH), ceil((1.0 * hA) / TILE_WIDTH), 1);
    // Block dimensions are set to TILE_WIDTH x TILE_WIDTH
    dim3 threads(TILE_WIDTH, TILE_WIDTH, 1);

    //create timer
    float total_time = 0;
    struct timeval start, end;
    for (int j = 0; j < nIter; j++) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);
        // Call the CUDA kernel
        MatrixMulCUDA<<<grid, threads>>>(d_A, d_B, d_C, wA, hA, wB, hB);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }

    // Print kernel latency
    printf("GPU execution time: %.3f ms\n",  total_time );

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(void) {
    srand(time(0));
    for (int testiter = 0; testiter < TestSamples; testiter++) {
        printf(">>>>>>>>>>>>>>>>> test No. %d >>>>>>>>>>>>>>>>>\n", testiter + 1);

        dim3 dimsA(rand() % (DimMAX - DimMIN + 1) + DimMIN, rand() % (DimMAX - DimMIN + 1) + DimMIN, 1);
        dim3 dimsB(rand() % (DimMAX - DimMIN + 1) + DimMIN, dimsA.x, 1);
        printf("[Matrix Multiplication of (%d,%d) x (%d,%d) ]\n", dimsA.y, dimsA.x, dimsB.y, dimsB.x);

        unsigned int size_A = dimsA.x * dimsA.y;
        size_t mem_size_A = sizeof(float) * size_A;
        float *h_A = (float *)malloc(mem_size_A);

        unsigned int size_B = dimsB.x * dimsB.y;
        size_t mem_size_B = sizeof(float) * size_B;
        float *h_B = (float *)malloc(mem_size_B);

        unsigned int size_C = dimsA.y * dimsB.x;
        size_t mem_size_C = sizeof(float) * size_C;
        float *h_C = (float *)malloc(mem_size_C);

        if (h_A == NULL || h_B == NULL || h_C == NULL) {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < size_A; ++i) {
            h_A[i] = rand() / (float)RAND_MAX;
        }
        for (int i = 0; i < size_B; ++i) {
            h_B[i] = rand() / (float)RAND_MAX;
        }

        MatrixMultiplication(h_A, h_B, h_C, dimsA.x, mem_size_A, dimsB.x, mem_size_B, mem_size_C);

        // CPU matrix multiplication for verification
        float *h_C_CPU = (float *)malloc(mem_size_C);
        struct timeval cpu_start, cpu_end;
        gettimeofday(&cpu_start, NULL);
        for (int i = 0; i < dimsA.y; i++) {
            for (int j = 0; j < dimsB.x; j++) {
                float sum = 0.0f;
                for (int k = 0; k < dimsA.x; k++) {
                    sum += h_A[i * dimsA.x + k] * h_B[k * dimsB.x + j];
                }
                h_C_CPU[i * dimsB.x + j] = sum;
            }
        }
        gettimeofday(&cpu_end, NULL);
        double cpuTime = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_end.tv_usec - cpu_start.tv_usec) * 0.001;
        printf("CPU execution time: %.3f ms\n", cpuTime);

        // Check results
        for (int i = 0; i < dimsA.y; i++) {
            for (int j = 0; j < dimsB.x; j++) {
                if (fabs(h_C_CPU[i * dimsB.x + j] - h_C[i * dimsB.x + j]) > 1e-2) {
                    fprintf(stderr, "Result verification failed at element (%d,%d)!\n", i, j);
                    // Optional: print the mismatching values
                    printf("CPU result: %.3f, GPU result: %.3f\n", h_C_CPU[i * dimsB.x + j], h_C[i * dimsB.x + j]);
                }
            }
        }

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_CPU);
    }
    return 0;
}
