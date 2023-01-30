#include "helper_functions.cuh"

#define BLOCK_SIZE 1024


__global__ void solverKernel(float* A, float* x, float* result, size_t n, unsigned t, float h)
{
    __shared__ float ATile[BLOCK_SIZE];
    __shared__ float xTile[BLOCK_SIZE];
    float cValue = 1.0f;

    int aIdx = n * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Be careful not to go out of bounds
    if(xIdx < n)
    {
        // Copy row of A and x into shared memory in order to minimize the reads
        ATile[threadIdx.x] = A[aIdx];
        xTile[threadIdx.x] = x[xIdx];
        __syncthreads();

        for(int i=0; i<t; i++)
        {
            cValue += h * (ATile[aIdx] * xTile[xIdx]);
        }
        atomicAdd(&result[blockIdx.y], cValue);
    }
}

float SolveSODE_CUDA(float* A, float* x, float* result, size_t N, unsigned t)
{
    float *dA, *dx, *dResult;
    // Allocate the memory and memset result to 0
    checkCudaErrors(cudaMalloc(&dA, N*N*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dA, A, N*N*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&dx, N*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dx, x, N*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&dResult, N*sizeof(float)));
    checkCudaErrors(cudaMemset(dResult, 0, N*sizeof(float)));

    dim3 blockSize(BLOCK_SIZE, 1, 1);
    int numBlocksX = ceil(float(N)/BLOCK_SIZE);
    dim3 gridSize(numBlocksX, N, 1);

    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
            gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, NULL));


    float h = 0.01;
    solverKernel<<<blockSize, gridSize>>>(dA, dx, dResult, N, t, h);  

    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors(cudaEventRecord(stop, NULL));
    
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsedTimeMs = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMs, start, stop));

    checkCudaErrors(cudaMemcpy(result, dResult, N*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dx));
    checkCudaErrors(cudaFree(dResult));
    
    return elapsedTimeMs;
}

float SolveSODE_CUBLAS(float* A, float* x, float* result, size_t N, unsigned t)
{
    float *dA, *dx, *dResult;
    // Allocate the memory and memset result to 0
    checkCudaErrors(cudaMalloc(&dA, N*N*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dx, N*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dResult, N*sizeof(float)));
    checkCudaErrors(cudaMemset(dResult, 0, N*sizeof(float)));

    cublasCheckErrors(cublasSetVector(N, sizeof(float), x, 1, dx, 1));
    cublasCheckErrors(cublasSetMatrix(N, N, sizeof(float), A, N, dA, N));

    cublasHandle_t handle;
    cublasCheckErrors(cublasCreate(&handle));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float alpha = (-1.0f); // We want a negative
    float beta = 0.0f;

    checkCudaErrors(cudaEventRecord(start, NULL));
    cublasCheckErrors(cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, dA, N, dx, 1, &beta, dResult, 1));
    checkCudaErrors(cudaEventRecord(stop, NULL));
    
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsedTimeMs = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMs, start, stop));
    
    cublasCheckErrors(cublasGetVector(N, sizeof(float), dResult, 1, result, 1));

    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dx));
    checkCudaErrors(cudaFree(dResult));
    
    return elapsedTimeMs;
}