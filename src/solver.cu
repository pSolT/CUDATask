#include "helper_functions.cuh"

#define BLOCK_SIZE 1024

// Use dynamic shared memory
__global__ void solverKernel(float* A, float* x, float* result, size_t n)
{
    // Use dynamically allocated shared memory
    int tId = blockIdx.y * gridDim.x +  blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float cValue = A[tId] * x[col];
    atomicAdd(&result[blockIdx.y], cValue);
}

float SolveSODE_CUDA(float* A, float* x, float* result, size_t N)
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
    dim3 gridSize(N/BLOCK_SIZE, N, 1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, NULL));
    solverKernel<<<blockSize, gridSize>>>(dA, dx, dResult, N);  
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

float SolveSODE_CUBLAS(float* A, float* x, float* result, size_t N)
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