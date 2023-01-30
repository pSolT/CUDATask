#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <random>
#include "solver.cuh"

void GenerateRandomMatrix(size_t height, size_t width, float* output)
{
    srand (time(NULL));
    for(int i=0; i<(height * width); i++)
    {
        output[i] = float(rand() / (RAND_MAX + 1.));
    }
}

void TestCUDA(size_t A, unsigned t)
{
    float * matrix = (float*)malloc(A * A * sizeof(float));
    float * vector = (float*)malloc(A * sizeof(float));
    float * result = (float*)malloc(A * sizeof(float));

    GenerateRandomMatrix(A, A, matrix);
    GenerateRandomMatrix(A, 1, vector);

    float elapsedTimeMs = SolveSODE_CUDA(matrix, vector, result, A, t);
    printf("CUDA elapsed time: %.2f ms\n", elapsedTimeMs);

    free(matrix);
    free(vector);
    free(result);
}

void TestCUBLAS(size_t A, unsigned t)
{

    float * matrix = (float*)malloc(A * A * sizeof(float));
    float * vector = (float*)malloc(A * sizeof(float));
    float * result = (float*)malloc(A * sizeof(float));

    GenerateRandomMatrix(A, A, matrix);
    GenerateRandomMatrix(A, 1, vector);

    float elapsedTimeMs = SolveSODE_CUBLAS(matrix, vector, result, A, t);
    printf("CUBLAS elapsed time: %.2f ms\n", elapsedTimeMs);

    free(matrix);
    free(vector);
    free(result);
}

#endif