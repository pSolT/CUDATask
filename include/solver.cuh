#ifndef SOLVER_CUH
#define SOLVER_CUH

float SolveSODE_CUDA(float* A, float* x, float* result, size_t N, unsigned t);
float SolveSODE_CUBLAS(float* A, float* x, float* result, size_t N, unsigned t);

#endif