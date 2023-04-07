#ifndef SOLVE_CUDA_H
#define SOLVE_CUDA_H

struct PerfInfo;

// A is nrow x ncol
// b is nrow
// x is ncol

// solve single system on gpu
void solve_cuda_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

// solve multiple systems on gpu
void solve_batch_cuda_QR(int nrow, int ncol, int nbatch, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

#endif
