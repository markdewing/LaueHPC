#ifndef SOLVE_CUDA_H
#define SOLVE_CUDA_H

struct PerfInfo;

// A is nrow x ncol
// b is nrow
// x is ncol

// solve single system on gpu
template<typename T>
void solve_cuda_QR(int nrow, int ncol, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);

template<typename T>
void solve_cuda_SVD(int nrow, int ncol, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);

// solve multiple systems on gpu
template<typename T>
void solve_batch_cuda_QR(int nrow, int ncol, int nbatch, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);

template<typename T>
void solve_batch_cuda_LS(int nrow, int ncol, int nbatch, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);
#endif
