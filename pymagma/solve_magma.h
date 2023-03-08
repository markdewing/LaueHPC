#ifndef SOLVE_MAGMA_H
#define SOLVE_MAGMA_H

struct PerfInfo;

void init_magma();
void fini_magma();
// A is nrow x ncol
// b is nrow
// x is ncol

// solve single system on gpu
void solve_gpu_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

// Solve using simplest Magma interfaces
void solve_gpu_simple_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

void solve_gpu_simple_SVD(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);
#endif
