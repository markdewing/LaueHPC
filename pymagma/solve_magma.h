#ifndef SOLVE_MAGMA_H
#define SOLVE_MAGMA_H

void init_magma();
void fini_magma();
// A is nrow x ncol
// b is nrow
// x is ncol
void solve_cpu(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr);

// solve single system on gpu
void solve_gpu(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr);

// Solve using simplest Magma interfaces
void solve_gpu_simple(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr);

#endif
