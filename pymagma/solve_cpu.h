#ifndef SOLVE_CPU_H
#define SOLVE_CPU_H

// A is nrow x ncol
// b is nrow
// x is ncol
void solve_cpu_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr);

void solve_cpu_SVD(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr);

#endif
