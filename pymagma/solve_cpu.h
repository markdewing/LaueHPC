#ifndef SOLVE_CPU_H
#define SOLVE_CPU_H

struct PerfInfo;
// A is nrow x ncol
// b is nrow
// x is ncol
void solve_cpu_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

void solve_cpu_SVD(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);

template<typename T>
void solve_cpu_LS(int nrow, int ncol, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);

template<typename T>
void solve_batch_cpu_QR(int nrow, int ncol, int nbatch, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);

template<typename T>
void solve_batch_cpu_SVD(int nrow, int ncol, int nbatch, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf);
#endif
