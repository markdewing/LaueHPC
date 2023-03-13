
#include "solve_cuda.h"
#include "perf_info.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdexcept>


void solve_cuda_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    int min_mn = nrow < ncol ? nrow : ncol;

    cudaError_t err;
    double* dA;
    err = cudaMalloc((void **)&dA, sizeof(double) * nrow * ncol);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dA");

    double *dtau;
    err = cudaMalloc((void **)&dtau, sizeof(double) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtau");

    double* db;
    err = cudaMalloc((void **)&db, sizeof(double) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    int info;
    int *dinfo;
    err = cudaMalloc((void **)&dinfo, sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dinfo");


    err = cudaMemcpy(dA, A_ptr, sizeof(double)*nrow*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db, b_ptr, sizeof(double)*nrow, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");

    int lwork_geqrf;
    cusolverDnDgeqrf_bufferSize(cusolverH, nrow, ncol, dA, nrow, &lwork_geqrf);

    int lwork_ormqr;
    cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA, nrow, dtau, db, nrow, &lwork_ormqr);

    int lwork = std::max(lwork_geqrf, lwork_ormqr);

    printf("lwork size = %d\n",lwork);
    double *dwork;
    err = cudaMalloc((void **)&dwork, sizeof(double) * lwork);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");


    cusolverStatus_t serr;
    serr = cusolverDnDgeqrf(cusolverH, nrow, ncol, dA, nrow, dtau, dwork, lwork, dinfo);
    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgeqrf failed");
    cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0)
     std::runtime_error(std::string("dgeqrf info = ") + std::to_string(info));

    serr = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, ncol, dA, nrow, dtau, db, nrow, dwork, lwork, dinfo);
    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgeqrf failed");
    cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0)
        std::runtime_error(std::string("dormqr info = ") + std::to_string(info));

    double one(1.0);
    cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncol, 1, &one, dA, nrow, db, nrow);

    cudaMemcpy(result_ptr, db, sizeof(double)*ncol, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

