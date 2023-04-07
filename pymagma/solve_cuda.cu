
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
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

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
    cudaFree(dA);
    cudaFree(dtau);
    cudaFree(db);
    cudaFree(dinfo);
    cudaFree(dwork);
}

void solve_batch_cuda_QR(int nrow, int ncol, int nbatch, double* A_batch_ptr, double* b_batch_ptr, double* result_batch_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);

    int nstream = nbatch;
    cudaStream_t* streams = new cudaStream_t[nstream];
    cusolverDnHandle_t* cusolverH = new cusolverDnHandle_t[nstream];
    cublasHandle_t* cublasH = new cublasHandle_t[nstream];
    for (int i = 0; i < nstream; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        //cudaStreamCreate(&streams[i]);
        cusolverDnCreate(&cusolverH[i]);
        cusolverDnSetStream(cusolverH[i], streams[i]);
        cublasCreate(&cublasH[i]);
        cublasSetStream(cublasH[i], streams[i]);
    }

    int min_mn = nrow < ncol ? nrow : ncol;

    cudaError_t err;
    double* dA_batch;
    err = cudaMalloc((void **)&dA_batch, sizeof(double) * nrow * ncol * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    double *dtau;
    err = cudaMalloc((void **)&dtau, sizeof(double) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtau");

    double* db_batch;
    err = cudaMalloc((void **)&db_batch, sizeof(double) * nrow * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    int info;
    int *dinfo;
    err = cudaMalloc((void **)&dinfo, sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dinfo");

#if 0
    err = cudaMemcpy(dA_batch, A_batch_ptr, sizeof(double)*nrow*ncol*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db_batch, b_batch_ptr, sizeof(double)*nrow*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");
#endif
#if 1
    for (int ib = 0; ib < nbatch; ib++)  {
        err = cudaMemcpyAsync(dA_batch + ib*nrow*ncol, A_batch_ptr, sizeof(double)*nrow*ncol, cudaMemcpyHostToDevice, streams[ib]);
        if (err != cudaSuccess)
            throw std::runtime_error("failed to copy dA");

        err = cudaMemcpyAsync(db_batch + ib*nrow , b_batch_ptr, sizeof(double)*nrow, cudaMemcpyHostToDevice, streams[ib]);
        if (err != cudaSuccess)
            throw std::runtime_error("failed to copy db");
    }
#endif


    int lwork_geqrf;
    cusolverDnDgeqrf_bufferSize(cusolverH[0], nrow, ncol, dA_batch, nrow, &lwork_geqrf);

    int lwork_ormqr;
    cusolverDnDormqr_bufferSize(cusolverH[0], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA_batch, nrow, dtau, db_batch, nrow, &lwork_ormqr);

    int lwork = std::max(lwork_geqrf, lwork_ormqr);

    printf("lwork size = %d\n",lwork);
    double *dwork;
    err = cudaMalloc((void **)&dwork, sizeof(double) * lwork * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");


// Try launching streams in multiple threads
#pragma omp parallel for
    for (int ib = 0; ib < nbatch; ib++) {
        double *dA = dA_batch + ib*nrow*ncol;
        double *db = db_batch + ib*nrow;
        cusolverStatus_t serr;

        serr = cusolverDnDgeqrf(cusolverH[ib], nrow, ncol, dA, nrow, dtau, dwork + lwork*ib, lwork, dinfo);
        if (serr != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgeqrf failed");
        //cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, streams[ib]);
        //if (info != 0)
        // std::runtime_error(std::string("dgeqrf info = ") + std::to_string(info));

        serr = cusolverDnDormqr(cusolverH[ib], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, ncol, dA, nrow, dtau, db, nrow, dwork + lwork*ib, lwork, dinfo);
        if (serr != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgeqrf failed");
        //cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, streams[ib]);
        //if (info != 0)
        //    std::runtime_error(std::string("dormqr info = ") + std::to_string(info));

        double one(1.0);
        cublasDtrsm(cublasH[ib], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncol, 1, &one, dA, nrow, db, nrow);

        cudaMemcpyAsync(result_batch_ptr + ib*ncol, db, sizeof(double)*ncol, cudaMemcpyDeviceToHost, streams[ib]);
    }


    //cudaMemcpy(result_batch_ptr, db_batch, sizeof(double)*nrow*nbatch, cudaMemcpyDeviceToHost);
    //for (int i = 0; i < nstream; i++)
    //    cudaStreamSynchronize(streams[i]);

    cudaDeviceSynchronize();



    cudaFree(dA_batch);
    cudaFree(dtau);
    cudaFree(db_batch);
    cudaFree(dinfo);
    cudaFree(dwork);
}
