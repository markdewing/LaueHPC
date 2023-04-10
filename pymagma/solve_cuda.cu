
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


void solve_cuda_SVD(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf)
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

    double* db;
    err = cudaMalloc((void **)&db, sizeof(double) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    int info;
    int *dinfo;
    err = cudaMalloc((void **)&dinfo, sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dinfo");

    double* S = new double[min_mn];

    double* dS;
    err = cudaMalloc((void **)&dS, sizeof(double) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dS");

    double* dU;
    err = cudaMalloc((void **)&dU, sizeof(double) * nrow * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dU");

    double* dVT;
    err = cudaMalloc((void **)&dVT, sizeof(double) * ncol * ncol);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dVT");



    err = cudaMemcpy(dA, A_ptr, sizeof(double)*nrow*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db, b_ptr, sizeof(double)*nrow, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");


    int lwork_gesvd;
    cusolverDnDgesvd_bufferSize(cusolverH, nrow, ncol, &lwork_gesvd);

    int lwork = lwork_gesvd;

    printf("lwork size = %d\n",lwork);
    double *dwork;
    err = cudaMalloc((void **)&dwork, sizeof(double) * lwork);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");

    char jobu('A');
    char jobvt('A');

    cusolverStatus_t serr;
    serr = cusolverDnDgesvd(cusolverH, jobu, jobvt, nrow, ncol, dA, nrow, dS, dU, nrow, dVT, ncol, dwork, lwork, nullptr, dinfo);
    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgesvd failed"  + std::to_string(serr));

    err = cudaMemcpy(S, dS, sizeof(double)*min_mn, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy S_ptr");

    for (int i = 0; i < ncol; i++) {
        S[i] = 1.0/S[i];
    }

    double* tmp_ptr  = new double[nrow];
    double* dtmp;
    err = cudaMalloc((void **)&dtmp, sizeof(double) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtmp");

    // u.T * b
    double one(1.0);
    double zero(0.0);
    int incx(1);
    cublasDgemv(cublasH, CUBLAS_OP_T, nrow, nrow, &one, dU, nrow, db, incx, &zero, dtmp, incx);

    err = cudaMemcpy(tmp_ptr, dtmp, sizeof(double)*ncol, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy tmp_ptr to host");

    // S^-1 * (u.T * b)
    for (int i = 0; i < ncol; i++) {
        tmp_ptr[i] *= S[i];
    }

    err = cudaMemcpy(dtmp, tmp_ptr, sizeof(double)*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy tmp_ptr to device");

    // v.T * (S^-1 * (u.T * b))
    cublasDgemv(cublasH, CUBLAS_OP_T, ncol, ncol, &one, dVT, ncol, dtmp, incx, &zero, db, incx);

    cudaMemcpy(result_ptr, db, sizeof(double)*ncol, cudaMemcpyDeviceToHost);

    cudaFree(dtmp);
    cudaFree(dwork);
    cudaFree(dVT);
    cudaFree(dU);
    cudaFree(dS);
    cudaFree(db);
    cudaFree(dA);

    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

    delete[] S;
    delete[] tmp_ptr;
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
