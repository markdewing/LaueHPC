
#include "solve_cuda.h"
#include "perf_info.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdexcept>


template<typename T>
void solve_cuda_QR(int nrow, int ncol, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    int min_mn = nrow < ncol ? nrow : ncol;

    cudaError_t err;
    T* dA;
    err = cudaMalloc((void **)&dA, sizeof(T) * nrow * ncol);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T *dtau;
    err = cudaMalloc((void **)&dtau, sizeof(T) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtau");

    T* db;
    err = cudaMalloc((void **)&db, sizeof(T) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    int info;
    int *dinfo;
    err = cudaMalloc((void **)&dinfo, sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dinfo");


    err = cudaMemcpy(dA, A_ptr, sizeof(T)*nrow*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db, b_ptr, sizeof(T)*nrow, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");

    int lwork_geqrf;
    int lwork_ormqr;
    if constexpr(std::is_same<T,double>()) {
        cusolverDnDgeqrf_bufferSize(cusolverH, nrow, ncol, dA, nrow, &lwork_geqrf);
        cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA, nrow, dtau, db, nrow, &lwork_ormqr);
    }
    if constexpr(std::is_same<T,float>()) {
        cusolverDnSgeqrf_bufferSize(cusolverH, nrow, ncol, dA, nrow, &lwork_geqrf);
        cusolverDnSormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA, nrow, dtau, db, nrow, &lwork_ormqr);
    }


    int lwork = std::max(lwork_geqrf, lwork_ormqr);

    printf("lwork size = %d\n",lwork);
    T *dwork;
    err = cudaMalloc((void **)&dwork, sizeof(T) * lwork);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");


    cusolverStatus_t serr;
    if constexpr(std::is_same<T,double>())
        serr = cusolverDnDgeqrf(cusolverH, nrow, ncol, dA, nrow, dtau, dwork, lwork, dinfo);
    if constexpr(std::is_same<T,float>())
        serr = cusolverDnSgeqrf(cusolverH, nrow, ncol, dA, nrow, dtau, dwork, lwork, dinfo);

    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgeqrf failed");
    cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0)
     std::runtime_error(std::string("dgeqrf info = ") + std::to_string(info));

    if constexpr(std::is_same<T,double>())
        serr = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, ncol, dA, nrow, dtau, db, nrow, dwork, lwork, dinfo);
    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgeqrf failed");
    cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0)
        std::runtime_error(std::string("dormqr info = ") + std::to_string(info));

    T one(1.0);
    if constexpr(std::is_same<T,double>())
        cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncol, 1, &one, dA, nrow, db, nrow);

    cudaMemcpy(result_ptr, db, sizeof(T)*ncol, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(dA);
    cudaFree(dtau);
    cudaFree(db);
    cudaFree(dinfo);
    cudaFree(dwork);
}

template void solve_cuda_QR<double>(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);
template void solve_cuda_QR<float>(int nrow, int ncol, float* A_ptr, float* b_ptr, float* result_ptr, PerfInfo& perf);


template<typename T>
void solve_cuda_SVD(int nrow, int ncol, T* A_ptr, T* b_ptr, T* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    int min_mn = nrow < ncol ? nrow : ncol;

    cudaError_t err;
    T* dA;
    err = cudaMalloc((void **)&dA, sizeof(T) * nrow * ncol);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T* db;
    err = cudaMalloc((void **)&db, sizeof(T) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    int info;
    int *dinfo;
    err = cudaMalloc((void **)&dinfo, sizeof(int));
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dinfo");

    T* S = new T[min_mn];

    T* dS;
    err = cudaMalloc((void **)&dS, sizeof(T) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dS");

    T* dU;
    err = cudaMalloc((void **)&dU, sizeof(T) * nrow * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dU");

    T* dVT;
    err = cudaMalloc((void **)&dVT, sizeof(T) * ncol * ncol);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dVT");



    err = cudaMemcpy(dA, A_ptr, sizeof(T)*nrow*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db, b_ptr, sizeof(T)*nrow, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");


    int lwork_gesvd;
    if constexpr(std::is_same<T,double>())
        cusolverDnDgesvd_bufferSize(cusolverH, nrow, ncol, &lwork_gesvd);
    if constexpr(std::is_same<T,float>())
        cusolverDnSgesvd_bufferSize(cusolverH, nrow, ncol, &lwork_gesvd);

    int lwork = lwork_gesvd;

    printf("lwork size = %d\n",lwork);
    T *dwork;
    err = cudaMalloc((void **)&dwork, sizeof(T) * lwork);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");

    char jobu('A');
    char jobvt('A');

    cusolverStatus_t serr;
    if constexpr(std::is_same<T,double>())
        serr = cusolverDnDgesvd(cusolverH, jobu, jobvt, nrow, ncol, dA, nrow, dS, dU, nrow, dVT, ncol, dwork, lwork, nullptr, dinfo);
    if constexpr(std::is_same<T,float>())
        serr = cusolverDnSgesvd(cusolverH, jobu, jobvt, nrow, ncol, dA, nrow, dS, dU, nrow, dVT, ncol, dwork, lwork, nullptr, dinfo);

    if (serr != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnDgesvd failed"  + std::to_string(serr));

    err = cudaMemcpy(S, dS, sizeof(T)*min_mn, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy S_ptr");

    for (int i = 0; i < ncol; i++) {
        S[i] = 1.0/S[i];
    }

    T* tmp_ptr  = new T[nrow];
    T* dtmp;
    err = cudaMalloc((void **)&dtmp, sizeof(T) * nrow);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtmp");

    // u.T * b
    T one(1.0);
    T zero(0.0);
    int incx(1);
    if constexpr(std::is_same<T,double>())
        cublasDgemv(cublasH, CUBLAS_OP_T, nrow, nrow, &one, dU, nrow, db, incx, &zero, dtmp, incx);
    if constexpr(std::is_same<T,float>())
        cublasSgemv(cublasH, CUBLAS_OP_T, nrow, nrow, &one, dU, nrow, db, incx, &zero, dtmp, incx);

    err = cudaMemcpy(tmp_ptr, dtmp, sizeof(T)*ncol, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy tmp_ptr to host");

    // S^-1 * (u.T * b)
    for (int i = 0; i < ncol; i++) {
        tmp_ptr[i] *= S[i];
    }

    err = cudaMemcpy(dtmp, tmp_ptr, sizeof(T)*ncol, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy tmp_ptr to device");

    // v.T * (S^-1 * (u.T * b))
    if constexpr(std::is_same<T,double>())
        cublasDgemv(cublasH, CUBLAS_OP_T, ncol, ncol, &one, dVT, ncol, dtmp, incx, &zero, db, incx);
    if constexpr(std::is_same<T,float>())
        cublasSgemv(cublasH, CUBLAS_OP_T, ncol, ncol, &one, dVT, ncol, dtmp, incx, &zero, db, incx);

    cudaMemcpy(result_ptr, db, sizeof(T)*ncol, cudaMemcpyDeviceToHost);

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

template void solve_cuda_SVD<double>(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf);
template void solve_cuda_SVD<float>(int nrow, int ncol, float* A_ptr, float* b_ptr, float* result_ptr, PerfInfo& perf);

template<typename T>
void solve_batch_cuda_QR(int nrow, int ncol, int nbatch, T* A_batch_ptr, T* b_batch_ptr, T* result_batch_ptr, PerfInfo& perf)
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
    T* dA_batch;
    err = cudaMalloc((void **)&dA_batch, sizeof(T) * nrow * ncol * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T* dtau;
    err = cudaMalloc((void **)&dtau, sizeof(T) * min_mn);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dtau");

    T* db_batch;
    err = cudaMalloc((void **)&db_batch, sizeof(T) * nrow * nbatch);
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
        err = cudaMemcpyAsync(dA_batch + ib*nrow*ncol, A_batch_ptr, sizeof(T)*nrow*ncol, cudaMemcpyHostToDevice, streams[ib]);
        if (err != cudaSuccess)
            throw std::runtime_error("failed to copy dA");

        err = cudaMemcpyAsync(db_batch + ib*nrow , b_batch_ptr, sizeof(T)*nrow, cudaMemcpyHostToDevice, streams[ib]);
        if (err != cudaSuccess)
            throw std::runtime_error("failed to copy db");
    }
#endif


    int lwork_geqrf;
    int lwork_ormqr;
    if constexpr(std::is_same<T,double>()) {
        cusolverDnDgeqrf_bufferSize(cusolverH[0], nrow, ncol, dA_batch, nrow, &lwork_geqrf);
        cusolverDnDormqr_bufferSize(cusolverH[0], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA_batch, nrow, dtau, db_batch, nrow, &lwork_ormqr);
    }

    if constexpr(std::is_same<T,float>()) {
        cusolverDnSgeqrf_bufferSize(cusolverH[0], nrow, ncol, dA_batch, nrow, &lwork_geqrf);
        cusolverDnSormqr_bufferSize(cusolverH[0], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, nrow, dA_batch, nrow, dtau, db_batch, nrow, &lwork_ormqr);
    }

    int lwork = std::max(lwork_geqrf, lwork_ormqr);

    printf("lwork size = %d\n",lwork);
    T* dwork;
    err = cudaMalloc((void **)&dwork, sizeof(T) * lwork * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate dwork");


// Try launching streams in multiple threads
#pragma omp parallel for
    for (int ib = 0; ib < nbatch; ib++) {
        T* dA = dA_batch + ib*nrow*ncol;
        T* db = db_batch + ib*nrow;
        cusolverStatus_t serr;

        if constexpr(std::is_same<T,double>())
            serr = cusolverDnDgeqrf(cusolverH[ib], nrow, ncol, dA, nrow, dtau, dwork + lwork*ib, lwork, dinfo);
        if constexpr(std::is_same<T,float>())
            serr = cusolverDnSgeqrf(cusolverH[ib], nrow, ncol, dA, nrow, dtau, dwork + lwork*ib, lwork, dinfo);

        if (serr != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgeqrf failed");
        //cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, streams[ib]);
        //if (info != 0)
        // std::runtime_error(std::string("dgeqrf info = ") + std::to_string(info));

        if constexpr(std::is_same<T,double>())
            serr = cusolverDnDormqr(cusolverH[ib], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, ncol, dA, nrow, dtau, db, nrow, dwork + lwork*ib, lwork, dinfo);
        if constexpr(std::is_same<T,float>())
            serr = cusolverDnSormqr(cusolverH[ib], CUBLAS_SIDE_LEFT, CUBLAS_OP_T, nrow, 1, ncol, dA, nrow, dtau, db, nrow, dwork + lwork*ib, lwork, dinfo);
        if (serr != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cusolverDnDgeqrf failed");
        //cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, streams[ib]);
        //if (info != 0)
        //    std::runtime_error(std::string("dormqr info = ") + std::to_string(info));

        T one(1.0);
        if constexpr(std::is_same<T,double>())
            cublasDtrsm(cublasH[ib], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncol, 1, &one, dA, nrow, db, nrow);
        if constexpr(std::is_same<T,float>())
            cublasStrsm(cublasH[ib], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, ncol, 1, &one, dA, nrow, db, nrow);

        cudaMemcpyAsync(result_batch_ptr + ib*ncol, db, sizeof(T)*ncol, cudaMemcpyDeviceToHost, streams[ib]);
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

template void solve_batch_cuda_QR<double>(int nrow, int ncol, int nbatch, double* A_batch_ptr, double* b_batch_ptr, double* result_batch_ptr, PerfInfo& perf);
template void solve_batch_cuda_QR<float>(int nrow, int ncol, int nbatch, float* A_batch_ptr, float* b_batch_ptr, float* result_batch_ptr, PerfInfo& perf);

template<typename T>
void solve_batch_cuda_LS(int nrow, int ncol, int nbatch, T* A_batch_ptr, T* b_batch_ptr, T* result_batch_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);

    cublasHandle_t cublasH = nullptr;
    cublasCreate(&cublasH);

    cudaError_t err;
    T* dA_batch;
    err = cudaMalloc((void **)&dA_batch, sizeof(T) * nrow * ncol * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T** dA_ptrs;
    err = cudaMalloc((void ***)&dA_ptrs, sizeof(T*) * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dA_ptrs: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T** A_ptrs = new T*[nbatch];
    for (int i = 0; i < nbatch; i++) {
        A_ptrs[i] = dA_batch + i*nrow*ncol;
    }

    err = cudaMemcpy(dA_ptrs, A_ptrs, sizeof(T*)*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    T* db_batch;
    err = cudaMalloc((void **)&db_batch, sizeof(T) * nrow * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to allocate db");

    T** db_ptrs;
    err = cudaMalloc((void ***)&db_ptrs, sizeof(T*) * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate db_ptrs: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    T** b_ptrs = new T*[nbatch];
    for (int i = 0; i < nbatch; i++) {
        b_ptrs[i] = db_batch + i*nrow;
    }

    err = cudaMemcpy(db_ptrs, b_ptrs, sizeof(T*)*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy b_ptrs");

    err = cudaMemcpy(dA_batch, A_batch_ptr, sizeof(T)*nrow*ncol*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy dA");

    err = cudaMemcpy(db_batch, b_batch_ptr, sizeof(T)*nrow*nbatch, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error("failed to copy db");

    int* dinfos;
    err = cudaMalloc((void **)&dinfos, sizeof(int) * nbatch);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("failed to allocate dinfos: ") + cudaGetErrorName(err) + cudaGetErrorString(err));

    int info;
    int nrhs = 1;
    if constexpr(std::is_same<T,double>())
        cublasDgelsBatched(cublasH, CUBLAS_OP_N, nrow, ncol, nrhs, dA_ptrs, nrow, db_ptrs, nrow, &info, dinfos, nbatch);
    if constexpr(std::is_same<T,float>())
        cublasSgelsBatched(cublasH, CUBLAS_OP_N, nrow, ncol, nrhs, dA_ptrs, nrow, db_ptrs, nrow, &info, dinfos, nbatch);

    if (info != 0)
        throw std::runtime_error("cublasDgelsBatched info not zero : " + std::to_string(info));
    for (int ib = 0; ib < nbatch; ib++)
    {
        cudaMemcpy(result_batch_ptr + ib*ncol, db_batch, sizeof(T)*ncol, cudaMemcpyDeviceToHost);
    }
}

template void solve_batch_cuda_LS<double>(int nrow, int ncol, int nbatch, double* A_batch_ptr, double* b_batch_ptr, double* result_batch_ptr, PerfInfo& perf);
template void solve_batch_cuda_LS<float>(int nrow, int ncol, int nbatch, float* A_batch_ptr, float* b_batch_ptr, float* result_batch_ptr, PerfInfo& perf);
