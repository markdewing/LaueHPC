
#include "solve_cpu.h"
#include "perf_info.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_dlapack.h"
#include <stdio.h>
#include <stdexcept>
#include <chrono>

void solve_cpu_QR(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);
    double* tau = new double[ncol];
    int info;

    double work_query;
    int lwork = -1;
    dgeqrf_(&nrow, &ncol, A_ptr, &nrow, tau, &work_query, &lwork, &info);
    if (info != 0)
        printf("dgeqrf work query, info  = %d\n",info);
    lwork = int(work_query);
    printf("optimal lwork = %d\n",lwork);

    double* work = new double[lwork];

    dgeqrf_(&nrow, &ncol, A_ptr, &nrow, tau, work, &lwork, &info);
    if (info != 0)
        printf("dgeqrf info = %d\n",info);
    //for (int i = 0; i < 4; i++) {
    //    printf(" tau: %d %g\n",i,tau[i]);
    //}

    char side('L');
    char trans('T');
    int nrhs(1);
    dormqr_(&side, &trans, &nrow, &nrhs, &ncol, A_ptr, &nrow, tau, b_ptr, &nrow, work, &lwork, &info);
    if (info != 0)
        printf("dormqr info = %d\n",info);
    printf("done with dormqr\n");
    //for (int i = 0; i < 4; i++) {
    //    printf(" %d %g\n",i,b_ptr[i]);
    //}


    char left('L');
    char upper('U');
    char notrans('N');
    char nonunit('N');
    double one(1.0);
    dtrsm_(&left, &upper, &notrans, &nonunit, &ncol, &nrhs, &one, A_ptr, &nrow, b_ptr, &nrow);

    for (int i = 0; i < ncol; i++)
        result_ptr[i] = b_ptr[i];

    delete[] tau;
    delete[] work;
}

void solve_cpu_SVD(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);
    char jobu('A');
    char jobvt('A');

    int min_mn = nrow < ncol ? nrow : ncol;
    printf("nrow = %d ncol = %d\n",nrow,ncol);

    double* U_ptr =  new double[nrow * nrow];
    double* VT_ptr = new double[ncol * ncol];
    double* S_ptr = new double[min_mn];


    int lwork = -1;
    double work_query[1];
    int info;
    // Work query
    dgesvd_(&jobu, &jobvt, &nrow, &ncol, A_ptr, &nrow, S_ptr, U_ptr, &nrow, VT_ptr, &ncol, work_query, &lwork, &info);

    if (info != 0)
        throw std::runtime_error(std::string("Non zero info from degsvd work query: ")+ std::to_string(info));

    RecordComp record_dgesvd(perf, 0);
    lwork = work_query[0];
    printf("lwork for dgesvd = %d\n",lwork);
    double *work = new double[lwork];
    // actual call
    dgesvd_(&jobu, &jobvt, &nrow, &ncol, A_ptr, &nrow, S_ptr, U_ptr, &nrow, VT_ptr, &ncol, work, &lwork, &info);

    if (info != 0)
        throw std::runtime_error(std::string("Non zero info from degsvd: ")+ std::to_string(info));

    record_dgesvd.stop();

    for (int i = 0; i < ncol; i++)
        S_ptr[i] = 1.0/S_ptr[i];

    RecordComp record_dgemv(perf, 1);
    double* tmp_ptr =  new double[nrow];
    // u.T * b
    char trans('T');
    double one(1.0);
    double zero(0.0);
    int incx(1);
    dgemv_(&trans, &nrow, &nrow, &one, U_ptr, &nrow, b_ptr, &incx, &zero, tmp_ptr, &incx);

    // S^-1 * (u.T * b)
    for (int i = 0; i < ncol; i++)
        tmp_ptr[i] *= S_ptr[i];

    // vt.T * (S^-1 * (u.T * b))
    dgemv_(&trans, &ncol, &ncol, &one, VT_ptr, &ncol, tmp_ptr, &incx, &zero, result_ptr, &incx);
    record_dgemv.stop();

    delete[] tmp_ptr;
    delete[] work;
    delete[] S_ptr;
    delete[] VT_ptr;
    delete[] U_ptr;
}

void solve_cpu_LS(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);
    int info;

    char trans('N');
    int nrhs(1);

    double work_query;
    int lwork = -1;

    dgels_(&trans, &nrow, &ncol, &nrhs, A_ptr, &nrow, b_ptr, &nrow, &work_query, &lwork, &info);

    if (info != 0)
        printf("dgels work query, info  = %d\n",info);
    lwork = int(work_query);
    printf("optimal lwork = %d\n",lwork);

    double* work = new double[lwork];


    dgels_(&trans, &nrow, &ncol, &nrhs, A_ptr, &nrow, b_ptr, &nrow, work, &lwork, &info);
    if (info != 0)
        printf("dgels info = %d\n",info);

    for (int i = 0; i < ncol; i++)
        result_ptr[i] = b_ptr[i];

    delete[] work;
}

void solve_batch_cpu_QR(int nrow, int ncol, int nbatch, double* A_batch_ptr, double* b_batch_ptr, double* result_batch_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);
    double* tau = new double[ncol];
    int info;

    double work_query;
    int lwork = -1;
    dgeqrf_(&nrow, &ncol, A_batch_ptr, &nrow, tau, &work_query, &lwork, &info);
    if (info != 0)
        printf("dgeqrf work query, info  = %d\n",info);
    lwork = int(work_query);
    printf("optimal lwork = %d\n",lwork);

    double* work = new double[lwork];

    // Assume the work size is the same for each batch item

    for (int ib = 0; ib < nbatch; ib++) {
        double *A_ptr = A_batch_ptr + ib*nrow*ncol;
        double *b_ptr = b_batch_ptr + ib*nrow;
        double *result_ptr = result_batch_ptr + ib*ncol;

        dgeqrf_(&nrow, &ncol, A_ptr, &nrow, tau, work, &lwork, &info);
        if (info != 0)
            printf("dgeqrf info = %d\n",info);
        //for (int i = 0; i < 4; i++) {
        //    printf(" tau: %d %g\n",i,tau[i]);
        //}

        char side('L');
        char trans('T');
        int nrhs(1);
        dormqr_(&side, &trans, &nrow, &nrhs, &ncol, A_ptr, &nrow, tau, b_ptr, &nrow, work, &lwork, &info);
        if (info != 0)
            printf("dormqr info = %d\n",info);
        //printf("done with dormqr\n");
        //for (int i = 0; i < 4; i++) {
        //    printf(" %d %g\n",i,b_ptr[i]);
        //}


        char left('L');
        char upper('U');
        char notrans('N');
        char nonunit('N');
        double one(1.0);
        dtrsm_(&left, &upper, &notrans, &nonunit, &ncol, &nrhs, &one, A_ptr, &nrow, b_ptr, &nrow);

        for (int i = 0; i < ncol; i++)
            result_ptr[i] = b_ptr[i];
    }

    delete[] tau;
    delete[] work;
}

void solve_batch_cpu_SVD(int nrow, int ncol, int nbatch, double* A_batch_ptr, double* b_batch_ptr, double* result_batch_ptr, PerfInfo& perf)
{
    RecordElapsed recordElapsed(perf);
    char jobu('A');
    char jobvt('A');

    int min_mn = nrow < ncol ? nrow : ncol;
    printf("nrow = %d ncol = %d\n",nrow,ncol);

    double* U_ptr =  new double[nrow * nrow];
    double* VT_ptr = new double[ncol * ncol];
    double* S_ptr = new double[min_mn];


    int lwork = -1;
    double work_query[1];
    int info;
    // Work query
    dgesvd_(&jobu, &jobvt, &nrow, &ncol, A_batch_ptr, &nrow, S_ptr, U_ptr, &nrow, VT_ptr, &ncol, work_query, &lwork, &info);

    if (info != 0)
        throw std::runtime_error(std::string("Non zero info from degsvd work query: ")+ std::to_string(info));

    lwork = work_query[0];
    printf("lwork for dgesvd = %d\n",lwork);
    double *work = new double[lwork];

    double* tmp_ptr =  new double[nrow];

    for (int ib = 0; ib < nbatch; ib++)
    {
        double *A_ptr = A_batch_ptr + ib*nrow*ncol;
        double *b_ptr = b_batch_ptr + ib*nrow;
        double *result_ptr = result_batch_ptr + ib*ncol;

        // actual call
        dgesvd_(&jobu, &jobvt, &nrow, &ncol, A_ptr, &nrow, S_ptr, U_ptr, &nrow, VT_ptr, &ncol, work, &lwork, &info);

        if (info != 0)
            throw std::runtime_error(std::string("Non zero info from degsvd: ")+ std::to_string(info));

        for (int i = 0; i < ncol; i++)
            S_ptr[i] = 1.0/S_ptr[i];

        // u.T * b
        char trans('T');
        double one(1.0);
        double zero(0.0);
        int incx(1);
        dgemv_(&trans, &nrow, &nrow, &one, U_ptr, &nrow, b_ptr, &incx, &zero, tmp_ptr, &incx);

        // S^-1 * (u.T * b)
        for (int i = 0; i < ncol; i++)
            tmp_ptr[i] *= S_ptr[i];

        // vt.T * (S^-1 * (u.T * b))
        dgemv_(&trans, &ncol, &ncol, &one, VT_ptr, &ncol, tmp_ptr, &incx, &zero, result_ptr, &incx);
    }

    delete[] tmp_ptr;
    delete[] work;
    delete[] S_ptr;
    delete[] VT_ptr;
    delete[] U_ptr;
}
