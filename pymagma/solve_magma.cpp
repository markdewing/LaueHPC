
#include "solve_magma.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_dlapack.h"
#include <stdio.h>
#include <stdexcept>

void init_magma()
{
    magma_init();
    //magma_print_environment();
}

void fini_magma()
{
    magma_finalize();
}

void solve_gpu(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr)
{
    magma_device_t device = 0;
    magma_setdevice(device);
    magma_queue_t queue;
    magma_queue_create(device, &queue);

    int min_mn = nrow < ncol ? nrow : ncol;

    double* tau = new double[min_mn];

    double* dA;
    magma_int_t err;
    err = magma_dmalloc(&dA, nrow*ncol);
    if (err != MAGMA_SUCCESS)
        throw std::runtime_error("failed to allocate dA");

    magma_dsetvector(ncol*nrow, A_ptr, 1, dA, 1, queue);

    double* db;
    err = magma_dmalloc(&db, nrow);
    if (err != MAGMA_SUCCESS)
        throw std::runtime_error("failed to allocate db");

    magma_dsetvector(nrow, b_ptr, 1, db, 1, queue);

    magma_int_t nb = magma_get_dgeqrf_nb(nrow, ncol);
    magma_int_t size_dT = nb*(2*min_mn + std::ceil(ncol/32)*32);

    double* dT;
    err = magma_dmalloc(&dT, size_dT);
    if (err != MAGMA_SUCCESS)
        throw std::runtime_error("failed to allocate dT");

    magma_queue_sync(queue);
    magma_int_t info;
    info = magma_dgeqrf_gpu(nrow, ncol, dA, nrow, tau, dT, &info);
    if (info != 0)
        throw std::runtime_error(std::string("dgeqrf info = ") + std::to_string(info));

    magma_int_t lwork = -1;
    double hwork_query[1];

    // The call to dgeqrf_gpu structures the output differently from lapack.
    // Q is present in dA, but R is not. dT contains intermediate data to
    // make multiplying by Q faster, and to recreate R.
    // Calling dormqr works, but calling dtrsm will not.
    // There is a routine magma_dgeqrs_gpu that performs the solve.
    // The first operation is performs is calling magma_dormqr_gpu.  However,
    // that call segfaults, but if I call magma_dormqr_gpu myself, it does not.
    // The segfault is using an AMD GPU with rocm 5.1
    // With CUDA, it runs, but the answer is not correct.
#if 0

   // work size query
    magma_dormqr_gpu(MagmaLeft, MagmaTrans, nrow, 1, ncol, dA, nrow, tau, db, nrow, hwork_query, lwork, dT, nb, &info);
    if (info != 0)
        std::runtime_error(std::string("dormqr info = ") + std::to_string(info));

    lwork = hwork_query[0];
    printf("optimizal lwork = %d\n",lwork);

    double* hwork = new double[lwork];
    // real call
    magma_dormqr_gpu(MagmaLeft, MagmaTrans, nrow, 1, ncol, dA, nrow, tau, db, nrow, hwork, lwork, dT, nb, &info);
    if (info != 0)
        std::runtime_error(std::string("dormqr info = ") + std::to_string(info));


    // Copy some code from magma_dgeqrs_gpu, specialized to the case with 1 rhs.
    // unfinished
    char upper('U');
    char notrans('N');
    char nonunit('N');
    int ione(1);

    int ib = ncol - 1;

    dtrsv_(&upper, &notrans, &nonunit, &ib, hwork, &nrow, hwork +nrow*ib, &ione);

    double *dwork;
    if (nb < min_mn)
        dwork = dT + 2*min_mn*nb;
    else
        dwork = dT;

    int i = (min_mn -1)/nb * nb;
    magma_dsetmatrix(ib, 1, hwork+nrow*ib, nrow, dwork+i, min_mn, queue);

    magma_dgemv(MagamNoTrans, i, ib, -1.0, dA

#endif

    // This won't work because the upper part of dA is not set by magam_dgeqrf_gpu.
    //magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, ncol, 1, 1.0, dA, nrow, db, nrow, queue);
    //
#if 1
    // workspace query
    magma_dgeqrs_gpu(nrow, ncol, 1, dA, nrow, tau, db, dT, nrow, hwork_query, lwork, &info);
    if (info != 0)
        throw std::runtime_error(std::string("dgeqrs info = ") + std::to_string(info));

    lwork = hwork_query[0];
    printf("optimal lwork = %d\n",lwork);
    double* hwork = new double[lwork];
    // real call
    magma_dgeqrs_gpu(nrow, ncol, 1, dA, nrow, tau, db, dT, nrow, hwork, lwork, &info);
    if (info != 0)
        throw std::runtime_error(std::string("dgeqrs info = ") + std::to_string(info));
#endif

    magma_dgetvector(ncol, db, 1, result_ptr, 1, queue);
    magma_queue_sync(queue);

    for(int i = 0; i < 4; i++) {
        printf(" x %d %g\n",i,result_ptr[i]);
    }

    magma_free(dA);
    magma_free(db);
    magma_free(dT);

    delete[] hwork;
    delete[] tau;
}

// Solve on GPU using simplest Magma interfaces
void solve_gpu_simple(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr)
{
    double* tau = new double[ncol];
    int info;

    double work_query;
    int lwork = -1;
    magma_dgeqrf(nrow, ncol, A_ptr, nrow, tau, &work_query, lwork, &info);
    if (info != 0)
        printf("dgeqrf work query, info  = %d\n",info);
    lwork = int(work_query);
    printf("optimal lwork = %d\n",lwork);

    double* work = new double[lwork];

    magma_dgeqrf(nrow, ncol, A_ptr, nrow, tau, work, lwork, &info);
    if (info != 0)
        printf("dgeqrf info = %d\n",info);
    //printf("done with dgeqrf\n");

    int lwork1 = -1;
    magma_dormqr(MagmaLeft, MagmaTrans, ncol, 1, ncol, A_ptr, nrow, tau, b_ptr, nrow, &work_query, lwork1, &info);
    lwork1 = int(work_query);
    printf("optimal lwork = %d\n",lwork1);
    if (lwork1 > lwork) {
        delete work;
        double* work = new double[lwork1];
        lwork = lwork1;
    }

    magma_dormqr(MagmaLeft, MagmaTrans, nrow, 1, ncol, A_ptr, nrow, tau, b_ptr, nrow, work, lwork, &info);

    // dtrsm on GPU
#if 0
    // Need to set up data transfer to/from GPU
    magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, ncol, 1, 1.0, A_ptr, nrow, result_ptr, nrow, queue);
#endif

    // dtrsm on CPU
#if 1
    char left('L');
    char upper('U');
    char notrans('N');
    char nonunit('N');
    double one(1.0);
    int nrhs(1);
    dtrsm_(&left, &upper, &notrans, &nonunit, &ncol, &nrhs, &one, A_ptr, &nrow, b_ptr, &nrow);
#endif

    for (int i = 0; i < ncol; i++)
        result_ptr[i] = b_ptr[i];

    delete[] tau;
    delete[] work;
}

