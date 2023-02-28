
#include "solve_magma.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_dlapack.h"
#include <stdio.h>

void init_magma()
{
    magma_init();
    //magma_print_environment();
}

void fini_magma()
{
    magma_finalize();
}

void solve_cpu(int nrow, int ncol, double* A_ptr, double* b_ptr, double* result_ptr)
{

#if 0
    magma_device_t device = 0;
    magma_setdevice(device);
    magma_queue_t queue;
    magma_queue_create(device, &queue);
#endif

    double* tau = new double[ncol];
    int info;

    double work_query;
    int lwork = -1;
    //magma_dgeqrf(nrow, ncol, A_ptr, nrow, tau, &work_query, lwork, &info);
    dgeqrf_(&nrow, &ncol, A_ptr, &nrow, tau, &work_query, &lwork, &info);
    if (info != 0)
        printf("dgeqrf work query, info  = %d\n",info);
    lwork = int(work_query);
    printf("optimal lwork = %d\n",lwork);

    double* work = new double[lwork];

    //magma_dgeqrf(nrow, ncol, A_ptr, nrow, tau, work, lwork, &info);
    dgeqrf_(&nrow, &ncol, A_ptr, &nrow, tau, work, &lwork, &info);
    if (info != 0)
        printf("dgeqrf info = %d\n",info);
    printf("done with dgeqrf\n");
    //for (int i = 0; i < 4; i++) {
    //    printf(" tau: %d %g\n",i,tau[i]);
    //}

#if 0
    int lwork1 = -1;
    magma_dormqr(MagmaLeft, MagmaTrans, ncol, 1, ncol, A_ptr, nrow, tau, b_ptr, nrow, &work_query, lwork1, &info);
    lwork1 = int(work_query);
    printf("optimal lwork = %d\n",lwork1);
    if (lwork1 > lwork) {
        delete work;
        double* work = new double[lwork1];
        lwork = lwork1;
    }
#endif

    //magma_dormqr(MagmaLeft, MagmaTrans, nrow, 1, ncol, A_ptr, nrow, tau, b_ptr, nrow, work, lwork, &info);
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


    //magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaUnit, ncol, 1, 1.0, A_ptr, nrow, result_ptr, nrow, queue);
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
    //magma_queue_sync(queue);
}
