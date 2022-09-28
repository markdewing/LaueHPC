//https://www.netlib.org/lapack/lapacke.html

/* Calling DGELS using row-major order */

#include <stdio.h>
#include <stdlib.h>

#if 1
#include <lapacke.h>
#else
extern "C" {

  void dgels_(const char * trans, const int * m, const int * n, const int * nrhs, double * A, const int * lda, double * b,
	      const int * ldb, double * work, const int * lwork, const int * info);
  
}
#endif

int main (int argc, const char * argv[])
{
  printf("Reference A= \n");
  printf(" 1   1   1\n");
  printf(" 2   3   4\n");
  printf(" 3   5   2\n");
  printf(" 4   2   5\n");
  printf(" 5   4   3\n");

  printf("\nReference B= \n");
  printf(" -10  -3\n");
  printf("  12  14\n");
  printf("  14  12\n");
  printf("  16  16\n");
  printf("  18  16\n");
  
#if 1
  {
    printf("\nInitializing ROW_MAJOR example\n");
    
    double a[5][3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
    double b[5][2] = {-10,-3,12,14,14,12,16,16,18,16};
    lapack_int info,m,n,lda,ldb,nrhs;
    int i,j;

    double * ptr = &(a[0][0]);
    printf("a= ");
    for(int i=0; i<5*3; ++i) printf(" %f",ptr[i]);
    printf("\n");
    
    m = 5;
    n = 3;
    nrhs = 2;
    lda = 3;
    ldb = 2;
    
    info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*a,lda,*b,ldb);

    printf("n= %i  nrhs= %i  info= %i\n",n,nrhs,info);
    
    for(i=0;i<n;i++)
      {
	for(j=0;j<nrhs;j++)
	  {
	    printf("%lf ",b[i][j]);
	  }
	printf("\n");
      }

    printf("finished\n");
  }

  {
    printf("\nInitializing COL_MAJOR example\n");
    
    double a[5*3] = {1,2,3,4,5,1,3,5,2,4,1,4,2,5,3};
    double b[5*2] = {-10,12,14,16,18,-3,14,12,16,16};
    lapack_int info,m,n,lda,ldb,nrhs;
    int i,j;
    
    m = 5;
    n = 3;
    nrhs = 2;
    lda = 5;
    ldb = 5;
    
    info = LAPACKE_dgels(LAPACK_COL_MAJOR,'N',m,n,nrhs,a,lda,b,ldb);
    
    printf("n= %i  nrhs= %i  info= %i\n",n,nrhs,info);
    
    for(i=0;i<n;i++)
      {
	for(j=0;j<nrhs;j++)
	  {
	    printf("%lf ",b[i+ldb*j]);
	  }
	printf("\n");
      }
    
    printf("finished\n");
  }
#endif

#if 0
  {
    printf("\nInitializing ROW_MAJOR example\n");
    
    double a[5*3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
    double b[5*2] = {-10,-3,12,14,14,12,16,16,18,16};
    double * work = (double *) malloc(sizeof(double));
    int info,m,n,lda,ldb,nrhs,lwork;
    int i,j;
    
    m = 5;
    n = 3;
    nrhs = 2;
    lda = 3;
    ldb = 2;
    
    lwork = -1;

    
    
    dgels_((char *) "N", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    
    lwork = work[0];
    free(work);
    work = (double *) malloc(lwork*sizeof(double));
    
    dgels_((char *) "N", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    
    printf("n= %i  nrhs= %i  info= %i\n",n,nrhs,info);
    
    for(i=0;i<n;i++)
      {
	for(j=0;j<nrhs;j++)
	  {
	    printf("%lf ",b[i+ldb*j]);
	  }
	printf("\n");
      }
    
    printf("finished\n");
  }
  
  {
    printf("\nInitializing COL_MAJOR example\n");
    
    double a[5*3] = {1,2,3,4,5,1,3,5,2,4,1,4,2,5,3};
    double b[5*2] = {-10,12,14,16,18,-3,14,12,16,16};
    double * work = (double *) malloc(sizeof(double));
    int info,m,n,lda,ldb,nrhs,lwork;
    int i,j;
    
    m = 5;
    n = 3;
    nrhs = 2;
    lda = 5;
    ldb = 5;

    lwork = -1;
    
    dgels_((char *) "N", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);

    lwork = work[0];
    free(work);
    work = (double *) malloc(lwork*sizeof(double));
    
    dgels_((char *) "N", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    
    printf("n= %i  nrhs= %i  info= %i\n",n,nrhs,info);
    
    for(i=0;i<n;i++)
      {
	for(j=0;j<nrhs;j++)
	  {
	    printf("%lf ",b[i+ldb*j]);
	  }
	printf("\n");
      }
    
    printf("finished\n");
  }
#endif
}

/*
Initializing ROW_MAJOR example
n= 3  nrhs= 2  info= 0
2.000000 1.000000 
1.000000 1.000000 
1.000000 2.000000 
finished

Initializing COL_MAJOR example
n= 3  nrhs= 2  info= 0
2.000000 1.000000 
1.000000 1.000000 
1.000000 2.000000 
finished
*/
