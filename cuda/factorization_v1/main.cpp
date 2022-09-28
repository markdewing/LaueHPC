#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include "pm_cuda.h"

// ToDo:
// 0. move preprocess flags to command-line
// 1. extend to non-square matrices
// 2. profile data transfer
// 3. support for npy matrices
// 4. support for cuSolverSP and batched QR
// 5. support for single-precision
// 6. multi-gpu support
// 7. support for MAGMA and batched QR

#define _NUM_ITER 10

#define _DIM 800
#define _SIZE_BATCH 2048

extern int dev_solve(int, double *, double *, double *, const char *);
extern int dev_solve_batch(int, int, double *, double *, double *, const char *);

extern "C" {
  void dgemv_(const char * trans, const int * m, const int * n, const double * alpha, double * A,
	      const int * lda, double * x, const int * incx, const double * beta, double * y, const int * incy);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	      const double * alpha, double * A, const int * lda, double * b, const int * ldb, const double * beta,
	      double * c, const int * ldc);
}

void solve(int num_iter, int dim, double * d_A, double * d_x, double * x, double * d_b, double * x_ref, const char * mode)
{
  printf("\nLaunching solve(%s)\n",mode);
  
  int time_solve = 0;
  double time = 0.0;
  
  for(int iter=-1; iter<num_iter; ++iter) {
    
    double t0 = MPI_Wtime();
    
    int _time_solve = dev_solve(dim, d_A, d_x, d_b, mode);
    dev_pull(d_x, x, dim*sizeof(double));
    
    double t1 = MPI_Wtime();
    
    if(iter > -1) {
      time += t1 - t0;
      time_solve += _time_solve;
      
      double diff = 0.0;
      for(int i=0; i<dim; ++i) diff += (x[i] - x_ref[i]) * (x[i] - x_ref[i]);
      printf("diff= %f  time= %f ms  time_solve= %f us\n",diff,time*1000.0/(iter+1), (double) time_solve/(iter+1));
      
      //      for(int i=0; i<dim; ++i) printf("i= %i  x= %f\n",i,x[i]);
    }
  }

  double time_per_mil = (double) time_solve / num_iter / 1000 / 1000 * 1000000; // seconds
  printf("Time per million solves= %f s\n",time_per_mil);
}

void solve_batch(int num_iter, int size_batch, int dim, double * A, double * x, double * b, double * x_ref, const char * mode)
{
  printf("\nLaunching solve_batch(%s)\n",mode);
  
  int time_solve = 0;
  double time = 0.0;
  
  for(int iter=-1; iter<num_iter; ++iter) {
    
    double t0 = MPI_Wtime();
    
    int _time_solve = dev_solve_batch(size_batch, dim, A, x, b, mode);
    
    double t1 = MPI_Wtime();
    
    if(iter > -1) {
      time += t1 - t0;
      time_solve += _time_solve;
      
      double diff = 0.0;
      for(int i=0; i<size_batch*dim; ++i) diff += (x[i] - x_ref[i]) * (x[i] - x_ref[i]);
      printf("diff= %f  time= %f ms  time_solve= %f us\n",diff,time*1000.0/(iter+1), (double) time_solve/(iter+1));
      
      //      for(int i=0; i<size_batch*dim; ++i) printf("i= %i  x= %f\n",i,x[i]);
    }
  }
  
  double time_per_mil = (double) time_solve / num_iter / 1000 / 1000 * 1000000 / size_batch; // seconds
  printf("Time per million solves= %f s\n",time_per_mil);
}

int main(int argc, char *argv[])
{
  // Initialize MPI
  
  int rnk, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;
  
  MPI_Comm_rank(world, &rnk);
  MPI_Comm_size(world, &nprocs);

  // Initialize gpu

  int num_devices = dev_num_devices();
  printf(" # of devices detected= %i\n",num_devices);
  
  int ngpus = 1;
  printf(" Running on %i GPUs!\n",ngpus);
  
  int gpu_id = 0;
  dev_set_device(gpu_id);
  
  // Initialize data (to be replaced by CLI arguments)
  
  int num_iter = _NUM_ITER;
  int dim = _DIM;
  int nrow = _DIM;
  int ncol = _DIM;
  int lda = _DIM;
  int ldb = _DIM;
  int ldc = _DIM;
  int size_batch = _SIZE_BATCH;

  size_t size_A = dim * dim * sizeof(double);
  size_t size_b = dim * sizeof(double);

  printf("\nKernel w/ num_iter= %i  size_batch= %i  nrow,ncol= %i x %i\n", num_iter, size_batch, nrow, ncol);
  printf("  size_A= %lu (%f GB)  total= %f GB\n",size_A, (double)size_A / 1024/1024/1024, (double) size_batch * size_A / 1024/1024/1024);
  printf("  size_b= %lu (%f GB)  total= %f GB\n",size_b, (double)size_b / 1024/1024/1024, (double) size_batch * size_b / 1024/1024/1024);
  printf("  Host memory allocated: %f GB\n\n",(double) (size_A + 3 * size_b) * size_batch / 1024/1024/1024);
  
  double * A = (double *) malloc(size_batch * size_A);
  double * x = (double *) malloc(size_batch * size_b);
  double * b = (double *) malloc(size_batch * size_b);
  
  double * x_ref = (double *) malloc(size_batch * size_b);

#if 0 // Diagonal square matrix A
  bool symmetric = true;
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * dim*dim;
    int indx_b = ibatch * dim;
    
    for(int irow=0; irow<dim; ++irow) {
      for(int icol=0; icol<dim; ++icol) {
	if(irow == icol) A[indx_A + irow*dim+icol] = irow + 1;
	else A[indx_A + irow*dim+icol] = 0.0;
      }
      x[indx_b + irow] = 0.0;
      b[indx_b + irow] = irow + 1;

      x_ref[indx_b + irow] = 1.0;
    }

  }
#endif
  
#if 0 // Diagonal square matrix A
  bool symmetric = true;
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * dim*dim;
    int indx_b = ibatch * dim;
    
    for(int irow=0; irow<dim; ++irow) {
      for(int icol=0; icol<dim; ++icol) {
	if(irow == icol) A[indx_A + irow*dim+icol] = irow + 1;
	else A[indx_A + irow*dim+icol] = 0.0;
      }

      x_ref[indx_b + irow] = 1.0;
    }

    // b = A.x
    const double alpha = 1.0;
    const int incx = 1;
    const double beta = 0.0;
    const int incy = 1;
    dgemv_((char*) "N", &nrow, &ncol, &alpha, &A[indx_A], &lda, &x_ref[indx_b], &incx, &beta, &b[indx_b], &incy);
  }
#endif

  srand(12345 + rnk);
  
#if 1 // Random positive-definite square matrix A; first {A,b} is duplicated
  printf("size_A= %lu\n",size_A);
  double * Acopy = (double *) malloc(size_A);
  
  bool symmetric = true;
  for(int irow=0; irow<dim; ++irow) {
    for(int icol=irow; icol<dim; ++icol) {
      double value = rand() % 100 / 50.0;
      Acopy[irow*dim+icol] = value;
      Acopy[icol*dim+irow] = value;
    }
    
    x_ref[irow] = rand() % 100 / 50.0;
  }
  
  const double alpha = 1.0;
  const int incx = 1;
  const double beta = 0.0;
  const int incy = 1;
  
  // A = A.A^T
  dgemm_((char *) "N", (char *) "N", &nrow, &ncol, &ncol, &alpha, Acopy, &lda, Acopy, &ldb, &beta, A, &ldc);
  
  // b = A.x
  dgemv_((char*) "N", &nrow, &ncol, &alpha, A, &lda, x_ref, &incx, &beta, b, &incy);

  // copy A and b to create batch
  
  for(int ibatch=1; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * dim*dim;
    int indx_b = ibatch * dim;
    
    for(int i=0; i<nrow*ncol; ++i) A[indx_A + i] = A[i];
    for(int i=0; i<nrow; ++i) b[indx_b + i] = b[i];
    for(int i=0; i<nrow; ++i) x_ref[indx_b + i] = x_ref[i];
  }
  free(Acopy);
#endif
  
#if 0 // Random positive-definite square matrix A; every {A,b} is unique
  printf("size_A= %lu\n",size_A);
  double * Acopy = (double *) malloc(size_A);
  
  bool symmetric = true;
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * dim*dim;
    int indx_b = ibatch * dim;
    
    for(int irow=0; irow<dim; ++irow) {
      for(int icol=irow; icol<dim; ++icol) {
	double value = rand() % 100 / 50.0;
	Acopy[irow*dim+icol] = value;
	Acopy[icol*dim+irow] = value;
      }

      x_ref[indx_b + irow] = rand() % 100 / 50.0;
    }

    const double alpha = 1.0;
    const int incx = 1;
    const double beta = 0.0;
    const int incy = 1;
    
    // A = A.A^T
    dgemm_((char *) "N", (char *) "N", &nrow, &ncol, &ncol, &alpha, Acopy, &lda, Acopy, &ldb, &beta, &A[indx_A], &ldc);
    
    
    // b = A.x
    dgemv_((char*) "N", &nrow, &ncol, &alpha, &A[indx_A], &lda, &x_ref[indx_b], &incx, &beta, &b[indx_b], &incy);
  }
  free(Acopy);
#endif

#if 0
  printf("Matrix A[0]= \n");
  for(int irow=0; irow<dim; ++irow) {
    for(int icol=0; icol<dim; ++icol) printf(" %f",A[irow*dim+icol]);
    printf("\n");
  }

  printf("Vector b[0]= \n");
  for(int irow=0; irow<dim; ++irow) {
    printf(" %f\n",b[irow]);
  }
  
  printf("Vector x_ref[0]= \n");
  for(int irow=0; irow<dim; ++irow) {
    printf(" %f\n",x_ref[irow]);
  }
#endif
  
  // push data to gpu
  
  double * d_A = (double *) dev_malloc(size_batch * size_A);
  dev_push(d_A, A, size_batch * size_A);
  
  double * d_x = (double *) dev_malloc(size_batch * size_b);
  
  double * d_b = (double *) dev_malloc(size_batch * size_b);
  dev_push(d_b, b, size_batch * size_b);

  // do something useful
  
  if(symmetric) solve(num_iter, dim, d_A, d_x, x, d_b, x_ref, "LU");
  
  if(symmetric) solve(num_iter, dim, d_A, d_x, x, d_b, x_ref, "CHOL");
  
  solve(num_iter, dim, d_A, d_x, x, d_b, x_ref, "QR");

  if(symmetric) solve_batch(num_iter, size_batch, dim, A, x, b, x_ref, "CHOL");

  // clean-up
  
  dev_free(d_b);
  dev_free(d_x);
  dev_free(d_A);

  free(x_ref);
  free(b);
  free(x);
  free(A);

  MPI_Finalize();
}
