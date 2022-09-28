#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <mpi.h>

#if 1
#include "cblas.h"
#endif

#if _CHECK_CPU
#include "lapacke.h"
#endif

#ifdef _USE_GPU
#include "pm_cuda.h"
#else
int dev_num_devices() { return 0; }
void dev_set_device(int) {}
void * dev_malloc(int) { return nullptr; }
void dev_free(void *) {};
void dev_push(void *, void *, int) {}
void solve(int, int, int, void *, void *, void *, void *, void *, const char *) {}
void solve_batch(int, int, int, int, void *, void *, void *, void *, const char *) {}
#endif

// ToDo:
// 0. move preprocess flags to command-line
// 1. column-major ordering DONE
// 2. extend to non-square matrices DONE
// 3. support for MAGMA and batched QR
// 4. profile data transfer
// 5. support for npy matrices
// 6. support for cuSolverSP and batched QR
// 7. support for single-precision
// 8. multi-gpu support

// 9. SVD, singular values

// 10. is x vector flipped?? No, Doga had [::-1] on solution returned.
// 11. consider padding matrices to have identical dimensions for batching?

// LAPACK, cuSolver, MAGMA all support column-major ordering


#define _NUM_ITER 10

// problem with this is that ALL matrices hard-coded to this...
#define _NUM_ROW 798
#define _NUM_COL 256

#define _SIZE_BATCH 1

#define _NUM_INPUT_MATRICES 1

#ifdef _USE_GPU
extern int dev_solve(int, int, double *, double *, double *, const char *);
extern int dev_solve_batch(int, int, int, double *, double *, double *, const char *);
#endif

extern "C" {
  void dgemv_(const char * trans, const int * m, const int * n, const double * alpha, double * A,
	      const int * lda, double * x, const int * incx, const double * beta, double * y, const int * incy);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	      const double * alpha, double * A, const int * lda, double * b, const int * ldb, const double * beta,
	      double * c, const int * ldc);
}

#if _CHECK_CPU
void solve_cpu(int nrow, int ncol, double * A, double * x, double * b, double * x_ref)
{
  printf("\nLaunching solve_cpu(LU)\n");

  int info;
  int nrhs = 1;
  int lda = nrow;

  int max = (nrow > ncol) ? nrow : ncol;
  double * tmp = (double *) malloc(max * sizeof(double));
  double * Acopy = (double *) malloc(nrow * ncol * sizeof(double));

  for(int imat=0; imat<_NUM_INPUT_MATRICES; ++imat) {
    int indx_A = imat * nrow * ncol;
    int indx_b = imat * nrow;
    int indx_x = imat * ncol;

    // make copies
    for(int irow=0; irow<nrow; ++irow) tmp[irow] = b[indx_b + irow];
    for(int i=0; i<nrow*ncol; ++i) Acopy[i] = A[indx_A + i];
    
    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', nrow, ncol, nrhs, Acopy, lda, tmp, nrow);
    
    // printf("solution= \n");
    // for(int icol=0; icol<ncol; ++icol) printf(" %f  %f\n",tmp[icol],x_ref[indx_x+icol]);
    
    {
      char fname[50];
      sprintf(fname,"soln.%i.txt",imat);
      std::ofstream outfile(fname);
      for(int icol=0; icol<ncol; ++icol) outfile << tmp[icol] << "\n";
      
      outfile.close();
    }

    for(int i=0; i<ncol; ++i) x[indx_x + i] = tmp[i];
    
    double diff = 0.0;
    for(int i=0; i<ncol; ++i) diff += (x[indx_x+i] - x_ref[indx_x+i]) * (x[indx_x+i] - x_ref[indx_x+i]);

    // residual with reference solution
    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    const double alpha = 1.0;
    const double beta = 0.0;
    const int incx = 1;
    const int incy = 1;

    // residual with reference solution
    cblas_dgemv(layout, trans, nrow, ncol, alpha, &(A[indx_A]), lda, &(x_ref[indx_x]), incx, beta, tmp, incy);
    for(int irow=0; irow<nrow; ++irow) tmp[irow] -= b[indx_b + irow];

    double residual_ref = 0.0;
    for(int irow=0; irow<nrow; ++irow) residual_ref += tmp[irow] * tmp[irow];

    // residual with computed solution
    dgemv_((char*) "N", &nrow, &ncol, &alpha, &(A[indx_A]), &lda, &(x[indx_x]), &incx, &beta, tmp, &incy);
    for(int irow=0; irow<nrow; ++irow) tmp[irow] -= b[indx_b + irow];

    double residual = 0.0;
    for(int irow=0; irow<nrow; ++irow) residual += tmp[irow] * tmp[irow];

    printf("imat= %i  soln. diff= %f  residual_ref= %f  residual= %f\n",imat, diff, sqrt(residual_ref), sqrt(residual));
  }

  free(tmp);
  free(Acopy);
}
#endif


#ifdef _USE_GPU
void solve(int num_iter, int nrow, int ncol, double * d_A, double * d_x, double * x, double * d_b, double * x_ref, const char * mode)
{
  printf("\nLaunching solve(%s)\n",mode);
  
  int time_solve = 0;
  double time = 0.0;

  dev_pull(d_x, x, ncol*sizeof(double));

  printf("starting loop\n");
  
  for(int iter=-1; iter<num_iter; ++iter) {
    
    double t0 = MPI_Wtime();
    
    int _time_solve = dev_solve(nrow, ncol, d_A, d_x, d_b, mode);
    dev_pull(d_x, x, ncol*sizeof(double));
    
    double t1 = MPI_Wtime();
    
    if(iter > -1) {
      time += t1 - t0;
      time_solve += _time_solve;
      
      double diff = 0.0;
      for(int i=0; i<ncol; ++i) diff += (x[i] - x_ref[i]) * (x[i] - x_ref[i]);
      printf("diff= %f  time= %f ms  time_solve= %f us\n",diff,time*1000.0/(iter+1), (double) time_solve/(iter+1));
      
      //      for(int i=0; i<ncol; ++i) printf("i= %i  x= %f\n",i,x[i]);
    }
  }

  double time_per_mil = (double) time_solve / num_iter / 1000 / 1000 * 1000000; // seconds
  printf("Time per million solves= %f s\n",time_per_mil);
}

void solve_batch(int num_iter, int size_batch, int nrow, int ncol, double * A, double * x, double * b, double * x_ref, const char * mode)
{
  printf("\nLaunching solve_batch(%s)\n",mode);
  
  int time_solve = 0;
  double time = 0.0;
  
  for(int iter=-1; iter<num_iter; ++iter) {
    
    double t0 = MPI_Wtime();
    
    int _time_solve = dev_solve_batch(size_batch, nrow, ncol, A, x, b, mode);
    
    double t1 = MPI_Wtime();
    
    if(iter > -1) {
      time += t1 - t0;
      time_solve += _time_solve;
      
      double diff = 0.0;
      for(int i=0; i<size_batch*ncol; ++i) diff += (x[i] - x_ref[i]) * (x[i] - x_ref[i]);
      printf("diff= %f  time= %f ms  time_solve= %f us\n",diff,time*1000.0/(iter+1), (double) time_solve/(iter+1));
      
      //      for(int i=0; i<size_batch*nrow; ++i) printf("i= %i  x= %f\n",i,x[i]);
    }
  }
  
  double time_per_mil = (double) time_solve / num_iter / 1000 / 1000 * 1000000 / size_batch; // seconds
  printf("Time per million solves= %f s\n",time_per_mil);
}
#endif

void init_matrix_1(int nrow, int ncol, int size_batch, double * A, double * b, double * x, double * x_ref, bool &symmetric)
{
  symmetric = true;
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * nrow*ncol;
    int indx_b = ibatch * ncol;
    
    for(int irow=0; irow<nrow; ++irow) {
      for(int icol=0; icol<ncol; ++icol) {
	if(irow == icol) A[indx_A + irow*ncol+icol] = irow + 1;
	else A[indx_A + irow*ncol+icol] = 0.0;
      }
      b[indx_b + irow] = irow + 1;
    }
    
    int indx_x = ibatch * nrow;
    for(int icol=0; icol<ncol; ++icol) {
      x[indx_x + icol] = 0.0;
      x_ref[indx_x + icol] = 1.0;      
    }
    
  }
}

void init_matrix_2(int nrow, int ncol, int size_batch, double * A, double * b, double * x, double * x_ref, bool &symmetric)
{
  symmetric = true;
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * nrow*ncol;
    int indx_b = ibatch * nrow;
    
    for(int irow=0; irow<nrow; ++irow) {
      for(int icol=0; icol<ncol; ++icol) {
	if(irow == icol) A[indx_A + irow*ncol+icol] = irow + 1;
	else A[indx_A + irow*ncol+icol] = 0.0;
      }
    }

    int indx_x = ibatch * ncol;
    for(int icol=0; icol<ncol; ++icol) x_ref[indx_x + icol] = 1.0;

    // b = A.x
    const int lda = ncol;
    const double alpha = 1.0;
    const int incx = 1;
    const double beta = 0.0;
    const int incy = 1;
    dgemv_((char*) "N", &nrow, &ncol, &alpha, &A[indx_A], &lda, &x_ref[indx_b], &incx, &beta, &b[indx_b], &incy);
  }
}

void init_matrix_3(int nrow, int ncol, int size_batch, double * A, double * b, double * x, double * x_ref, bool &symmetric)
{  
  symmetric = (nrow == ncol);
  for(int icol=0; icol<ncol; ++icol) x_ref[icol] = rand() % 100 / 50.0;
  for(int irow=0; irow<nrow; ++irow) b[irow] = 0.0;

  const int lda = nrow;
  const int ldb = ncol;
  const int ldc = nrow;
  const double alpha = 1.0;
  const int incx = 1;
  const double beta = 0.0;
  const int incy = 1;

  if(symmetric) {
    
    for(int irow=0; irow<nrow; ++irow) {
      for(int icol=irow; icol<ncol; ++icol) {
	double value = rand() % 100 / 50.0;
	A[irow*ncol+icol] = value;
	A[icol*nrow+irow] = value;
      }
    }
    
  } else {
    
    for(int irow=0; irow<nrow; ++irow) {
      for(int icol=0; icol<ncol; ++icol) {
	double value = rand() % 100 / 50.0;
	A[irow*ncol+icol] = value;
      }
    }

  }
  
  // b = A.x
  
  for(int irow=0; irow<nrow; ++irow) {
    double sum = 0.0;
    for(int icol=0; icol<ncol; ++icol) sum += A[icol*nrow+irow] * x_ref[icol];
    b[irow] = sum;
  }
  
  // copy A and b to create batch
  
  for(int ibatch=1; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * nrow*ncol;
    int indx_b = ibatch * nrow;
    int indx_x = ibatch * ncol;
    
    for(int i=0; i<nrow*ncol; ++i) A[indx_A + i] = A[i];
    for(int i=0; i<nrow; ++i) b[indx_b + i] = b[i];
    for(int i=0; i<ncol; ++i) x_ref[indx_x + i] = x_ref[i];
  }
}

void init_matrix_4(int nrow, int ncol, int size_batch, double * A, double * b, double * x, double * x_ref, bool &symmetric)
{
  double * Acopy = (double *) malloc(nrow*ncol*sizeof(double));
  
  symmetric = (nrow == ncol);
  for(int ibatch=0; ibatch<size_batch; ++ibatch) {
    int indx_A = ibatch * nrow*ncol;
    int indx_b = ibatch * nrow;
    
    for(int irow=0; irow<nrow; ++irow) {
      for(int icol=0; icol<ncol; ++icol) {
	double value = rand() % 100 / 50.0;
	Acopy[irow*ncol+icol] = value;
      }
    }

    int indx_x = ibatch * ncol;
    for(int icol=0; icol<ncol; ++icol) x_ref[indx_x + icol] = rand() % 100 / 50.0;

    const int lda = nrow;
    const int ldb = ncol;
    const int ldc = nrow;
    const double alpha = 1.0;
    const int incx = 1;
    const double beta = 0.0;
    const int incy = 1;

    if(symmetric) {
    
      // A = A.A^T
      dgemm_((char *) "N", (char *) "N", &nrow, &ncol, &ncol, &alpha, Acopy, &lda, Acopy, &ldb, &beta, &A[indx_A], &ldc);

    } else {

      for(int i=0; i<nrow*ncol; ++i) A[indx_A + i] = Acopy[i];
      
    }
    
    // b = A.x
    //    dgemv_((char*) "N", &nrow, &ncol, &alpha, &A[indx_A], &lda, &x_ref[indx_b], &incx, &beta, &b[indx_b], &incy);
    for(int irow=0; irow<nrow; ++irow) {
      double sum = 0.0;
      for(int icol=0; icol<ncol; ++icol) sum += A[indx_A + icol*nrow+irow] * x_ref[indx_x+icol];
      b[indx_b+irow] = sum;
    }
  }
  free(Acopy);
}

void init_matrix_file(int nrow, int ncol, int size_batch, double * A, double * b, double * x, double * x_ref, bool &symmetric)
{
  // we're lazy...
  if(_SIZE_BATCH < _NUM_INPUT_MATRICES) {
    printf("Error: _SIZE_BATCH < _NUM_INPUT_MATRICES\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  symmetric = false;

  const int lda = nrow;
  const int ldb = ncol;
  const int ldc = nrow;
  const double alpha = 1.0;
  const int incx = 1;
  const double beta = 0.0;
  const int incy = 1;
  
  //  char const * dir = "/Users/cjknight/Documents/Projects/Doga/to-chris/";
  char const * dir = "/projects/catalyst/knight/projects/Doga/to-chris";

  for(int imat=0; imat<_NUM_INPUT_MATRICES; ++imat) {
    char fname[100];
    
    int indx_A = imat * nrow*ncol;
    int indx_b = imat * nrow;
    int indx_x = imat * ncol;
    
    {
      sprintf(fname, "%s/kernel/A.%i.txt",dir,imat);
      std::ifstream inputfile(fname);
      
      if(!inputfile.is_open()) {
	printf("init_matrix_file :: error opening %s file\n",fname);
	MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      for(int irow=0; irow<nrow; ++irow) {
	for(int icol=0; icol<ncol; ++icol) {
	  inputfile >> A[indx_A + icol*nrow+irow];
	}
      }
      
      inputfile.close();
    }
    
    {
      sprintf(fname, "%s/data/b.%i.txt",dir,imat);
      std::ifstream inputfile(fname);
      
      if(!inputfile.is_open()) {
	printf("init_matrix_file :: error opening %s file\n",fname);
	MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      for(int irow=0; irow<nrow; ++irow) inputfile >> b[indx_b + irow];
      
      inputfile.close();
    }
    
    {
      sprintf(fname, "%s/solutions/x.%i.txt",dir,imat);
      std::ifstream inputfile(fname);
      
      if(!inputfile.is_open()) {
	printf("init_matrix_file :: error opening %s file\n",fname);
	MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      for(int icol=0; icol<ncol; ++icol) inputfile >> x_ref[indx_x + icol];
      
      inputfile.close();
    }

  }
  
  // copy inputs to create larger batch

  int imat = 0;
  for(int ibatch=_NUM_INPUT_MATRICES; ibatch<size_batch; ++ibatch) {
    int indx_A_src = imat * nrow * ncol;
    int indx_b_src = imat * nrow;
    int indx_x_src = imat * ncol;
    
    int indx_A_dest = ibatch * nrow*ncol;
    int indx_b_dest = ibatch * nrow;
    int indx_x_dest = ibatch * ncol;
    
    for(int i=0; i<nrow*ncol; ++i) A[indx_A_dest + i] = A[indx_A_src + i];
    for(int i=0; i<nrow; ++i) b[indx_b_dest + i] = b[indx_b_src + i];
    for(int i=0; i<ncol; ++i) x_ref[indx_x_dest + i] = x_ref[indx_x_src + i];
  }
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

#if _USE_GPU
  printf(" -- _USE_GPU defined\n");
#else
  printf(" -- _USE_GPU not defined\n");
#endif
  
  int num_devices = dev_num_devices();
  printf(" # of devices detected= %i\n",num_devices);
  
  int ngpus = 1;
  printf(" Running on %i GPUs!\n",ngpus);
  
  int gpu_id = 0;
  dev_set_device(gpu_id);
  
  // Initialize data (to be replaced by CLI arguments)
  
  int num_iter = _NUM_ITER;
  int nrow = _NUM_ROW;
  int ncol = _NUM_COL;
  
  int lda = _NUM_ROW;
  int ldb = 1;

  int size_batch = _SIZE_BATCH;

  size_t size_A = nrow * ncol * sizeof(double);
  size_t size_b = nrow * sizeof(double);
  size_t size_x = ncol * sizeof(double);

  printf("\nKernel w/ num_iter= %i  size_batch= %i  nrow,ncol= %i x %i\n", num_iter, size_batch, nrow, ncol);
  printf("  size_A= %lu (%f GB)  total= %f GB\n",size_A, (double)size_A / 1024/1024/1024, (double) size_batch * size_A / 1024/1024/1024);
  printf("  size_b= %lu (%f GB)  total= %f GB\n",size_b, (double)size_b / 1024/1024/1024, (double) size_batch * size_b / 1024/1024/1024);
  printf("  size_x= %lu (%f GB)  total= %f GB\n",size_x, (double)size_x / 1024/1024/1024, (double) size_batch * size_x / 1024/1024/1024);
  printf("  Host memory allocated: %f GB\n\n",(double) (size_A + size_b + 2 * size_x) * size_batch / 1024/1024/1024);
  
  double * A = (double *) malloc(size_batch * size_A); // nrow X ncol
  double * x = (double *) malloc(size_batch * size_x); // ncol X 1
  double * b = (double *) malloc(size_batch * size_b); // nrow X 1
  
  double * x_ref = (double *) malloc(size_batch * size_x);

  bool symmetric;
  
  srand(12345 + rnk);

  printf(" -- initializing matrices\n");
  
  // Diagonal square matrix
  //  init_matrix_1(nrow, ncol, size_batch, A, b, x, x_ref, symmetric);

  // Diagonal square matrix using blas
  //  init_matrix_2(nrow, ncol, size_batch, A, b, x, x_ref, symmetric);

  // Random positive matrix A; symmetric if nrow == ncol; first {A,b} is duplicated
  //init_matrix_3(nrow, ncol, size_batch, A, b, x, x_ref, symmetric);
  
  // Random positive matrix A; symmetric if nrow == ncol; every {A,b} is unique (helpful for testing iterative methods)
  //  init_matrix_4(nrow, ncol, size_batch, A, b, x, x_ref, symmetric);

  init_matrix_file(nrow, ncol, size_batch, A, b, x, x_ref, symmetric);
  
  // print first matrix for debugging
  
#if 0
  printf("Matrix A[0] (Column-Major)= \n");
  for(int irow=0; irow<nrow; ++irow) {
    for(int icol=0; icol<ncol; ++icol) printf(" %f",A[icol*nrow+irow]);
    printf("\n");
  }

  // printf("Matrix A[0]= \n");
  // for(int irow=0; irow<nrow; ++irow) {
  //   for(int icol=0; icol<ncol; ++icol) printf("(%i,%i,%i) %f\n",irow,icol,icol*nrow+irow,A[icol*nrow+irow]);
  // }
  
  // printf("Matrix A= ");
  // for(int i=0; i<nrow*ncol; ++i) printf(" %f",A[i]);
  // printf("\n");

  printf("Vector b[0]= \n");
  for(int irow=0; irow<nrow; ++irow) printf(" %f\n",b[irow]);
  
  printf("Vector x_ref[0]= \n");
  for(int icol=0; icol<ncol; ++icol) printf(" %f\n",x_ref[icol]);
#endif

  // cpu solve

#if _CHECK_CPU
  printf(" -- checking result on cpu\n");
  
  solve_cpu(nrow, ncol, A, x, b, x_ref);
#endif

  printf(" -- pushing data to gpu\n");

  double * d_A = (double *) dev_malloc(size_batch * size_A);
  dev_push(d_A, A, size_batch * size_A);
  
  double * d_x = (double *) dev_malloc(size_batch * size_x);
  dev_push(d_x, x, size_batch * size_x);
  
  double * d_b = (double *) dev_malloc(size_batch * size_b);
  dev_push(d_b, b, size_batch * size_b);

  printf(" -- doing something useful on gpus\n");

  solve(num_iter, nrow, ncol, d_A, d_x, x, d_b, x_ref, "LU");

  if(symmetric) solve(num_iter, nrow, ncol, d_A, d_x, x, d_b, x_ref, "CHOL");
  
  solve(num_iter, nrow, ncol, d_A, d_x, x, d_b, x_ref, "QR");

  if(symmetric) solve_batch(num_iter, size_batch, nrow, ncol, A, x, b, x_ref, "CHOL");

  printf(" -- cleaning up\n");
  
  dev_free(d_b);
  dev_free(d_x);
  dev_free(d_A);

  free(x_ref);
  free(b);
  free(x);
  free(A);
  
  MPI_Finalize();
}
