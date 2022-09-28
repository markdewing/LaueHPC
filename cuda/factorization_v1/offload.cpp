#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <vector>

#include <chrono>

#include "pm_cuda.h"

#include "cublas_v2.h"
#include "cusolverDn.h"

// copied from cuda/samples/7_CUDALibraries/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp

/*
 *  solve A*x = b by Cholesky factorization
 *
 */

int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double *Acopy,
                     int lda, const double *b, double *x)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;
  
  int bufferSize = 0;
  int *info = NULL;
  double *buffer = NULL;
  double *A = NULL;
  int h_info = 0;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  cusolverDnDpotrf_bufferSize(handle, uplo, n, (double *)Acopy,
			      lda, &bufferSize);

  cudaMalloc(&info, sizeof(int));
  cudaMalloc(&buffer, sizeof(double) * bufferSize);
  cudaMalloc(&A, sizeof(double) * lda * n);

  // prepare a copy of A because potrf will overwrite A with L
  cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice);
  cudaMemset(info, 0, sizeof(int));

  auto t0 = high_resolution_clock::now();
  
  cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info);

  cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

  if (0 != h_info) fprintf(stderr, "Error: Cholesky factorization failed\n");

  cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);

  cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info);

  cudaDeviceSynchronize();
  
  auto t1 = high_resolution_clock::now();
  auto ms_int = duration_cast<microseconds>(t1 - t0);
  int time = ms_int.count();
  //  std::cout << time << " us elapsed" << std::endl;
  
  cudaFree(info);
  cudaFree(buffer);
  cudaFree(A);

  return time;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */

int linearSolverLU(cusolverDnHandle_t handle, int n, const double *Acopy,
                   int lda, const double *b, double *x)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;
  
  int bufferSize = 0;
  int *info = NULL;
  double *buffer = NULL;
  double *A = NULL;
  int *ipiv = NULL;  // pivoting sequence
  int h_info = 0;

  cusolverDnDgetrf_bufferSize(handle, n, n, (double *)Acopy,
                              lda, &bufferSize);

  cudaMalloc(&info, sizeof(int));
  cudaMalloc(&buffer, sizeof(double) * bufferSize);
  cudaMalloc(&A, sizeof(double) * lda * n);
  cudaMalloc(&ipiv, sizeof(int) * n);

  // prepare a copy of A because getrf will overwrite A with L
  cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice);
  cudaMemset(info, 0, sizeof(int));

  auto t0 = high_resolution_clock::now();
  
  cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info);
  cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

  if (0 != h_info) fprintf(stderr, "Error: LU factorization failed\n");

  cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);
  cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
  cudaDeviceSynchronize();
  
  auto t1 = high_resolution_clock::now();
  auto ms_int = duration_cast<microseconds>(t1 - t0);
  int time = ms_int.count();
  //  std::cout << time << " us elapsed" << std::endl;
  
  cudaFree(info);
  cudaFree(buffer);
  cudaFree(A);
  cudaFree(ipiv);

  return time;
}

/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(cusolverDnHandle_t handle, int n, const double *Acopy,
                   int lda, const double *b, double *x)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;
  
  cublasHandle_t cublasHandle = NULL;  // used in residual evaluation
  
  int bufferSize = 0;
  int bufferSize_geqrf = 0;
  int bufferSize_ormqr = 0;
  int *info = NULL;
  double *buffer = NULL;
  double *A = NULL;
  double *tau = NULL;
  int h_info = 0;
  const double one = 1.0;

  cublasCreate(&cublasHandle);

  cusolverDnDgeqrf_bufferSize(handle, n, n, (double *)Acopy,
                              lda, &bufferSize_geqrf);
  cusolverDnDormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
			      CUBLAS_OP_T, n, 1, n, A, lda,
			      NULL, x, n, &bufferSize_ormqr);

  _CUDA_CHECK_ERRORS();

  //  printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);

  bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf : bufferSize_ormqr;

  cudaMalloc(&info, sizeof(int));
  cudaMalloc(&buffer, sizeof(double) * bufferSize);
  cudaMalloc(&A, sizeof(double) * lda * n);
  cudaMalloc((void **)&tau, sizeof(double) * n);

  _CUDA_CHECK_ERRORS();
  
  // prepare a copy of A because getrf will overwrite A with L
  cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice);

  cudaMemset(info, 0, sizeof(int));

  // compute QR factorization
  
  auto t0 = high_resolution_clock::now();
  
  cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info);

  cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

  if (0 != h_info) fprintf(stderr, "Error: LU factorization failed\n");

  cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);

  // compute Q^T*b
  cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1,
		   n, A, lda, tau, x, n, buffer, bufferSize,
		   info);

  // x = R \ Q^T*b
  cublasDtrsm(cublasHandle, CUBLAS_SIDE_LEFT,
	      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
	      CUBLAS_DIAG_NON_UNIT, n, 1, &one, A, lda, x, n);
  cudaDeviceSynchronize();
  
  auto t1 = high_resolution_clock::now();
  auto ms_int = duration_cast<microseconds>(t1 - t0);
  int time = ms_int.count();
  //  std::cout << time << " us elapsed" << std::endl;
  
  cublasDestroy(cublasHandle);
  
  cudaFree(info);
  cudaFree(buffer);
  cudaFree(A);
  cudaFree(tau);

  return time;
}

// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/potrfBatched/cusolver_potrfBatched_example.cu
int linearSolverCHOL_batch(cusolverDnHandle_t handle, int _batchSize, int n, const double * Acopy,
			   int _lda, const double * bcopy, double * x)
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::microseconds;
  
  cudaStream_t stream = NULL;
  
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const int batchSize = _batchSize;
  const int nrhs = 1;
  const int m = n;
  const int lda = m;
  const int ldb = m;
  
  std::vector<int> infoArray(batchSize, 0); /* host copy of error info */
  
  std::vector<double *> Aarray(batchSize, nullptr);
  std::vector<double *> Barray(batchSize, nullptr);
  
  double **d_Aarray = nullptr;
  double **d_Barray = nullptr;
  int *d_infoArray = nullptr;
  
  /* step 1: create cusolver handle, bind a stream */
  cusolverDnCreate(&handle);
  
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(handle, stream);

  _CUDA_CHECK_ERRORS();
  
  /* step 2: copy A to device */
  for (int j = 0; j < batchSize; j++) {
    cudaMalloc(reinterpret_cast<void **>(&Aarray[j]), sizeof(double) * lda * m);
    cudaMalloc(reinterpret_cast<void **>(&Barray[j]), sizeof(double) * ldb * nrhs);
  }
  cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size());
  
  cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size());
  cudaMalloc(reinterpret_cast<void **>(&d_Barray), sizeof(double *) * Barray.size());

  _CUDA_CHECK_ERRORS();

  for(int j=0; j<batchSize; ++j) {
    cudaMemcpyAsync(Aarray[j], &Acopy[j*lda*ldb], sizeof(double) * lda*ldb,
		    cudaMemcpyHostToDevice, stream);
  
    cudaMemcpyAsync(Barray[j], &bcopy[j*ldb], sizeof(double) * ldb, cudaMemcpyHostToDevice, stream);
  }
 
  cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(),
  		  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(double) * Barray.size(),
  			     cudaMemcpyHostToDevice, stream);

  cudaStreamSynchronize(stream);
  
  _CUDA_CHECK_ERRORS();

  auto t0 = high_resolution_clock::now();
  
  /* step 3: Cholesky factorization */
  cusolverDnDpotrfBatched(handle, uplo, m, d_Aarray, lda, d_infoArray, batchSize);
  
  
  // step 4: solve A*X = B
  cusolverDnDpotrsBatched(handle, uplo, m, nrhs, /* only support rhs = 1*/
			  d_Aarray, lda, d_Barray, ldb, d_infoArray, batchSize);

  cudaStreamSynchronize(stream);
  
  auto t1 = high_resolution_clock::now();
  auto ms_int = duration_cast<microseconds>(t1 - t0);
  int time = ms_int.count();
  //  std::cout << time << " us elapsed" << std::endl;
  
  cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int), cudaMemcpyDeviceToHost, stream);
  for(int j=0; j<batchSize; ++j) {
    cudaMemcpyAsync(&x[j*lda], Barray[j], sizeof(double) * ldb, cudaMemcpyDeviceToHost, stream);
  }
  
  cudaStreamSynchronize(stream);

  _CUDA_CHECK_ERRORS();
  
  //  std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
  if (0 > infoArray[0]) {
    std::printf("%d-th parameter is wrong \n", -infoArray[0]);
    exit(1);
  }
  
  /* free resources */
  cudaFree(d_Aarray);
  cudaFree(d_Barray);
  cudaFree(d_infoArray);
  for (int j = 0; j < batchSize; j++) {
    cudaFree(Aarray[j]);
    cudaFree(Barray[j]);
  }
  
  cusolverDnDestroy(handle);
  
  cudaStreamDestroy(stream);
  
  return time;
}

int dev_solve(int dim, double * A, double * x, double * b, const char * mode)
{  
  cusolverDnHandle_t handle = NULL;
  
  cusolverDnCreate(&handle);
  _CUDA_CHECK_ERRORS();

  if(strcmp(mode, "QR") == 0) return linearSolverQR(handle, dim, A, dim, b, x);
  else if(strcmp(mode, "LU") == 0) return linearSolverLU(handle, dim, A, dim, b, x);
  else if(strcmp(mode, "CHOL") == 0) return linearSolverCHOL(handle, dim, A, dim, b, x);
  else {
    printf("unknown mode= %s\n",mode);
    return -1;
  }
}

int dev_solve_batch(int size_batch, int dim, double * A, double * x, double * b, const char * mode)
{

  cusolverDnHandle_t handle = NULL;
  
  cusolverDnCreate(&handle);
  _CUDA_CHECK_ERRORS();

  if(strcmp(mode, "CHOL") == 0) return linearSolverCHOL_batch(handle, size_batch, dim, A, dim, b, x);
  else {
    printf("unknown mode= %s\n",mode);
    return -1;
  }
}
