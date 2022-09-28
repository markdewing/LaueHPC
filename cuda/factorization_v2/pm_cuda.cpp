#ifdef _USE_GPU

#include <stdio.h>
#include <iostream>

#define _CUDA_CHECK_ERRORS()               \
{                                          \
  cudaError err = cudaGetLastError();      \
  if(err != cudaSuccess) {                 \
    std::cout                              \
      << "CUDA error with code "           \
      << cudaGetErrorString(err)           \
      << " in file " << __FILE__           \
      << " at line " << __LINE__           \
      << ". Exiting...\n";                 \
    exit(1);                               \
  }                                        \
}

int dev_num_devices()
{
  int num_devices;

  cudaGetDeviceCount(&num_devices);
  _CUDA_CHECK_ERRORS();
  
  return num_devices;
}

void dev_properties(int ndev)
{

  for(int i=0; i<ndev; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    _CUDA_CHECK_ERRORS();

    char name[256];
    strcpy(name, prop.name);

    printf("  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]\n", i, name);
  }

}

int dev_check_peer(int rank, int ngpus)
{
  int err = 0;
  if(rank == 0) printf("\nChecking P2P Access\n");
  for(int ig=0; ig<ngpus; ++ig) {
    cudaSetDevice(ig);
    //if(rank == 0) printf("Device i= %i\n",ig);

    int n = 1;
    for(int jg=0; jg<ngpus; ++jg) {
      if(jg != ig) {
        int access;
        cudaDeviceCanAccessPeer(&access, ig, jg);
        n += access;

        //if(rank == 0) printf("  --  Device j= %i  access= %i\n",jg,access);
      }
    }
    if(n != ngpus) err += 1;
  }

  return err;
}

void dev_set_device(int id)
{
  cudaSetDevice(id);
  _CUDA_CHECK_ERRORS();
}

void * dev_malloc(size_t N)
{
  void * ptr;
  cudaMalloc((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  return ptr;
}

void * dev_malloc_host(size_t N)
{
  void * ptr;
  cudaMallocHost((void**) &ptr, N);
  _CUDA_CHECK_ERRORS();
  return ptr;
}

void dev_free(void * ptr)
{
  cudaFree(ptr);
  _CUDA_CHECK_ERRORS();
}

void dev_free_host(void * ptr)
{
  cudaFreeHost(ptr);
  _CUDA_CHECK_ERRORS();
}

void dev_push(void * d_ptr, void * h_ptr, size_t N)
{
  cudaMemcpy(d_ptr, h_ptr, N, cudaMemcpyHostToDevice);
  _CUDA_CHECK_ERRORS();
}

void dev_pull(void * d_ptr, void * h_ptr, size_t N)
{
  cudaMemcpy(h_ptr, d_ptr, N, cudaMemcpyDeviceToHost);
  _CUDA_CHECK_ERRORS();
}

void dev_copy(void * a_ptr, void * b_ptr, size_t N)
{
  cudaMemcpy(b_ptr, a_ptr, N, cudaMemcpyDeviceToDevice);
  _CUDA_CHECK_ERRORS();
}

#endif
