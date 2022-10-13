# MAGMA

A tarball for the MAGMA software was downloaded [here](https://icl.utk.edu/magma/software/index.html). Documentation is available at the same website [here](https://icl.utk.edu/projectsfiles/magma/doxygen/).

## Building on ThetaGPU

The following was used to quickly get a build running on ThetaGPU @ ALCF. This was done using an interactive job on a compute node. The Netlib blas and lapack libraries were used here as the focus is very much on the GPU capabilities.

```
$ ssh knight@theta.alcf.anl.gov
$ module load cobalt/cobalt-gpu
$ qsub -I -n 1 -t 60 -q single-gpu -A Catalyst --attrs filesystems=home

$ cd /home/knight/soft/thetagpu/magma/magma-2.6.2/build

$ cmake -DMAGMA_ENABLE_CUDA=ON \
        -DCMAKE_INSTALL_PREFIX=/home/knight/soft/thetagpu/magma/build_thetagpu \
        -DGPU_TARGET='Ampere' \
        -DBLA_VENDOR=Generic \
        -DLAPACK_LIBRARIES="-L/home/knight/soft/thetagpu/lapack/lib -llapack -lrefblas -lgfortran" \
       ..

$ make -j 8
$ make install
```

## Building on Polaris

A similar script can be used to build the MAGMA library on Polaris using the GNU programming environment and CPU math libraries provided by HPE.

```
$ ssh knight@polaris.alcf.anl.gov
$ qsub -I -l select=1,walltime=1:00:00 -q debug -A Catalyst -l filesystems=home:grand
$ module swap PrgEnv-nvhpc PrgEnv-gnu
$ module load cray-libsci

$ cd /home/knight/soft/polaris/magma/magma-2.6.2/build

$ cmake -DMAGMA_ENABLE_CUDA=ON \
        -DCMAKE_INSTALL_PREFIX=/home/knight/soft/polaris/magma/build_polaris \
        -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn \
        -DLAPACK_LIBRARIES="-lsci_gnu_82_mpi -lsci_gnu_82" \
        -DGPU_TARGET='Ampere' \
        -DBLA_VENDOR=Generic \
        ..

$ make -j 16
$ make install
```

## E4S Module on Polaris
There is a magma module provided as part of the `e4s/22.05/PrgEnv-gnu` module that provides the necessary libraries and headers for building applications.
```
$ ssh knight@polaris.alcf.anl.gov
$ module load e4s/22.05/PrgEng-gnu
$ module load magma
```
Note, loading this E4S module will switch the programming environment to `PrgEnv-gnu` and load a `cudatoolkit-standalone` module. 

## Running the test examples
A small set of test examples were created based on the tests provided in the `magma-2.6.2/testing` directory. There are thus several dependencies generated as part of the magma build that are required to run these particular test examples.


