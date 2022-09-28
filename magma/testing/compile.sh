

#FILE=testing_dgetrs_batched
FILE=testing_dgesv_rbt_batched

/usr/bin/c++   -I/usr/local/cuda/include \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/build/include \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/include \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/control \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/magmablas \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/sparse/include \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/sparse/control \
	       -I/home/knight/soft/thetagpu/magma/magma-2.6.2/testing  \
	       -std=c++11 -fopenmp -Wall -Wno-unused-function \
	       -o ${FILE}.cpp.o -c ${FILE}.cpp

/usr/bin/c++   -std=c++11 -fopenmp -Wall -Wno-unused-function -rdynamic \
	       ${FILE}.cpp.o  -o ${FILE}  \
	       -L/home/knight/soft/thetagpu/magma/magma-2.6.2/build/lib -Wl,-rpath,/home/knight/soft/thetagpu/magma/magma-2.6.2/build/lib:/usr/local/cuda/lib64 \
	       -ltester -llapacktest -lmagma \
	       /usr/local/cuda/lib64/libcudart_static.a -lpthread -ldl -lrt \
	       /usr/local/cuda/lib64/libcudadevrt.a /usr/local/cuda/lib64/libcudart.so \
	       /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcusparse.so \
	       -L/home/knight/soft/thetagpu/lapack/lib -llapack -lrefblas -lgfortran 

