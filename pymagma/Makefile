
include machine/polaris.inc
#include machine/local_desktop.inc

USE_CUDA = yes

CXX = g++
CXXFLAGS = -g -fPIC $(MAGMA_INC) -DADD_ $(PYTHON_INC) $(CUDA_INC) $(PYBIND_INC)


$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<


sources = solve_magma.o solve_cpu.o solver.o

ifeq ($(USE_CUDA),yes)
	sources += solve_cuda.o
	CXXFLAGS += -DUSE_CUDA
endif

solver:  $(sources)
	$(CXX) $(CXXFLAGS) $(sources) -shared -o solver.so $(MAGMA_LIB) -lmagma $(CUDA_LIB) -lcusolver

solve_cuda.o: solve_cuda.cu
	nvcc -Xcompiler=-fPIC -g -c solve_cuda.cu

clean:
	rm *.o
