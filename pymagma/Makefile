
include machine/polaris.inc
#include machine/local_desktop.inc
#include machine/local_thor.inc

USE_CUDA = yes
USE_MAGMA = no

CXX = g++
CXXFLAGS = -g -fPIC $(MAGMA_INC) -DADD_ $(PYTHON_INC) $(CUDA_INC) $(PYBIND_INC)  -fopenmp


$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<


sources = solve_cpu.o solver.o

ifeq ($(USE_MAGMA),yes)
	sources += solve_magma.o
	CXXFLAGS += -DUSE_MAGMA
    LIBS += $(MAGMA_LIB) -lmagma
else
    LIBS += -llapack -lblas
endif

ifeq ($(USE_CUDA),yes)
	sources += solve_cuda.o
	CXXFLAGS += -DUSE_CUDA
    LIBS += $(CUDA_LIB) -lcusolver -lcudart
endif

solver:  $(sources)
	$(CXX) $(CXXFLAGS) $(sources) -shared -o solver.so $(LIBS)

solve_cuda.o: solve_cuda.cu
	nvcc -Xcompiler=-fPIC -Xcompiler=-fopenmp -g -c solve_cuda.cu

clean:
	rm *.o
