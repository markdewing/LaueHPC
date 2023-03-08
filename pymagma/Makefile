
include machine/polaris.inc
#include machine/local_desktop.inc

CXX = g++
CXXFLAGS = -g -fPIC $(MAGMA_INC) -DADD_ $(PYTHON_INC) $(CUDA_INC) $(PYBIND_INC)


$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

solver:  solve_magma.o solve_cpu.o solver.o
	$(CXX) $(CXXFLAGS) solver.o solve_magma.o solve_cpu.o -shared -o solver.so $(MAGMA_LIB) -lmagma

clean:
	rm *.o
