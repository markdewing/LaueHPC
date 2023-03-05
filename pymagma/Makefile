
#MAGMA_INC = -I/home/mdewing/physics/codes/linalg/magma/usr/include
# Using the 'magma' module
MAGMA_LIB = -L/soft/libraries/magma-2.7.0/lib

# System python on Polaris
#PYTHON_INC = -I/usr/include/python3.6m
# Install pybind with "pip install --user pybind11"
#PYBIND_INC = -I/home/mdewing/.local/lib/python3.6/site-packages/pybind11/include

# Using 'cray-python' module on Polaris
# Should set using appropriate env variables (CRAY_PYTHON_PREFIX?)
PYTHON_INC = -I/opt/cray/pe/python/3.9.12.1/include/python3.9/
PYBIND_INC = -I/opt/cray/pe/python/3.9.12.1/lib/python3.9/site-packages/pybind11/include

CUDA_INC = -I/soft/compilers/cudatoolkit/cuda-11.8.0/include

CXX = g++
CXXFLAGS = -g -fPIC $(MAGMA_INC) -DADD_ $(PYTHON_INC) $(CUDA_INC) $(PYBIND_INC)


$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

solver:  solve_magma.o solve_cpu.o solver.o
	$(CXX) $(CXXFLAGS) solver.o solve_magma.o solve_cpu.o -shared -o solver.so $(MAGMA_LIB) -lmagma

clean:
	rm *.o
