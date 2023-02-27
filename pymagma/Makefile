
MAGMA_INC = /home/mdewing/physics/codes/linalg/magma/usr/include
MAGMA_LIB = /home/mdewing/physics/codes/linalg/magma/usr/lib

# Might also need path to pybind11
PYTHON_INC = /usr/include/python3.6m

CXX = g++
CXXFLAGS = -g -fPIC -I$(MAGMA_INC) -DADD_ -I$(PYTHON_INC)


$(OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

solver:  solve_magma.o solver.o
	$(CXX) $(CXXFLAGS) solver.o solve_magma.o -shared -o solver.so -L${MAGMA_LIB} -lmagma

clean:
	rm *.o
