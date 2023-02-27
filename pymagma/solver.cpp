#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "solve_magma.h"

namespace py = pybind11;

// Solves Ax=b
// Input A,b, return x
py::array_t<double> solve(py::array_t<double> A, py::array_t<double> b) {

    // Get a pointer to the data in the input array
    auto A_ptr = static_cast<double *>(A.request().ptr);
    auto A_shape = A.shape();
    int nrow = A_shape[0];
    int ncol = A_shape[1];
    //printf("nrow = %d ncol = %d\n",nrow,ncol);

    auto b_ptr = static_cast<double *>(b.request().ptr);
    auto b_shape = b.shape();

    auto result = py::array_t<double>(ncol);
    auto result_ptr = static_cast<double *>(result.request().ptr);

    solve_cpu(nrow, ncol, A_ptr, b_ptr, result_ptr);

    return result;
}

void init()
{
    init_magma();
}

void fini()
{
    fini_magma();
}

// Define the module
PYBIND11_MODULE(solver, m) {
    m.doc() = "Python interface to magma solver"; // Add a docstring to the module
    m.def("solve", &solve, "Solve Ax=b for x");
    m.def("init", &init, "Initialize magma");
    m.def("fini", &fini, "Shut down magma");
}


