#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "solve_cpu.h"
#ifdef USE_MAGMA
#include "solve_magma.h"
#endif
#ifdef  USE_CUDA
#include "solve_cuda.h"
#endif
#include "perf_info.h"
#include <string>

namespace py = pybind11;

// Solves Ax=b
// Input A,b, return x
py::array_t<double> solve(py::array_t<double> A, py::array_t<double> b, const std::string& place, const std::string& method, PerfInfo &perf) {

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

    if (method == "qr")
    {
        if (place == "cpu")
            solve_cpu_QR(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#ifdef USE_MAGMA
        else if (place == "gpu")
            solve_gpu_QR(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
        else if (place == "gpu_simple")
            solve_gpu_simple_QR(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#endif
#ifdef USE_CUDA
        else if (place == "cuda")
            solve_cuda_QR(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#endif
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else if (method == "svd")
    {
        if (place == "cpu")
            solve_cpu_SVD(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#ifdef USE_MAGMA
        else if (place == "gpu_simple")
            solve_gpu_simple_SVD(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#endif
#ifdef USE_CUDA
        else if (place == "cuda")
            solve_cuda_SVD(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
#endif
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else if (method == "ls")
    {
        if (place == "cpu")
            solve_cpu_LS(nrow, ncol, A_ptr, b_ptr, result_ptr, perf);
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else
        throw std::invalid_argument(std::string("unknown solution method: ") + method);

    return result;
}

// Solves Ax=b for a number of systems at the same time
// Input A,b, return x
// Batch is last dimension for A,b, and x
py::array_t<double> solve_batch(py::array_t<double> A, py::array_t<double> b, const std::string& place, const std::string& method, PerfInfo &perf) {

    // Get a pointer to the data in the input array
    auto A_ptr = static_cast<double *>(A.request().ptr);
    auto A_shape = A.shape();
    int nrow = A_shape[0];
    int ncol = A_shape[1];
    int nbatch = A_shape[2];
    //printf("nrow = %d ncol = %d nbatch = %d\n",nrow,ncol,nbatch);

    auto b_ptr = static_cast<double *>(b.request().ptr);
    auto b_shape = b.shape();

    auto result = py::array_t<double, py::array::f_style>( {ncol,nbatch} );
    auto result_ptr = static_cast<double *>(result.request().ptr);

    if (method == "qr")
    {
        if (place == "cpu")
            solve_batch_cpu_QR(nrow, ncol, nbatch, A_ptr, b_ptr, result_ptr, perf);
#ifdef USE_CUDA
        else if (place == "cuda")
            solve_batch_cuda_QR(nrow, ncol, nbatch, A_ptr, b_ptr, result_ptr, perf);
#endif
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else if (method == "svd")
    {
        if (place == "cpu")
            solve_batch_cpu_SVD(nrow, ncol, nbatch, A_ptr, b_ptr, result_ptr, perf);
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else if (method == "ls")
    {
#ifdef USE_CUDA
        if (place == "cuda")
            solve_batch_cuda_LS(nrow, ncol, nbatch, A_ptr, b_ptr, result_ptr, perf);
#endif
        else
            throw std::invalid_argument(std::string("unknown execution place: ") + place + std::string(" for solution method: ") + method);
    }
    else
        throw std::invalid_argument(std::string("unknown solution method: ") + method);

    return result;
}

void init()
{
#ifdef USE_MAGMA
    init_magma();
#endif
}

void fini()
{
#ifdef USE_MAGMA
    fini_magma();
#endif
}

// Define the module
PYBIND11_MODULE(solver, m) {
    m.doc() = "Python interface to magma solver"; // Add a docstring to the module
    py::class_<PerfInfo>(m, "PerfInfo")
        .def(py::init<>())
        .def_readwrite("elapsed", &PerfInfo::elapsed)
        .def("get_comp", &PerfInfo::get_comp);
    m.def("solve", &solve, "Solve Ax=b for x",py::arg("A"),py::arg("b"),
                 py::arg("place")="cpu", py::arg("method")="qr", py::arg("perf") = PerfInfo()
                     );
    m.def("solve_batch", &solve_batch, "Solve Ax=b for x",py::arg("A"),py::arg("b"),
                 py::arg("place")="cpu", py::arg("method")="qr", py::arg("perf") = PerfInfo()
                     );
    m.def("init", &init, "Initialize magma");
    m.def("fini", &fini, "Shut down magma");
}


