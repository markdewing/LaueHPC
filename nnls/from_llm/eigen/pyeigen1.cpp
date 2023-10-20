
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>


Eigen::VectorXd non_negative_least_squares(const Eigen::MatrixXd& A, const Eigen::VectorXd& y, double epsilon = 1e-6);
Eigen::VectorXf non_negative_least_squares_float(const Eigen::MatrixXf& A, const Eigen::VectorXf& y, double epsilon = 1e-6);

namespace py = pybind11;

Eigen::VectorXd test1(Eigen::Ref<Eigen::MatrixXd> A, Eigen::Ref<Eigen::VectorXd> b)
{
    Eigen::VectorXd x = non_negative_least_squares(A, b);
    return x;
}

Eigen::VectorXf testf(Eigen::Ref<Eigen::MatrixXf> A, Eigen::Ref<Eigen::VectorXf> b)
{
    Eigen::VectorXf x = non_negative_least_squares_float(A, b);
    return x;
}

PYBIND11_MODULE(solver, m) {
    m.doc() = "Python interface to Eigen solver"; // Add a docstring to the module

    m.def("solve", &test1, "Solve Ax=b for x",py::arg("A"),py::arg("b"));
    m.def("solvef", &testf, "Solve Ax=b for x",py::arg("A"),py::arg("b"));
}

