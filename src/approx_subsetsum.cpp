#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double sum_array(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.unchecked<1>();
    const py::ssize_t n = buf.shape(0);

    double s = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        s += buf(i);
    }
    return s;
}

PYBIND11_MODULE(approx_subsetsum, m) {
    m.doc() = "Example pybind11 module exposing a NumPy-taking function";

    m.def("sum_array", &sum_array, "Return the sum of a 1D NumPy array of float64");
}
