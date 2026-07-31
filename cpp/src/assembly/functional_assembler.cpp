#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

double integrate_functional(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights) {
    const auto v = values.request();
    const auto w = weights.request();
    if (v.ndim != 2 || w.ndim != 2 ||
        v.shape[0] != w.shape[0] || v.shape[1] != w.shape[1]) {
        throw std::invalid_argument(
            "functional values and weights must have matching "
            "(entities, quadrature) shapes"
        );
    }
    const auto* value = static_cast<const double*>(v.ptr);
    const auto* weight = static_cast<const double*>(w.ptr);
    double result = 0.0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < v.size; ++i) {
            result += value[i] * weight[i];
        }
    }
    return result;
}

}  // namespace

void native_fem::bind_functional_assembler(py::module_& module) {
    module.def(
        "integrate_functional",
        &integrate_functional,
        py::arg("values"),
        py::arg("weights")
    );
}
