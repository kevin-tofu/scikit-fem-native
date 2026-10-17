#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <stdexcept>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

py::tuple tabulate_hex27_shapes(
    py::array_t<double, py::array::c_style | py::array::forcecast> points) {
    if (points.ndim() != 2 || points.shape(0) != 3) {
        throw std::invalid_argument("Hex27 reference points must have shape (3, nq)");
    }
    const py::ssize_t count = points.shape(1);
    py::array_t<double> values({count, py::ssize_t{27}});
    py::array_t<double> gradients({count, py::ssize_t{27}, py::ssize_t{3}});
    const auto* x = points.data();
    auto* shape = values.mutable_data();
    auto* gradient = gradients.mutable_data();

    for (py::ssize_t q = 0; q < count; ++q) {
        std::array<std::array<double, 3>, 3> v{};
        std::array<std::array<double, 3>, 3> d{};
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate = x[axis * count + q];
            v[axis] = {{2. * (coordinate - .5) * (coordinate - 1.),
                4. * coordinate * (1. - coordinate),
                2. * coordinate * (coordinate - .5)}};
            d[axis] = {{4. * coordinate - 3., 4. - 8. * coordinate,
                4. * coordinate - 1.}};
        }
        int node = 0;
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i, ++node) {
                    shape[q * 27 + node] = v[0][i] * v[1][j] * v[2][k];
                    const auto offset = (q * 27 + node) * 3;
                    gradient[offset] = d[0][i] * v[1][j] * v[2][k];
                    gradient[offset + 1] = v[0][i] * d[1][j] * v[2][k];
                    gradient[offset + 2] = v[0][i] * v[1][j] * d[2][k];
                }
            }
        }
    }
    return py::make_tuple(std::move(values), std::move(gradients));
}

}  // namespace

namespace native_fem {

void bind_basis_tabulation(py::module_& module) {
    module.def("tabulate_hex27_shapes", &tabulate_hex27_shapes,
        py::arg("points"),
        "Tabulate Hex27 shape values and reference gradients.");
}

}  // namespace native_fem
