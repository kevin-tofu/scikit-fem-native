#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "native_fem/python_bindings.hpp"
#include "native_fem/parallel.hpp"

namespace py = pybind11;

namespace {

double inverse(const double* matrix, double* result, int dimension) {
    if (dimension == 2) {
        const double determinant = matrix[0] * matrix[3] -
            matrix[1] * matrix[2];
        if (determinant == 0.) throw std::invalid_argument(
            "singular element geometry");
        result[0] = matrix[3] / determinant;
        result[1] = -matrix[1] / determinant;
        result[2] = -matrix[2] / determinant;
        result[3] = matrix[0] / determinant;
        return determinant;
    }
    const double determinant =
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (determinant == 0.) throw std::invalid_argument(
        "singular element geometry");
    result[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant;
    result[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant;
    result[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant;
    result[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant;
    result[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant;
    result[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant;
    result[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant;
    result[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant;
    result[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant;
    return determinant;
}

py::tuple tabulate_basis_geometry(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> connectivity,
    py::array_t<double, py::array::c_style | py::array::forcecast> shape,
    py::array_t<double, py::array::c_style | py::array::forcecast> reference_gradients,
    py::array_t<double, py::array::c_style | py::array::forcecast> geometry_shape,
    py::array_t<double, py::array::c_style | py::array::forcecast> geometry_gradients,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights) {
    const auto p = coordinates.request();
    const auto t = connectivity.request();
    const auto s = shape.request();
    const auto rg = reference_gradients.request();
    const auto gs = geometry_shape.request();
    const auto gg = geometry_gradients.request();
    const auto w = weights.request();
    if (p.ndim != 2 || (p.shape[0] != 2 && p.shape[0] != 3) ||
        t.ndim != 2 || s.ndim != 2 || rg.ndim != 3 || gs.ndim != 2 ||
        gg.ndim != 3 || w.ndim != 1)
        throw std::invalid_argument("invalid basis geometry input rank");
    const py::ssize_t dimension = p.shape[0];
    const py::ssize_t points = p.shape[1];
    const py::ssize_t geometry_nodes = t.shape[0];
    const py::ssize_t elements = t.shape[1];
    const py::ssize_t quadrature = s.shape[0];
    const py::ssize_t nodes = s.shape[1];
    if (rg.shape[0] != quadrature || rg.shape[1] != nodes ||
        rg.shape[2] != dimension || gs.shape[0] != quadrature ||
        gs.shape[1] != geometry_nodes || gg.shape[0] != quadrature ||
        gg.shape[1] != geometry_nodes || gg.shape[2] != dimension ||
        w.shape[0] != quadrature)
        throw std::invalid_argument("incompatible basis geometry shapes");

    py::array_t<double> tabulated_shape({elements, quadrature, nodes});
    py::array_t<double> gradients({elements, quadrature, nodes, dimension});
    py::array_t<double> dx({elements, quadrature});
    py::array_t<double> global({elements, quadrature, dimension});
    auto* output_shape = static_cast<double*>(tabulated_shape.request().ptr);
    auto* output_gradient = static_cast<double*>(gradients.request().ptr);
    auto* output_dx = static_cast<double*>(dx.request().ptr);
    auto* output_global = static_cast<double*>(global.request().ptr);
    const auto* point_data = static_cast<const double*>(p.ptr);
    const auto* cell_data = static_cast<const std::int64_t*>(t.ptr);
    const auto* shape_data = static_cast<const double*>(s.ptr);
    const auto* reference_data = static_cast<const double*>(rg.ptr);
    const auto* geometry_shape_data = static_cast<const double*>(gs.ptr);
    const auto* geometry_gradient_data = static_cast<const double*>(gg.ptr);
    const auto* weight_data = static_cast<const double*>(w.ptr);

    {
        py::gil_scoped_release release;
        native_fem::parallel_for(
            static_cast<std::size_t>(elements),
            [&](std::size_t element_begin, std::size_t element_end) {
        for (py::ssize_t e = static_cast<py::ssize_t>(element_begin);
             e < static_cast<py::ssize_t>(element_end); ++e) {
            for (py::ssize_t q = 0; q < quadrature; ++q) {
                double jacobian[9]{};
                double jacobian_inverse[9]{};
                double physical_point[3]{};
                for (py::ssize_t a = 0; a < geometry_nodes; ++a) {
                    const auto vertex = cell_data[a * elements + e];
                    if (vertex < 0 || vertex >= points)
                        throw std::invalid_argument("connectivity index is out of bounds");
                    for (py::ssize_t i = 0; i < dimension; ++i) {
                        const double coordinate = point_data[i * points + vertex];
                        physical_point[i] +=
                            geometry_shape_data[q * geometry_nodes + a] * coordinate;
                        for (py::ssize_t j = 0; j < dimension; ++j)
                            jacobian[i * dimension + j] += coordinate *
                                geometry_gradient_data[
                                    (q * geometry_nodes + a) * dimension + j];
                    }
                }
                for (py::ssize_t i = 0; i < dimension; ++i)
                    output_global[(e * quadrature + q) * dimension + i] =
                        physical_point[i];
                const double determinant = inverse(
                    jacobian, jacobian_inverse, static_cast<int>(dimension));
                output_dx[e * quadrature + q] =
                    std::abs(determinant) * weight_data[q];
                for (py::ssize_t a = 0; a < nodes; ++a) {
                    output_shape[(e * quadrature + q) * nodes + a] =
                        shape_data[q * nodes + a];
                    for (py::ssize_t j = 0; j < dimension; ++j) {
                        double value = 0.;
                        for (py::ssize_t k = 0; k < dimension; ++k)
                            value += reference_data[
                                (q * nodes + a) * dimension + k] *
                                jacobian_inverse[k * dimension + j];
                        output_gradient[
                            ((e * quadrature + q) * nodes + a) * dimension + j] = value;
                    }
                }
            }
        }
        });
    }
    return py::make_tuple(tabulated_shape, gradients, dx, global);
}

}  // namespace

void native_fem::bind_basis_geometry(py::module_& module) {
    module.def("tabulate_basis_geometry", &tabulate_basis_geometry);
}
