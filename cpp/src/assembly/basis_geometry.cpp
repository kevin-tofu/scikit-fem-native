#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "native_fem/python_bindings.hpp"
#include "native_fem/parallel.hpp"

namespace py = pybind11;

namespace {

double determinant(const double* matrix, int dimension) {
    if (dimension == 2)
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];
    return
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

double inverse(const double* matrix, double* result, int dimension) {
    if (dimension == 2) {
        const double det = determinant(matrix, dimension);
        result[0] = matrix[3] / det;
        result[1] = -matrix[1] / det;
        result[2] = -matrix[2] / det;
        result[3] = matrix[0] / det;
        return det;
    }
    const double det = determinant(matrix, dimension);
    result[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / det;
    result[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / det;
    result[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / det;
    result[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / det;
    result[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / det;
    result[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / det;
    result[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / det;
    result[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / det;
    result[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / det;
    return det;
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
    py::array_t<double> determinants({elements, quadrature});
    py::array_t<double> determinant_tolerances({elements, quadrature});
    py::array_t<double> condition_numbers({elements, quadrature});
    auto* output_shape = static_cast<double*>(tabulated_shape.request().ptr);
    auto* output_gradient = static_cast<double*>(gradients.request().ptr);
    auto* output_dx = static_cast<double*>(dx.request().ptr);
    auto* output_global = static_cast<double*>(global.request().ptr);
    auto* output_determinant = static_cast<double*>(determinants.request().ptr);
    auto* output_tolerance = static_cast<double*>(
        determinant_tolerances.request().ptr);
    auto* output_condition = static_cast<double*>(
        condition_numbers.request().ptr);
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
                double jacobian_scale = 0.;
                for (py::ssize_t i = 0; i < dimension * dimension; ++i)
                    jacobian_scale = std::max(
                        jacobian_scale, std::abs(jacobian[i]));
                const double scale_power = dimension == 2
                    ? jacobian_scale * jacobian_scale
                    : jacobian_scale * jacobian_scale * jacobian_scale;
                const double tolerance = 64. *
                    std::numeric_limits<double>::epsilon() * scale_power;
                const double det = determinant(
                    jacobian, static_cast<int>(dimension));
                output_determinant[e * quadrature + q] = det;
                output_tolerance[e * quadrature + q] = tolerance;
                if (!std::isfinite(det) || !std::isfinite(tolerance) ||
                    !(std::abs(det) > tolerance))
                    continue;
                inverse(jacobian, jacobian_inverse, static_cast<int>(dimension));
                double jacobian_norm_squared = 0.;
                double inverse_norm_squared = 0.;
                for (py::ssize_t i = 0; i < dimension * dimension; ++i) {
                    jacobian_norm_squared += jacobian[i] * jacobian[i];
                    inverse_norm_squared +=
                        jacobian_inverse[i] * jacobian_inverse[i];
                }
                output_condition[e * quadrature + q] = std::sqrt(
                    jacobian_norm_squared * inverse_norm_squared);
                output_dx[e * quadrature + q] =
                    std::abs(det) * weight_data[q];
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
    for (py::ssize_t e = 0; e < elements; ++e) {
        for (py::ssize_t q = 0; q < quadrature; ++q) {
            const double det = output_determinant[e * quadrature + q];
            const double tolerance = output_tolerance[e * quadrature + q];
            if (std::isfinite(det) && std::isfinite(tolerance) &&
                std::abs(det) > tolerance)
                continue;
            std::ostringstream message;
            message << "invalid element geometry: cell=" << e
                << ", quadrature_point=" << q << std::scientific
                << std::setprecision(17) << ", determinant=" << det
                << ", tolerance=" << tolerance;
            message << ", reason=near_singular_or_non_finite";
            throw std::invalid_argument(message.str());
        }
        const bool negative = output_determinant[e * quadrature] < 0.;
        for (py::ssize_t q = 1; q < quadrature; ++q) {
            const double det = output_determinant[e * quadrature + q];
            if ((det < 0.) == negative) continue;
            std::ostringstream message;
            message << "invalid element geometry: cell=" << e
                << ", quadrature_point=" << q << std::scientific
                << std::setprecision(17) << ", determinant=" << det
                << ", tolerance=" << output_tolerance[e * quadrature + q]
                << ", reason=orientation_change";
            throw std::invalid_argument(message.str());
        }
    }
    return py::make_tuple(
        tabulated_shape, gradients, dx, global, determinants,
        determinant_tolerances, condition_numbers);
}

}  // namespace

void native_fem::bind_basis_geometry(py::module_& module) {
    module.def("tabulate_basis_geometry", &tabulate_basis_geometry);
}
