#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <stdexcept>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using Vector3 = std::array<double, 3>;
using Gradient3 = std::array<Vector3, 10>;

void tetrahedron_shape(
    const Vector3& reference, py::ssize_t nodes,
    std::array<double, 10>& values, Gradient3& gradients) {
    const std::array<double, 4> barycentric{{
        1. - reference[0] - reference[1] - reference[2],
        reference[0], reference[1], reference[2]
    }};
    static constexpr std::array<Vector3, 4> barycentric_gradients{{
        Vector3{-1., -1., -1.}, Vector3{1., 0., 0.},
        Vector3{0., 1., 0.}, Vector3{0., 0., 1.}
    }};
    for (py::ssize_t node = 0; node < 4; ++node) {
        if (nodes == 4) {
            values[static_cast<std::size_t>(node)] = barycentric[node];
            gradients[static_cast<std::size_t>(node)] =
                barycentric_gradients[static_cast<std::size_t>(node)];
        } else {
            values[static_cast<std::size_t>(node)] =
                barycentric[node] * (2. * barycentric[node] - 1.);
            for (int axis = 0; axis < 3; ++axis)
                gradients[static_cast<std::size_t>(node)][axis] =
                    (4. * barycentric[node] - 1.) *
                    barycentric_gradients[static_cast<std::size_t>(node)][axis];
        }
    }
    if (nodes == 10) {
        static constexpr std::array<std::array<int, 2>, 6> pairs{{
            {{0, 1}}, {{1, 2}}, {{2, 0}}, {{0, 3}}, {{1, 3}}, {{2, 3}}
        }};
        for (int edge = 0; edge < 6; ++edge) {
            const int first = pairs[edge][0];
            const int second = pairs[edge][1];
            const int node = 4 + edge;
            values[node] = 4. * barycentric[first] * barycentric[second];
            for (int axis = 0; axis < 3; ++axis) {
                gradients[node][axis] = 4. * (
                    barycentric[second] * barycentric_gradients[first][axis] +
                    barycentric[first] * barycentric_gradients[second][axis]);
            }
        }
    }
}

double determinant3(const std::array<double, 9>& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
}

std::array<double, 9> inverse3(const std::array<double, 9>& a, double det) {
    if (!std::isfinite(det) || det == 0.)
        throw std::runtime_error("isoparametric mapping has a singular Jacobian");
    const double r = 1. / det;
    return {{
        (a[4] * a[8] - a[5] * a[7]) * r,
        (a[2] * a[7] - a[1] * a[8]) * r,
        (a[1] * a[5] - a[2] * a[4]) * r,
        (a[5] * a[6] - a[3] * a[8]) * r,
        (a[0] * a[8] - a[2] * a[6]) * r,
        (a[2] * a[3] - a[0] * a[5]) * r,
        (a[3] * a[7] - a[4] * a[6]) * r,
        (a[1] * a[6] - a[0] * a[7]) * r,
        (a[0] * a[4] - a[1] * a[3]) * r
    }};
}

void mapping_data(
    const double* coordinates, py::ssize_t nodes,
    const std::array<double, 10>& shape, const Gradient3& reference_gradient,
    Vector3& point, std::array<double, 9>& jacobian) {
    point = {0., 0., 0.};
    jacobian.fill(0.);
    for (py::ssize_t node = 0; node < nodes; ++node) {
        for (int axis = 0; axis < 3; ++axis) {
            const double coordinate = coordinates[axis * nodes + node];
            point[axis] += coordinate * shape[static_cast<std::size_t>(node)];
            for (int direction = 0; direction < 3; ++direction)
                jacobian[axis * 3 + direction] += coordinate *
                    reference_gradient[static_cast<std::size_t>(node)][direction];
        }
    }
}

py::tuple evaluate_tet_surface_points(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::array_t<double, py::array::c_style | py::array::forcecast> physical_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> reference_tangents) {
    const auto c = coordinates.request();
    const auto p = physical_points.request();
    const auto t = reference_tangents.request();
    if (c.ndim != 2 || c.shape[0] != 3 || (c.shape[1] != 4 && c.shape[1] != 10) ||
        p.ndim != 2 || p.shape[1] != 3 || t.ndim != 2 ||
        t.shape[0] != 3 || t.shape[1] != 2)
        throw std::invalid_argument("invalid tetrahedral surface evaluation inputs");
    const py::ssize_t nodes = c.shape[1];
    const py::ssize_t count = p.shape[0];
    const auto* coordinate_data = static_cast<const double*>(c.ptr);
    const auto* point_data = static_cast<const double*>(p.ptr);
    const auto* tangent_data = static_cast<const double*>(t.ptr);
    py::array_t<double> values({count, nodes});
    py::array_t<double> gradients({count, nodes, py::ssize_t{3}});
    py::array_t<double> normals({count, py::ssize_t{3}});
    auto* value_data = values.mutable_data();
    auto* gradient_data = gradients.mutable_data();
    auto* normal_data = normals.mutable_data();
    Vector3 centroid{0., 0., 0.};
    for (py::ssize_t node = 0; node < nodes; ++node)
        for (int axis = 0; axis < 3; ++axis)
            centroid[axis] += coordinate_data[axis * nodes + node] / nodes;
    const std::array<Vector3, 2> tangents{{
        Vector3{tangent_data[0], tangent_data[2], tangent_data[4]},
        Vector3{tangent_data[1], tangent_data[3], tangent_data[5]}
    }};

    for (py::ssize_t q = 0; q < count; ++q) {
        Vector3 reference = nodes == 4
            ? Vector3{.25, .25, .25} : Vector3{.25, .25, .25};
        std::array<double, 10> shape{};
        Gradient3 reference_gradient{};
        bool converged = false;
        for (int iteration = 0; iteration < 15; ++iteration) {
            tetrahedron_shape(reference, nodes, shape, reference_gradient);
            Vector3 mapped{};
            std::array<double, 9> jacobian{};
            mapping_data(coordinate_data, nodes, shape, reference_gradient,
                mapped, jacobian);
            const auto inverse = inverse3(jacobian, determinant3(jacobian));
            Vector3 residual{};
            for (int axis = 0; axis < 3; ++axis)
                residual[axis] = mapped[axis] - point_data[q * 3 + axis];
            Vector3 increment{};
            for (int direction = 0; direction < 3; ++direction)
                for (int axis = 0; axis < 3; ++axis)
                    increment[direction] += inverse[direction * 3 + axis] * residual[axis];
            double increment_norm_squared = 0.;
            for (int direction = 0; direction < 3; ++direction) {
                reference[direction] -= increment[direction];
                increment_norm_squared += increment[direction] * increment[direction];
            }
            if (std::sqrt(increment_norm_squared) < 1e-12) {
                converged = true;
                break;
            }
        }
        if (!converged)
            throw std::runtime_error("isoparametric inverse mapping did not converge");

        tetrahedron_shape(reference, nodes, shape, reference_gradient);
        Vector3 mapped{};
        std::array<double, 9> jacobian{};
        mapping_data(coordinate_data, nodes, shape, reference_gradient,
            mapped, jacobian);
        const auto inverse = inverse3(jacobian, determinant3(jacobian));
        Vector3 physical_tangents[2]{};
        for (int tangent = 0; tangent < 2; ++tangent)
            for (int axis = 0; axis < 3; ++axis)
                for (int direction = 0; direction < 3; ++direction)
                    physical_tangents[tangent][axis] +=
                        jacobian[axis * 3 + direction] * tangents[tangent][direction];
        Vector3 normal{
            physical_tangents[0][1] * physical_tangents[1][2] -
                physical_tangents[0][2] * physical_tangents[1][1],
            physical_tangents[0][2] * physical_tangents[1][0] -
                physical_tangents[0][0] * physical_tangents[1][2],
            physical_tangents[0][0] * physical_tangents[1][1] -
                physical_tangents[0][1] * physical_tangents[1][0]
        };
        const double norm = std::sqrt(normal[0] * normal[0] +
            normal[1] * normal[1] + normal[2] * normal[2]);
        if (!(norm > 0.) || !std::isfinite(norm))
            throw std::runtime_error("surface normal is singular");
        for (double& component : normal) component /= norm;
        const double outward = normal[0] * (mapped[0] - centroid[0]) +
            normal[1] * (mapped[1] - centroid[1]) +
            normal[2] * (mapped[2] - centroid[2]);
        if (outward < 0.)
            for (double& component : normal) component = -component;
        for (py::ssize_t node = 0; node < nodes; ++node) {
            value_data[q * nodes + node] = shape[static_cast<std::size_t>(node)];
            for (int axis = 0; axis < 3; ++axis) {
                double physical_gradient = 0.;
                for (int direction = 0; direction < 3; ++direction)
                    physical_gradient +=
                        reference_gradient[static_cast<std::size_t>(node)][direction] *
                        inverse[direction * 3 + axis];
                gradient_data[(q * nodes + node) * 3 + axis] = physical_gradient;
            }
        }
        for (int axis = 0; axis < 3; ++axis)
            normal_data[q * 3 + axis] = normal[axis];
    }
    return py::make_tuple(std::move(values), std::move(gradients), std::move(normals));
}

}  // namespace

namespace native_fem {

void bind_curved_surface(py::module_& module) {
    module.def("evaluate_tet_surface_points", &evaluate_tet_surface_points,
        py::arg("coordinates"), py::arg("physical_points"),
        py::arg("reference_tangents"),
        "Evaluate Tet4/Tet10 shape values, physical gradients, and normals.");
}

}  // namespace native_fem
