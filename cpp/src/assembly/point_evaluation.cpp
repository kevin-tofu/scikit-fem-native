#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using Vector2 = std::array<double, 2>;
using Matrix2 = std::array<double, 4>;  // row-major, physical by reference

struct ShapeData {
    std::array<double, 6> values{};
    std::array<Vector2, 6> gradients{};
    int size = 0;
};

ShapeData triangle_shape(const Vector2& reference, bool quadratic) {
    const std::array<double, 3> bary{{
        1. - reference[0] - reference[1], reference[0], reference[1]
    }};
    const std::array<Vector2, 3> bary_gradients{{
        Vector2{-1., -1.}, Vector2{1., 0.}, Vector2{0., 1.}
    }};
    ShapeData result;
    result.size = quadratic ? 6 : 3;
    for (int node = 0; node < 3; ++node) {
        if (quadratic) {
            result.values[node] = bary[node] * (2. * bary[node] - 1.);
            result.gradients[node] = {
                (4. * bary[node] - 1.) * bary_gradients[node][0],
                (4. * bary[node] - 1.) * bary_gradients[node][1]
            };
        } else {
            result.values[node] = bary[node];
            result.gradients[node] = bary_gradients[node];
        }
    }
    if (quadratic) {
        static constexpr int pairs[3][2]{{0, 1}, {1, 2}, {0, 2}};
        for (int edge = 0; edge < 3; ++edge) {
            const int first = pairs[edge][0];
            const int second = pairs[edge][1];
            const int node = 3 + edge;
            result.values[node] = 4. * bary[first] * bary[second];
            result.gradients[node] = {
                4. * (bary[second] * bary_gradients[first][0] +
                    bary[first] * bary_gradients[second][0]),
                4. * (bary[second] * bary_gradients[first][1] +
                    bary[first] * bary_gradients[second][1])
            };
        }
    }
    return result;
}

bool inverse2(const Matrix2& matrix, Matrix2& inverse, double& determinant) {
    determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
    if (std::abs(determinant) <= std::numeric_limits<double>::min()) return false;
    const double reciprocal = 1. / determinant;
    inverse = {matrix[3] * reciprocal, -matrix[1] * reciprocal,
        -matrix[2] * reciprocal, matrix[0] * reciprocal};
    return true;
}

Vector2 mapped_point(
    const double* coordinates, py::ssize_t coordinate_count,
    const std::int64_t* cell_nodes, py::ssize_t cells, py::ssize_t cell,
    const ShapeData& shape) {
    Vector2 mapped{0., 0.};
    for (int node = 0; node < shape.size; ++node) {
        const auto global = cell_nodes[node * cells + cell];
        mapped[0] += coordinates[global] * shape.values[node];
        mapped[1] += coordinates[coordinate_count + global] * shape.values[node];
    }
    return mapped;
}

Matrix2 mapping_jacobian(
    const double* coordinates, py::ssize_t coordinate_count,
    const std::int64_t* cell_nodes, py::ssize_t cells, py::ssize_t cell,
    const ShapeData& shape) {
    Matrix2 jacobian{0., 0., 0., 0.};
    for (int node = 0; node < shape.size; ++node) {
        const auto global = cell_nodes[node * cells + cell];
        const double x = coordinates[global];
        const double y = coordinates[coordinate_count + global];
        jacobian[0] += x * shape.gradients[node][0];
        jacobian[1] += x * shape.gradients[node][1];
        jacobian[2] += y * shape.gradients[node][0];
        jacobian[3] += y * shape.gradients[node][1];
    }
    return jacobian;
}

struct NativePointResult {
    std::int64_t cell = -1;
    Vector2 reference{std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
    bool boundary = false;
    bool ambiguous = false;
    bool outside = true;
};

py::tuple evaluate_triangles(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> geometry_cells,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> global_cell_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> candidate_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> candidates,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> element_dofs,
    py::array_t<double, py::array::c_style | py::array::forcecast> coefficients,
    double physical_tolerance, double reference_tolerance,
    bool geometry_quadratic, int field_order, int components) {
    const auto p = coordinates.request();
    const auto g = geometry_cells.request();
    const auto ids = global_cell_ids.request();
    const auto x = points.request();
    const auto offsets = candidate_offsets.request();
    const auto candidate = candidates.request();
    const auto dofs = element_dofs.request();
    const auto coeff = coefficients.request();
    if (p.ndim != 2 || p.shape[0] != 2 || g.ndim != 2 ||
        (g.shape[0] != 3 && g.shape[0] != 6) || ids.ndim != 1 ||
        ids.shape[0] != g.shape[1] || x.ndim != 2 || x.shape[0] != 2 ||
        offsets.ndim != 1 || offsets.shape[0] != x.shape[1] + 1 ||
        candidate.ndim != 1 || dofs.ndim != 2 || dofs.shape[1] != g.shape[1] ||
        coeff.ndim != 1 || (field_order < 0 || field_order > 2) || components < 1)
        throw std::invalid_argument("invalid triangle point-evaluation array shape");
    const int geometry_nodes = static_cast<int>(g.shape[0]);
    if (geometry_quadratic != (geometry_nodes == 6))
        throw std::invalid_argument("geometry order does not match connectivity");
    const int field_nodes = field_order == 0 ? 1 : field_order == 1 ? 3 : 6;
    if (dofs.shape[0] != field_nodes * components)
        throw std::invalid_argument("element DOF count does not match field shape");

    const auto* coordinate_data = static_cast<const double*>(p.ptr);
    const auto* geometry_data = static_cast<const std::int64_t*>(g.ptr);
    const auto* cell_id_data = static_cast<const std::int64_t*>(ids.ptr);
    const auto* point_data = static_cast<const double*>(x.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offsets.ptr);
    const auto* candidate_data = static_cast<const std::int64_t*>(candidate.ptr);
    const auto* dof_data = static_cast<const std::int64_t*>(dofs.ptr);
    const auto* coefficient_data = static_cast<const double*>(coeff.ptr);
    const py::ssize_t coordinate_count = p.shape[1];
    const py::ssize_t cells = g.shape[1];
    const py::ssize_t point_count = x.shape[1];
    const py::ssize_t dof_count = coeff.shape[0];
    std::vector<NativePointResult> locations(static_cast<std::size_t>(point_count));
    std::vector<double> values(static_cast<std::size_t>(components * point_count),
        std::numeric_limits<double>::quiet_NaN());
    std::vector<double> gradients(static_cast<std::size_t>(components * 2 * point_count),
        std::numeric_limits<double>::quiet_NaN());

    {
        py::gil_scoped_release release;
        for (py::ssize_t point_id = 0; point_id < point_count; ++point_id) {
            const Vector2 point{
                point_data[point_id], point_data[point_count + point_id]
            };
            auto& location = locations[static_cast<std::size_t>(point_id)];
            int hit_count = 0;
            Vector2 chosen_reference{};
            py::ssize_t chosen_local = -1;
            std::int64_t chosen_global = std::numeric_limits<std::int64_t>::max();
            for (std::int64_t index = offset_data[point_id];
                 index < offset_data[point_id + 1]; ++index) {
                const auto local_cell = candidate_data[index];
                if (local_cell < 0 || local_cell >= cells) continue;
                std::array<Vector2, 6> control_points{};
                const int control_count = geometry_quadratic ? 6 : 3;
                for (int node = 0; node < control_count; ++node) {
                    const auto global = geometry_data[node * cells + local_cell];
                    control_points[node] = {
                        coordinate_data[global], coordinate_data[coordinate_count + global]
                    };
                }
                if (geometry_quadratic) {
                    static constexpr int edge_pairs[3][2]{{0, 1}, {1, 2}, {0, 2}};
                    for (int edge = 0; edge < 3; ++edge) {
                        const auto first = edge_pairs[edge][0];
                        const auto second = edge_pairs[edge][1];
                        const Vector2 midpoint = control_points[3 + edge];
                        control_points[3 + edge] = {
                            2. * midpoint[0] - .5 *
                                (control_points[first][0] + control_points[second][0]),
                            2. * midpoint[1] - .5 *
                                (control_points[first][1] + control_points[second][1])
                        };
                    }
                }
                double minimum_x = control_points[0][0];
                double maximum_x = minimum_x;
                double minimum_y = control_points[0][1];
                double maximum_y = minimum_y;
                for (int node = 1; node < control_count; ++node) {
                    minimum_x = std::min(minimum_x, control_points[node][0]);
                    maximum_x = std::max(maximum_x, control_points[node][0]);
                    minimum_y = std::min(minimum_y, control_points[node][1]);
                    maximum_y = std::max(maximum_y, control_points[node][1]);
                }
                if (point[0] < minimum_x - physical_tolerance ||
                    point[0] > maximum_x + physical_tolerance ||
                    point[1] < minimum_y - physical_tolerance ||
                    point[1] > maximum_y + physical_tolerance)
                    continue;

                const auto n0 = geometry_data[local_cell];
                const auto n1 = geometry_data[cells + local_cell];
                const auto n2 = geometry_data[2 * cells + local_cell];
                const Matrix2 corner_jacobian{
                    coordinate_data[n1] - coordinate_data[n0],
                    coordinate_data[n2] - coordinate_data[n0],
                    coordinate_data[coordinate_count + n1] - coordinate_data[coordinate_count + n0],
                    coordinate_data[coordinate_count + n2] - coordinate_data[coordinate_count + n0]
                };
                Matrix2 inverse_corner{};
                double determinant = 0.;
                if (!inverse2(corner_jacobian, inverse_corner, determinant)) continue;
                Vector2 reference{
                    inverse_corner[0] * (point[0] - coordinate_data[n0]) +
                        inverse_corner[1] * (point[1] - coordinate_data[coordinate_count + n0]),
                    inverse_corner[2] * (point[0] - coordinate_data[n0]) +
                        inverse_corner[3] * (point[1] - coordinate_data[coordinate_count + n0])
                };

                bool converged = true;
                if (geometry_quadratic) {
                    converged = false;
                    for (int iteration = 0; iteration < 30; ++iteration) {
                        const ShapeData geometry_shape = triangle_shape(reference, true);
                        const Vector2 mapped = mapped_point(
                            coordinate_data, coordinate_count, geometry_data,
                            cells, local_cell, geometry_shape);
                        const Vector2 residual{mapped[0] - point[0], mapped[1] - point[1]};
                        if (std::max(std::abs(residual[0]), std::abs(residual[1])) <=
                            physical_tolerance) {
                            converged = true;
                            break;
                        }
                        const Matrix2 jacobian = mapping_jacobian(
                            coordinate_data, coordinate_count, geometry_data,
                            cells, local_cell, geometry_shape);
                        Matrix2 inverse_jacobian{};
                        if (!inverse2(jacobian, inverse_jacobian, determinant)) break;
                        const Vector2 step{
                            inverse_jacobian[0] * residual[0] + inverse_jacobian[1] * residual[1],
                            inverse_jacobian[2] * residual[0] + inverse_jacobian[3] * residual[1]
                        };
                        reference[0] -= step[0];
                        reference[1] -= step[1];
                        if (std::max(std::abs(step[0]), std::abs(step[1])) <=
                            reference_tolerance) {
                            const auto final_shape = triangle_shape(reference, true);
                            const auto final_point = mapped_point(
                                coordinate_data, coordinate_count, geometry_data,
                                cells, local_cell, final_shape);
                            converged = std::max(
                                std::abs(final_point[0] - point[0]),
                                std::abs(final_point[1] - point[1])) <= physical_tolerance;
                            break;
                        }
                    }
                }
                if (!converged) continue;
                const Vector2 barycentric{
                    1. - reference[0] - reference[1], reference[0]
                };
                const double third_barycentric = reference[1];
                const double minimum_barycentric = std::min(
                    {barycentric[0], barycentric[1], third_barycentric});
                const double maximum_barycentric = std::max(
                    {barycentric[0], barycentric[1], third_barycentric});
                if (minimum_barycentric < -reference_tolerance ||
                    maximum_barycentric > 1. + reference_tolerance)
                    continue;
                ++hit_count;
                const auto global_cell = cell_id_data[local_cell];
                if (global_cell < chosen_global) {
                    chosen_global = global_cell;
                    chosen_local = local_cell;
                    chosen_reference = reference;
                }
            }
            if (chosen_local < 0) continue;
            location.cell = chosen_global;
            location.reference = chosen_reference;
            location.outside = false;
            location.ambiguous = hit_count > 1;
            const double bary[3]{
                1. - chosen_reference[0] - chosen_reference[1],
                chosen_reference[0], chosen_reference[1]
            };
            location.boundary = std::min({std::abs(bary[0]), std::abs(bary[1]),
                std::abs(bary[2])}) <= reference_tolerance;

            ShapeData field_shape;
            if (field_order == 0) {
                field_shape.size = 1;
                field_shape.values[0] = 1.;
            } else {
                field_shape = triangle_shape(
                    chosen_reference, field_order == 2);
            }
            Matrix2 jacobian{};
            if (geometry_quadratic) {
                jacobian = mapping_jacobian(
                    coordinate_data, coordinate_count, geometry_data, cells,
                    chosen_local, triangle_shape(chosen_reference, true));
            } else {
                const auto n0 = geometry_data[chosen_local];
                const auto n1 = geometry_data[cells + chosen_local];
                const auto n2 = geometry_data[2 * cells + chosen_local];
                jacobian = {
                    coordinate_data[n1] - coordinate_data[n0],
                    coordinate_data[n2] - coordinate_data[n0],
                    coordinate_data[coordinate_count + n1] - coordinate_data[coordinate_count + n0],
                    coordinate_data[coordinate_count + n2] - coordinate_data[coordinate_count + n0]
                };
            }
            Matrix2 inverse_jacobian{};
            double field_determinant = 0.;
            if (!inverse2(jacobian, inverse_jacobian, field_determinant)) {
                location = NativePointResult{};
                continue;
            }
            for (int component = 0; component < components; ++component) {
                double field_value = 0.;
                Vector2 field_gradient{0., 0.};
                for (int node = 0; node < field_nodes; ++node) {
                    const auto dof = dof_data[(node * components + component) * cells + chosen_local];
                    if (dof < 0 || dof >= dof_count) continue;
                    const double coefficient = coefficient_data[dof];
                    field_value += field_shape.values[node] * coefficient;
                    const double gradient_xi = field_shape.gradients[node][0];
                    const double gradient_eta = field_shape.gradients[node][1];
                    field_gradient[0] += coefficient *
                        (gradient_xi * inverse_jacobian[0] +
                         gradient_eta * inverse_jacobian[2]);
                    field_gradient[1] += coefficient *
                        (gradient_xi * inverse_jacobian[1] +
                         gradient_eta * inverse_jacobian[3]);
                }
                values[component * point_count + point_id] = field_value;
                gradients[(component * 2) * point_count + point_id] = field_gradient[0];
                gradients[(component * 2 + 1) * point_count + point_id] = field_gradient[1];
            }
        }
    }

    py::array_t<double> value_output(
        py::array::ShapeContainer{py::ssize_t{components}, point_count});
    py::array_t<double> gradient_output(
        py::array::ShapeContainer{
            py::ssize_t{components}, py::ssize_t{2}, point_count});
    py::array_t<std::int64_t> cell_output(point_count);
    py::array_t<double> reference_output(
        py::array::ShapeContainer{py::ssize_t{2}, point_count});
    py::array_t<bool> boundary_output(point_count);
    py::array_t<bool> ambiguous_output(point_count);
    py::array_t<bool> outside_output(point_count);
    std::copy(values.begin(), values.end(), value_output.mutable_data());
    std::copy(gradients.begin(), gradients.end(), gradient_output.mutable_data());
    auto* cell_ids = cell_output.mutable_data();
    auto* references = reference_output.mutable_data();
    auto* boundary = boundary_output.mutable_data();
    auto* ambiguous = ambiguous_output.mutable_data();
    auto* outside = outside_output.mutable_data();
    for (py::ssize_t point = 0; point < point_count; ++point) {
        const auto& location = locations[static_cast<std::size_t>(point)];
        cell_ids[point] = location.cell;
        references[point] = location.reference[0];
        references[point_count + point] = location.reference[1];
        boundary[point] = location.boundary;
        ambiguous[point] = location.ambiguous;
        outside[point] = location.outside;
    }
    return py::make_tuple(value_output, gradient_output, cell_output,
        reference_output, boundary_output, ambiguous_output, outside_output);
}

}  // namespace

namespace native_fem {

void bind_point_evaluation(py::module_& module) {
    module.def("evaluate_triangles", &evaluate_triangles,
        py::arg("coordinates"), py::arg("geometry_cells"),
        py::arg("global_cell_ids"), py::arg("points"),
        py::arg("candidate_offsets"), py::arg("candidates"),
        py::arg("element_dofs"), py::arg("coefficients"),
        py::arg("physical_tolerance"), py::arg("reference_tolerance"),
        py::arg("geometry_quadratic"), py::arg("field_order"),
        py::arg("components"));
}

}  // namespace native_fem
