#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using V2 = std::array<double, 2>;
using V3 = std::array<double, 3>;

V2 add(V2 a, V2 b) { return {a[0] + b[0], a[1] + b[1]}; }
V2 sub(V2 a, V2 b) { return {a[0] - b[0], a[1] - b[1]}; }
V2 scale(V2 a, double s) { return {s * a[0], s * a[1]}; }
V3 add(V3 a, V3 b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
V3 sub(V3 a, V3 b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
V3 scale(V3 a, double s) {
    return {s * a[0], s * a[1], s * a[2]};
}
double dot(V3 a, V3 b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
V3 cross(V3 a, V3 b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}
double norm(V3 a) { return std::sqrt(dot(a, a)); }
double cross2(V2 a, V2 b) { return a[0] * b[1] - a[1] * b[0]; }

std::vector<V2> clip(
    std::vector<V2> polygon,
    const std::array<V2, 3>& triangle,
    double tolerance
) {
    const double orientation =
        cross2(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
        >= 0.0 ? 1.0 : -1.0;
    for (int edge_index = 0; edge_index < 3 && !polygon.empty(); ++edge_index) {
        const V2 a = triangle[edge_index];
        const V2 b = triangle[(edge_index + 1) % 3];
        const V2 edge = sub(b, a);
        std::vector<V2> output;
        output.reserve(polygon.size() + 1);
        V2 previous = polygon.back();
        bool previous_inside =
            orientation * cross2(edge, sub(previous, a)) >= -tolerance;
        for (const V2 current : polygon) {
            const bool current_inside =
                orientation * cross2(edge, sub(current, a)) >= -tolerance;
            if (current_inside != previous_inside) {
                const V2 segment = sub(current, previous);
                const double denominator = cross2(segment, edge);
                if (std::abs(denominator) > tolerance) {
                    const double parameter =
                        cross2(sub(a, previous), edge) / denominator;
                    output.push_back(add(previous, scale(segment, parameter)));
                }
            }
            if (current_inside) {
                output.push_back(current);
            }
            previous = current;
            previous_inside = current_inside;
        }
        polygon.swap(output);
    }
    return polygon;
}

std::array<double, 3> barycentric(
    V2 point,
    const std::array<V2, 3>& triangle
) {
    const V2 first = sub(triangle[1], triangle[0]);
    const V2 second = sub(triangle[2], triangle[0]);
    const V2 rhs = sub(point, triangle[0]);
    const double determinant = cross2(first, second);
    const double u = cross2(rhs, second) / determinant;
    const double v = cross2(first, rhs) / determinant;
    return {1.0 - u - v, u, v};
}

template <class T>
py::array_t<T> array_from(
    const std::vector<T>& values,
    const std::vector<py::ssize_t>& shape
) {
    py::array_t<T> result(shape);
    std::memcpy(
        result.mutable_data(),
        values.data(),
        values.size() * sizeof(T)
    );
    return result;
}

py::dict build_triangle_supermesh(
    py::array_t<double, py::array::c_style | py::array::forcecast> master_points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        master_triangles,
    py::array_t<double, py::array::c_style | py::array::forcecast> slave_points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        slave_triangles,
    double tolerance,
    double projection_tolerance
) {
    const auto mp = master_points.request();
    const auto mt = master_triangles.request();
    const auto sp = slave_points.request();
    const auto st = slave_triangles.request();
    if (mp.ndim != 2 || mp.shape[0] != 3 ||
        sp.ndim != 2 || sp.shape[0] != 3 ||
        mt.ndim != 2 || mt.shape[0] != 3 ||
        st.ndim != 2 || st.shape[0] != 3) {
        throw std::invalid_argument(
            "points and triangles must have shapes (3, points) and "
            "(3, triangles)"
        );
    }
    const auto* master_point = static_cast<const double*>(mp.ptr);
    const auto* slave_point = static_cast<const double*>(sp.ptr);
    const auto* master_node = static_cast<const std::int64_t*>(mt.ptr);
    const auto* slave_node = static_cast<const std::int64_t*>(st.ptr);
    const int master_count = static_cast<int>(mt.shape[1]);
    const int slave_count = static_cast<int>(st.shape[1]);

    auto point = [](const double* points, py::ssize_t count, std::int64_t node) {
        return V3{
            points[node],
            points[count + node],
            points[2 * count + node],
        };
    };
    auto triangle_node = [](const std::int64_t* triangles, int count, int local, int tri) {
        return triangles[local * count + tri];
    };

    std::vector<std::array<double, 3>> master_min(master_count), master_max(master_count);
    std::vector<std::array<double, 3>> slave_min(slave_count), slave_max(slave_count);
    auto bounds = [&](const double* points, py::ssize_t point_count,
                      const std::int64_t* triangles, int triangle_count,
                      auto& minimum, auto& maximum) {
        for (int triangle = 0; triangle < triangle_count; ++triangle) {
            minimum[triangle] = {
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
            };
            maximum[triangle] = {
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
            };
            for (int local = 0; local < 3; ++local) {
                const V3 xyz = point(
                    points, point_count,
                    triangle_node(triangles, triangle_count, local, triangle)
                );
                for (int d = 0; d < 3; ++d) {
                    minimum[triangle][d] =
                        std::min(minimum[triangle][d], xyz[d]);
                    maximum[triangle][d] =
                        std::max(maximum[triangle][d], xyz[d]);
                }
            }
            for (int d = 0; d < 3; ++d) {
                minimum[triangle][d] -= tolerance;
                maximum[triangle][d] += tolerance;
            }
        }
    };
    const double broad_tolerance = std::max(tolerance, projection_tolerance);
    bounds(master_point, mp.shape[1], master_node, master_count, master_min, master_max);
    bounds(slave_point, sp.shape[1], slave_node, slave_count, slave_min, slave_max);
    for (auto& minimum : master_min) for (double& x : minimum) x -= broad_tolerance - tolerance;
    for (auto& maximum : master_max) for (double& x : maximum) x += broad_tolerance - tolerance;
    for (auto& minimum : slave_min) for (double& x : minimum) x -= broad_tolerance - tolerance;
    for (auto& maximum : slave_max) for (double& x : maximum) x += broad_tolerance - tolerance;

    std::vector<int> master_order(master_count), slave_order(slave_count);
    std::iota(master_order.begin(), master_order.end(), 0);
    std::iota(slave_order.begin(), slave_order.end(), 0);
    std::sort(master_order.begin(), master_order.end(), [&](int a, int b) {
        return master_min[a][0] < master_min[b][0];
    });
    std::sort(slave_order.begin(), slave_order.end(), [&](int a, int b) {
        return slave_min[a][0] < slave_min[b][0];
    });

    constexpr double a = .445948490915965;
    constexpr double b = .091576213509771;
    const std::array<std::array<double, 3>, 6> quadrature{{
        {a, a, 1 - 2 * a}, {a, 1 - 2 * a, a}, {1 - 2 * a, a, a},
        {b, b, 1 - 2 * b}, {b, 1 - 2 * b, b}, {1 - 2 * b, b, b},
    }};
    const std::array<double, 6> quadrature_weight{{
        .223381589678011, .223381589678011, .223381589678011,
        .109951743655322, .109951743655322, .109951743655322,
    }};

    std::vector<std::int64_t> master_indices, slave_indices;
    std::vector<double> row_shape, column_shape, weights, coordinates;
    std::vector<double> master_normals, slave_normals, gaps;
    const std::size_t initial = static_cast<std::size_t>(
        std::max(master_count, slave_count)
    );
    master_indices.reserve(initial);
    slave_indices.reserve(initial);
    row_shape.reserve(initial * 18);
    column_shape.reserve(initial * 18);
    weights.reserve(initial * 6);
    coordinates.reserve(initial * 18);
    master_normals.reserve(initial * 18);
    slave_normals.reserve(initial * 18);
    gaps.reserve(initial * 6);

    std::size_t candidate_count = 0;
    std::size_t overlap_count = 0;
    std::size_t noncoplanar_count = 0;
    double area_total = 0.0;
    double maximum_gap = 0.0;
    std::vector<int> active;
    active.reserve(slave_count);
    std::size_t slave_start = 0;

    {
        py::gil_scoped_release release;
        for (const int master : master_order) {
            while (slave_start < slave_order.size() &&
                   slave_min[slave_order[slave_start]][0] <= master_max[master][0]) {
                active.push_back(slave_order[slave_start++]);
            }
            std::size_t kept = 0;
            for (const int slave : active) {
                if (slave_max[slave][0] >= master_min[master][0]) {
                    active[kept++] = slave;
                }
            }
            active.resize(kept);
            for (const int slave : active) {
                bool intersects = true;
                for (int d = 0; d < 3; ++d) {
                    intersects = intersects &&
                        slave_max[slave][d] >= master_min[master][d] &&
                        slave_min[slave][d] <= master_max[master][d];
                }
                if (!intersects) continue;
                ++candidate_count;

                std::array<V3, 3> master_xyz, slave_xyz;
                for (int local = 0; local < 3; ++local) {
                    master_xyz[local] = point(
                        master_point, mp.shape[1],
                        triangle_node(master_node, master_count, local, master)
                    );
                    slave_xyz[local] = point(
                        slave_point, sp.shape[1],
                        triangle_node(slave_node, slave_count, local, slave)
                    );
                }
                V3 tangent0 = sub(master_xyz[1], master_xyz[0]);
                V3 normal = cross(tangent0, sub(master_xyz[2], master_xyz[0]));
                const double normal_norm = norm(normal);
                const double tangent_norm = norm(tangent0);
                if (normal_norm <= tolerance || tangent_norm <= tolerance) continue;
                normal = scale(normal, 1.0 / normal_norm);
                tangent0 = scale(tangent0, 1.0 / tangent_norm);
                const V3 tangent1 = cross(normal, tangent0);

                std::array<V2, 3> master_2d, slave_2d;
                double plane_gap = 0.0;
                for (int local = 0; local < 3; ++local) {
                    const V3 master_delta = sub(master_xyz[local], master_xyz[0]);
                    const V3 slave_delta = sub(slave_xyz[local], master_xyz[0]);
                    master_2d[local] = {
                        dot(master_delta, tangent0), dot(master_delta, tangent1)
                    };
                    slave_2d[local] = {
                        dot(slave_delta, tangent0), dot(slave_delta, tangent1)
                    };
                    plane_gap = std::max(
                        plane_gap, std::abs(dot(slave_delta, normal))
                    );
                }
                maximum_gap = std::max(maximum_gap, plane_gap);
                if (plane_gap > projection_tolerance) {
                    ++noncoplanar_count;
                    continue;
                }
                std::vector<V2> polygon{
                    master_2d[0], master_2d[1], master_2d[2]
                };
                polygon = clip(std::move(polygon), slave_2d, tolerance);
                if (polygon.size() < 3) continue;
                bool has_area = false;
                for (std::size_t fan = 1; fan + 1 < polygon.size(); ++fan) {
                    const std::array<V2, 3> triangle{{
                        polygon[0], polygon[fan], polygon[fan + 1]
                    }};
                    const double area = .5 * std::abs(cross2(
                        sub(triangle[1], triangle[0]),
                        sub(triangle[2], triangle[0])
                    ));
                    if (area <= tolerance) continue;
                    has_area = true;
                    area_total += area;
                    master_indices.push_back(master);
                    slave_indices.push_back(slave);

                    V3 slave_normal = cross(
                        sub(slave_xyz[1], slave_xyz[0]),
                        sub(slave_xyz[2], slave_xyz[0])
                    );
                    slave_normal = scale(slave_normal, 1.0 / norm(slave_normal));
                    if (dot(slave_normal, normal) > 0.0) {
                        slave_normal = scale(slave_normal, -1.0);
                    }
                    for (int q = 0; q < 6; ++q) {
                        V2 uv{0.0, 0.0};
                        for (int vertex = 0; vertex < 3; ++vertex) {
                            uv = add(uv, scale(
                                triangle[vertex], quadrature[q][vertex]
                            ));
                        }
                        const auto master_value = barycentric(uv, master_2d);
                        const auto slave_value = barycentric(uv, slave_2d);
                        for (double value : master_value) row_shape.push_back(value);
                        for (double value : slave_value) column_shape.push_back(value);
                        const V3 physical = add(
                            master_xyz[0],
                            add(scale(tangent0, uv[0]), scale(tangent1, uv[1]))
                        );
                        for (double value : physical) coordinates.push_back(value);
                        for (double value : normal) master_normals.push_back(value);
                        for (double value : slave_normal) slave_normals.push_back(value);
                        V3 slave_physical{0.0, 0.0, 0.0};
                        for (int vertex = 0; vertex < 3; ++vertex) {
                            slave_physical = add(
                                slave_physical,
                                scale(slave_xyz[vertex], slave_value[vertex])
                            );
                        }
                        gaps.push_back(dot(sub(slave_physical, physical), normal));
                        weights.push_back(area * quadrature_weight[q]);
                    }
                }
                overlap_count += has_area ? 1 : 0;
            }
        }
    }
    if (master_indices.empty()) {
        throw std::invalid_argument(
            "triangle surfaces have no positive-area overlap"
        );
    }
    const py::ssize_t count = static_cast<py::ssize_t>(master_indices.size());
    py::dict result;
    result["master_indices"] = array_from(master_indices, {count});
    result["slave_indices"] = array_from(slave_indices, {count});
    result["row_shape"] = array_from(row_shape, {count, 6, 3});
    result["column_shape"] = array_from(column_shape, {count, 6, 3});
    result["weights"] = array_from(weights, {count, 6});
    result["coordinates"] = array_from(coordinates, {count, 6, 3});
    result["master_normals"] = array_from(master_normals, {count, 6, 3});
    result["slave_normals"] = array_from(slave_normals, {count, 6, 3});
    result["gaps"] = array_from(gaps, {count, 6});
    result["candidate_count"] = candidate_count;
    result["overlap_count"] = overlap_count;
    result["integration_triangle_count"] = master_indices.size();
    result["overlap_area"] = area_total;
    result["noncoplanar_rejection_count"] = noncoplanar_count;
    result["maximum_plane_gap"] = maximum_gap;
    return result;
}

}  // namespace

void native_fem::bind_supermesh_builder(py::module_& module) {
    module.def(
        "build_triangle_supermesh",
        &build_triangle_supermesh,
        py::arg("master_points"),
        py::arg("master_triangles"),
        py::arg("slave_points"),
        py::arg("slave_triangles"),
        py::arg("tolerance") = 1e-10,
        py::arg("projection_tolerance") = 1e-10
    );
}
