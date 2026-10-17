#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using Point = std::array<double, 2>;

double distance(const Point& left, const Point& right) {
    return std::hypot(left[0] - right[0], left[1] - right[1]);
}

std::vector<Point> clipped_triangle(
    const std::array<double, 3>& values, double tolerance) {
    const std::array<Point, 3> canonical{{
        Point{0., 0.}, Point{1., 0.}, Point{0., 1.}
    }};
    std::vector<Point> polygon;
    for (int edge = 0; edge < 3; ++edge) {
        const int previous = (edge + 2) % 3;
        const bool previous_inside = values[previous] <= tolerance;
        const bool current_inside = values[edge] <= tolerance;
        if (previous_inside != current_inside) {
            const double fraction = values[previous] /
                (values[previous] - values[edge]);
            polygon.push_back({
                canonical[previous][0] + fraction *
                    (canonical[edge][0] - canonical[previous][0]),
                canonical[previous][1] + fraction *
                    (canonical[edge][1] - canonical[previous][1]),
            });
        }
        if (current_inside) polygon.push_back(canonical[edge]);
    }
    std::vector<Point> unique;
    for (const auto& point : polygon) {
        const bool duplicate = std::any_of(
            unique.begin(), unique.end(), [&](const Point& other) {
                return distance(point, other) <= 1.e-13;
            });
        if (!duplicate) unique.push_back(point);
    }
    return unique;
}

std::pair<std::vector<double>, std::vector<double>> gauss_legendre_unit(
    int count) {
    std::vector<double> points(static_cast<std::size_t>(count));
    std::vector<double> weights(static_cast<std::size_t>(count));
    const double pi = std::acos(-1.);
    for (int index = 0; index < (count + 1) / 2; ++index) {
        double root = std::cos(pi * (index + .75) / (count + .5));
        double derivative = 0.;
        for (int iteration = 0; iteration < 32; ++iteration) {
            double previous = 1.;
            double current = root;
            for (int degree = 2; degree <= count; ++degree) {
                const double next = ((2. * degree - 1.) * root * current -
                    (degree - 1.) * previous) / degree;
                previous = current;
                current = next;
            }
            const double polynomial = count == 1 ? root : current;
            const double prior = count == 1 ? 1. : previous;
            derivative = count * (root * polynomial - prior) /
                (root * root - 1.);
            const double step = polynomial / derivative;
            root -= step;
            if (std::abs(step) < 2.e-16) break;
        }
        const double weight = 1. / ((1. - root * root) * derivative * derivative);
        const double left = .5 * (1. - root);
        const double right = .5 * (1. + root);
        points[static_cast<std::size_t>(index)] = left;
        points[static_cast<std::size_t>(count - 1 - index)] = right;
        weights[static_cast<std::size_t>(index)] = weight;
        weights[static_cast<std::size_t>(count - 1 - index)] = weight;
    }
    return {points, weights};
}

struct Rule {
    std::vector<std::array<double, 3>> barycentric;
    std::vector<double> weights;
};

Rule triangle_rule(int intorder) {
    if (intorder == 1)
        return {{{{1. / 3., 1. / 3., 1. / 3.}}}, {0.5}};
    if (intorder == 2)
        return {{
            {{2. / 3., 1. / 6., 1. / 6.}},
            {{1. / 6., 2. / 3., 1. / 6.}},
            {{1. / 6., 1. / 6., 2. / 3.}},
        }, {1. / 6., 1. / 6., 1. / 6.}};
    const int count = std::max(2, static_cast<int>(std::ceil((intorder + 2.) / 2.)));
    const auto [points, weights] = gauss_legendre_unit(count);
    Rule result;
    result.barycentric.reserve(static_cast<std::size_t>(count * count));
    result.weights.reserve(static_cast<std::size_t>(count * count));
    for (int first = 0; first < count; ++first) {
        for (int second = 0; second < count; ++second) {
            const double x = points[static_cast<std::size_t>(first)];
            const double y = (1. - x) * points[static_cast<std::size_t>(second)];
            result.barycentric.push_back({1. - x - y, x, y});
            result.weights.push_back(
                weights[static_cast<std::size_t>(first)] *
                weights[static_cast<std::size_t>(second)] * (1. - x));
        }
    }
    return result;
}

py::tuple cut_triangle_quadrature(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> connectivity,
    py::array_t<double, py::array::c_style | py::array::forcecast> level_values,
    double tolerance, int intorder, bool inside) {
    const auto p = coordinates.request();
    const auto t = connectivity.request();
    const auto phi = level_values.request();
    if (p.ndim != 2 || p.shape[0] != 2 || t.ndim != 2 || t.shape[0] != 3 ||
        phi.ndim != 1 || phi.shape[0] != p.shape[1])
        throw std::invalid_argument("Tri3 cut quadrature expects 2D points and 3-node cells");
    if (intorder < 1) throw std::invalid_argument("intorder must be positive");

    const auto* point_data = static_cast<const double*>(p.ptr);
    const auto* cell_data = static_cast<const std::int64_t*>(t.ptr);
    const auto* phi_data = static_cast<const double*>(phi.ptr);
    const py::ssize_t nodes = p.shape[1];
    const py::ssize_t cells = t.shape[1];
    const double sign = inside ? 1. : -1.;
    const Rule rule = triangle_rule(intorder);
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(cells + 1), 0);
    std::vector<double> output_points, output_reference, output_weights, output_normals;
    std::vector<std::int64_t> output_cells;

    {
    py::gil_scoped_release release;
    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        std::array<std::int64_t, 3> vertex{};
        std::array<Point, 3> physical{};
        std::array<double, 3> values{};
        for (int local = 0; local < 3; ++local) {
            vertex[local] = cell_data[local * cells + cell];
            if (vertex[local] < 0 || vertex[local] >= nodes)
                throw std::invalid_argument("connectivity index is out of bounds");
            physical[local] = {
                point_data[vertex[local]], point_data[nodes + vertex[local]]
            };
            values[local] = sign * phi_data[vertex[local]];
        }
        const auto polygon = clipped_triangle(values, tolerance);
        const double j00 = physical[1][0] - physical[0][0];
        const double j01 = physical[2][0] - physical[0][0];
        const double j10 = physical[1][1] - physical[0][1];
        const double j11 = physical[2][1] - physical[0][1];
        const double determinant = j00 * j11 - j01 * j10;
        const double determinant_abs = std::abs(determinant);
        if (determinant_abs == 0.)
            throw std::invalid_argument("cut quadrature encountered a degenerate cell");
        const double dv0 = values[1] - values[0];
        const double dv1 = values[2] - values[0];
        double nx = (dv0 * j11 - dv1 * j10) / determinant;
        double ny = (j00 * dv1 - j01 * dv0) / determinant;
        const double normal_length = std::hypot(nx, ny);
        if (normal_length > 0.) { nx /= normal_length; ny /= normal_length; }
        else { nx = 0.; ny = 0.; }

        for (std::size_t triangle = 1; triangle + 1 < polygon.size(); ++triangle) {
            const std::array<Point, 3> subtriangle{{
                polygon[0], polygon[triangle], polygon[triangle + 1]
            }};
            const Point a{
                physical[0][0] + subtriangle[1][0] * j00 + subtriangle[1][1] * j01,
                physical[0][1] + subtriangle[1][0] * j10 + subtriangle[1][1] * j11
            };
            const Point b{
                physical[0][0] + subtriangle[2][0] * j00 + subtriangle[2][1] * j01,
                physical[0][1] + subtriangle[2][0] * j10 + subtriangle[2][1] * j11
            };
            const Point origin{
                physical[0][0] + subtriangle[0][0] * j00 + subtriangle[0][1] * j01,
                physical[0][1] + subtriangle[0][0] * j10 + subtriangle[0][1] * j11
            };
            const double subdet = std::abs(
                (a[0] - origin[0]) * (b[1] - origin[1]) -
                (b[0] - origin[0]) * (a[1] - origin[1]));
            for (std::size_t q = 0; q < rule.weights.size(); ++q) {
                const auto& bary = rule.barycentric[q];
                const Point reference{
                    bary[0] * subtriangle[0][0] + bary[1] * subtriangle[1][0] +
                        bary[2] * subtriangle[2][0],
                    bary[0] * subtriangle[0][1] + bary[1] * subtriangle[1][1] +
                        bary[2] * subtriangle[2][1]
                };
                output_reference.push_back(reference[0]);
                output_reference.push_back(reference[1]);
                output_points.push_back(
                    physical[0][0] + reference[0] * j00 + reference[1] * j01);
                output_points.push_back(
                    physical[0][1] + reference[0] * j10 + reference[1] * j11);
                output_weights.push_back(rule.weights[q] * subdet);
                output_normals.push_back(nx);
                output_normals.push_back(ny);
                output_cells.push_back(cell);
            }
        }
        offsets[static_cast<std::size_t>(cell + 1)] =
            static_cast<std::int64_t>(output_weights.size());
    }
    }

    const auto quadrature_points = static_cast<py::ssize_t>(output_weights.size());
    py::array_t<std::int64_t> cell_offsets(cells + 1);
    py::array_t<double> points({quadrature_points, py::ssize_t{2}});
    py::array_t<double> reference_points({quadrature_points, py::ssize_t{2}});
    py::array_t<double> weights(quadrature_points);
    py::array_t<std::int64_t> result_cells(quadrature_points);
    py::array_t<double> normals({quadrature_points, py::ssize_t{2}});
    std::copy(offsets.begin(), offsets.end(), cell_offsets.mutable_data());
    std::copy(output_points.begin(), output_points.end(), points.mutable_data());
    std::copy(output_reference.begin(), output_reference.end(), reference_points.mutable_data());
    std::copy(output_weights.begin(), output_weights.end(), weights.mutable_data());
    std::copy(output_cells.begin(), output_cells.end(), result_cells.mutable_data());
    std::copy(output_normals.begin(), output_normals.end(), normals.mutable_data());
    return py::make_tuple(
        cell_offsets, points, reference_points, weights, result_cells, normals);
}

}  // namespace

namespace native_fem {

void bind_cut_quadrature(py::module_& module) {
    module.def("cut_triangle_quadrature", &cut_triangle_quadrature,
        py::arg("coordinates"), py::arg("connectivity"),
        py::arg("level_values"), py::arg("tolerance"), py::arg("intorder"),
        py::arg("inside"));
}

}  // namespace native_fem
