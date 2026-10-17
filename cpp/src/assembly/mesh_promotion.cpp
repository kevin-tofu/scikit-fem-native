#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using Edge = std::pair<std::int64_t, std::int64_t>;

Edge canonical_edge(std::int64_t a, std::int64_t b) {
    return a < b ? Edge{a, b} : Edge{b, a};
}

py::tuple promote_tri_p2_mesh(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> connectivity) {
    const auto p = coordinates.request();
    const auto t = connectivity.request();
    if (p.ndim != 2 || p.shape[0] != 2 || t.ndim != 2 || t.shape[0] != 3) {
        throw std::invalid_argument(
            "linear triangle mesh requires coordinates (2, n) and connectivity (3, e)");
    }
    const py::ssize_t point_count = p.shape[1];
    const py::ssize_t cells = t.shape[1];
    const auto* point_data = static_cast<const double*>(p.ptr);
    const auto* cell_data = static_cast<const std::int64_t*>(t.ptr);
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(cells * 3));
    static constexpr std::array<std::array<int, 2>, 3> local_edges{{
        {{0, 1}}, {{1, 2}}, {{0, 2}},
    }};
    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (const auto& edge : local_edges) {
            const auto first = cell_data[edge[0] * cells + cell];
            const auto second = cell_data[edge[1] * cells + cell];
            if (first < 0 || first >= point_count || second < 0 || second >= point_count)
                throw std::invalid_argument("triangle connectivity index is out of bounds");
            edges.push_back(canonical_edge(first, second));
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    py::array_t<double> promoted_coordinates({py::ssize_t{2},
        point_count + static_cast<py::ssize_t>(edges.size())});
    py::array_t<std::int64_t> promoted_connectivity({py::ssize_t{6}, cells});
    auto* output_points = promoted_coordinates.mutable_data();
    auto* output_cells = promoted_connectivity.mutable_data();
    for (py::ssize_t axis = 0; axis < 2; ++axis) {
        std::copy(point_data + axis * point_count,
            point_data + (axis + 1) * point_count,
            output_points + axis * (point_count + edges.size()));
    }
    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (py::ssize_t local = 0; local < 3; ++local)
            output_cells[local * cells + cell] = cell_data[local * cells + cell];
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto [first, second] = edges[edge_index];
        const auto node = point_count + static_cast<py::ssize_t>(edge_index);
        for (py::ssize_t axis = 0; axis < 2; ++axis) {
            output_points[axis * (point_count + edges.size()) + node] = .5 * (
                point_data[axis * point_count + first] +
                point_data[axis * point_count + second]);
        }
    }
    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (py::ssize_t local = 0; local < 3; ++local) {
            const auto pair = local_edges[static_cast<std::size_t>(local)];
            const auto first = cell_data[pair[0] * cells + cell];
            const auto second = cell_data[pair[1] * cells + cell];
            const auto edge = canonical_edge(first, second);
            const auto found = std::lower_bound(edges.begin(), edges.end(), edge);
            const auto index = static_cast<py::ssize_t>(found - edges.begin());
            output_cells[(3 + local) * cells + cell] = point_count + index;
        }
    }
    return py::make_tuple(std::move(promoted_coordinates),
        std::move(promoted_connectivity));
}

}  // namespace

namespace native_fem {

void bind_mesh_promotion(py::module_& module) {
    module.def("promote_tri_p2_mesh", &promote_tri_p2_mesh,
        py::arg("coordinates"), py::arg("connectivity"),
        "Promote a linear triangle mesh to straight-sided P2 connectivity.");
}

}  // namespace native_fem
