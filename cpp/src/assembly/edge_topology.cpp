#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "native_fem/python_bindings.hpp"
#include "native_fem/parallel.hpp"

namespace py = pybind11;

namespace {

using Edge = std::pair<std::int64_t, std::int64_t>;

struct EdgeHash {
    std::size_t operator()(const Edge& edge) const noexcept {
        const auto first = std::hash<std::int64_t>{}(edge.first);
        const auto second = std::hash<std::int64_t>{}(edge.second);
        return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
    }
};

py::tuple build_oriented_edge_topology(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> connectivity,
    int dimension) {
    const auto input = connectivity.request();
    if (input.ndim != 2)
        throw std::invalid_argument("connectivity must have rank two");
    const py::ssize_t corners = dimension == 2 ? 3 : dimension == 3 ? 4 : 0;
    if (corners == 0 || input.shape[0] != corners)
        throw std::invalid_argument("connectivity shape does not match dimension");

    static constexpr std::array<std::array<int, 2>, 6> local_edges{{
        {{0, 1}}, {{1, 2}}, {{2, 0}}, {{0, 3}}, {{1, 3}}, {{2, 3}},
    }};
    const py::ssize_t edge_count = dimension == 2 ? 3 : 6;
    const py::ssize_t cells = input.shape[1];
    const auto* cell_data = static_cast<const std::int64_t*>(input.ptr);
    py::array_t<std::int64_t> element_edges({edge_count, cells});
    py::array_t<std::int8_t> signs({edge_count, cells});
    auto* element_data = element_edges.mutable_data();
    auto* sign_data = signs.mutable_data();
    std::unordered_map<Edge, std::int64_t, EdgeHash> edge_ids;
    edge_ids.reserve(static_cast<std::size_t>(cells * edge_count));
    std::vector<Edge> ordered_edges;
    ordered_edges.reserve(static_cast<std::size_t>(cells * edge_count));

    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (py::ssize_t first = 0; first < corners; ++first)
            for (py::ssize_t second = first + 1; second < corners; ++second)
                if (cell_data[first * cells + cell] ==
                    cell_data[second * cells + cell])
                    throw std::invalid_argument(
                        "cell " + std::to_string(cell) + " repeats a corner vertex");
        for (py::ssize_t local = 0; local < edge_count; ++local) {
            const auto local_edge = local_edges[static_cast<std::size_t>(local)];
            const auto first = cell_data[local_edge[0] * cells + cell];
            const auto second = cell_data[local_edge[1] * cells + cell];
            const Edge edge{std::min(first, second), std::max(first, second)};
            auto [iterator, inserted] = edge_ids.emplace(
                edge, static_cast<std::int64_t>(ordered_edges.size()));
            if (inserted) ordered_edges.push_back(edge);
            element_data[local * cells + cell] = iterator->second;
            sign_data[local * cells + cell] = first == edge.first ? 1 : -1;
        }
    }

    py::array_t<std::int64_t> edges(py::array::ShapeContainer{
        py::ssize_t{2}, static_cast<py::ssize_t>(ordered_edges.size())});
    auto* edge_data = edges.mutable_data();
    const auto total = static_cast<py::ssize_t>(ordered_edges.size());
    for (py::ssize_t index = 0; index < total; ++index) {
        edge_data[index] = ordered_edges[static_cast<std::size_t>(index)].first;
        edge_data[total + index] = ordered_edges[static_cast<std::size_t>(index)].second;
    }
    return py::make_tuple(edges, element_edges, signs);
}

py::tuple build_edge_csr_pattern(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> element_dofs,
    std::int64_t degrees_of_freedom) {
    const auto input = element_dofs.request();
    if (input.ndim != 2)
        throw std::invalid_argument("element DOFs must have rank two");
    if (degrees_of_freedom < 0)
        throw std::invalid_argument("number of DOFs must be nonnegative");
    const py::ssize_t local_dofs = input.shape[0];
    const py::ssize_t cells = input.shape[1];
    const auto* dofs = static_cast<const std::int64_t*>(input.ptr);
    std::vector<std::vector<std::int64_t>> row_columns(
        static_cast<std::size_t>(degrees_of_freedom));

    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (py::ssize_t row_local = 0; row_local < local_dofs; ++row_local) {
            const auto row = dofs[row_local * cells + cell];
            if (row < 0 || row >= degrees_of_freedom)
                throw std::invalid_argument("element DOF index is out of bounds");
            auto& columns = row_columns[static_cast<std::size_t>(row)];
            for (py::ssize_t column_local = 0; column_local < local_dofs;
                 ++column_local) {
                const auto column = dofs[column_local * cells + cell];
                if (column < 0 || column >= degrees_of_freedom)
                    throw std::invalid_argument("element DOF index is out of bounds");
                columns.push_back(column);
            }
        }
    }

    py::array_t<std::int64_t> indptr(degrees_of_freedom + 1);
    auto* indptr_data = indptr.mutable_data();
    indptr_data[0] = 0;
    for (std::int64_t row = 0; row < degrees_of_freedom; ++row) {
        auto& columns = row_columns[static_cast<std::size_t>(row)];
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        indptr_data[row + 1] = indptr_data[row] +
            static_cast<std::int64_t>(columns.size());
    }

    py::array_t<std::int64_t> indices(indptr_data[degrees_of_freedom]);
    auto* index_data = indices.mutable_data();
    for (std::int64_t row = 0; row < degrees_of_freedom; ++row)
        std::copy(
            row_columns[static_cast<std::size_t>(row)].begin(),
            row_columns[static_cast<std::size_t>(row)].end(),
            index_data + indptr_data[row]);

    py::array_t<std::int64_t> scatter(py::array::ShapeContainer{
        cells, local_dofs, local_dofs});
    auto* scatter_data = scatter.mutable_data();
    for (py::ssize_t cell = 0; cell < cells; ++cell) {
        for (py::ssize_t row_local = 0; row_local < local_dofs; ++row_local) {
            const auto row = dofs[row_local * cells + cell];
            const auto& columns = row_columns[static_cast<std::size_t>(row)];
            for (py::ssize_t column_local = 0; column_local < local_dofs;
                 ++column_local) {
                const auto column = dofs[column_local * cells + cell];
                const auto position = std::lower_bound(
                    columns.begin(), columns.end(), column);
                scatter_data[(cell * local_dofs + row_local) * local_dofs +
                    column_local] = indptr_data[row] +
                    std::distance(columns.begin(), position);
            }
        }
    }
    return py::make_tuple(indptr, indices, scatter);
}

void integrate_tet_n1(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> curls,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style | py::array::forcecast> mass_coefficient,
    py::array_t<double, py::array::c_style | py::array::forcecast> curl_coefficient,
    bool include_mass,
    bool include_curl,
    py::array_t<double, py::array::c_style> output) {
    const auto value_info = values.request();
    const auto curl_info = curls.request();
    const auto dx_info = dx.request();
    const auto mass_info = mass_coefficient.request();
    const auto coefficient_info = curl_coefficient.request();
    const auto output_info = output.request();
    if (value_info.ndim != 4 || curl_info.ndim != 4 || dx_info.ndim != 2 ||
        mass_info.ndim != 2 || coefficient_info.ndim != 2 ||
        output_info.ndim != 3)
        throw std::invalid_argument("invalid TetN1 integration input rank");
    const py::ssize_t cells = value_info.shape[0];
    const py::ssize_t local = value_info.shape[1];
    const py::ssize_t components = value_info.shape[2];
    const py::ssize_t quadrature = value_info.shape[3];
    if (local != 6 || components != 3 || curl_info.shape[0] != cells ||
        curl_info.shape[1] != local || curl_info.shape[2] != components ||
        curl_info.shape[3] != quadrature || dx_info.shape[0] != cells ||
        dx_info.shape[1] != quadrature || mass_info.shape[0] != cells ||
        mass_info.shape[1] != quadrature || coefficient_info.shape[0] != cells ||
        coefficient_info.shape[1] != quadrature || output_info.shape[0] != cells ||
        output_info.shape[1] != local || output_info.shape[2] != local)
        throw std::invalid_argument("incompatible TetN1 integration shapes");

    const auto* value_data = static_cast<const double*>(value_info.ptr);
    const auto* curl_data = static_cast<const double*>(curl_info.ptr);
    const auto* dx_data = static_cast<const double*>(dx_info.ptr);
    const auto* mass_data = static_cast<const double*>(mass_info.ptr);
    const auto* coefficient_data = static_cast<const double*>(coefficient_info.ptr);
    auto* output_data = static_cast<double*>(output_info.ptr);
    {
        py::gil_scoped_release release;
        native_fem::parallel_for(
            static_cast<std::size_t>(cells),
            [&](std::size_t cell_begin, std::size_t cell_end) {
        for (py::ssize_t cell = static_cast<py::ssize_t>(cell_begin);
             cell < static_cast<py::ssize_t>(cell_end); ++cell) {
            for (py::ssize_t row = 0; row < local; ++row) {
                for (py::ssize_t column = 0; column < local; ++column) {
                    double result = 0.;
                    for (py::ssize_t point = 0; point < quadrature; ++point) {
                        double value_dot = 0.;
                        double curl_dot = 0.;
                        for (py::ssize_t component = 0; component < components;
                             ++component) {
                            const auto row_offset =
                                ((cell * local + row) * components + component) *
                                quadrature + point;
                            const auto column_offset =
                                ((cell * local + column) * components + component) *
                                quadrature + point;
                            if (include_mass)
                                value_dot += value_data[row_offset] *
                                    value_data[column_offset];
                            if (include_curl)
                                curl_dot += curl_data[row_offset] *
                                    curl_data[column_offset];
                        }
                        const auto coefficient_offset = cell * quadrature + point;
                        result += dx_data[coefficient_offset] * (
                            mass_data[coefficient_offset] * value_dot +
                            coefficient_data[coefficient_offset] * curl_dot);
                    }
                    output_data[(cell * local + row) * local + column] = result;
                }
            }
        }
        });
    }
}

}  // namespace

void native_fem::bind_edge_topology(py::module_& module) {
    module.def(
        "build_oriented_edge_topology", &build_oriented_edge_topology,
        py::arg("connectivity"), py::arg("dimension"));
    module.def(
        "build_edge_csr_pattern", &build_edge_csr_pattern,
        py::arg("element_dofs"), py::arg("degrees_of_freedom"));
    module.def(
        "integrate_tet_n1", &integrate_tet_n1,
        py::arg("values"), py::arg("curls"), py::arg("dx"),
        py::arg("mass_coefficient"), py::arg("curl_coefficient"),
        py::arg("include_mass"), py::arg("include_curl"), py::arg("output"));
}
