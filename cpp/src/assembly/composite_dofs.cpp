#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

py::tuple assemble_composite_dofs(
    py::array_t<double, py::array::c_style | py::array::forcecast> coordinates,
    py::sequence active_nodes_input,
    py::sequence connectivity_input,
    std::vector<std::int64_t> components,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> local_layout) {
    const auto p = coordinates.request();
    const auto layout = local_layout.request();
    const py::ssize_t fields = static_cast<py::ssize_t>(components.size());
    if (p.ndim != 2 || (p.shape[0] != 2 && p.shape[0] != 3) || fields < 2 ||
        active_nodes_input.size() != components.size() ||
        connectivity_input.size() != components.size() ||
        layout.ndim != 2 || layout.shape[1] != 2) {
        throw std::invalid_argument("invalid composite degree-of-freedom inputs");
    }
    const py::ssize_t dimension = p.shape[0];
    const py::ssize_t nodes = p.shape[1];
    const auto* point_data = static_cast<const double*>(p.ptr);
    const auto* layout_data = static_cast<const std::int64_t*>(layout.ptr);

    std::vector<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>> active;
    std::vector<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>> connectivity;
    std::vector<py::ssize_t> local_nodes;
    std::vector<py::ssize_t> cells;
    active.reserve(static_cast<std::size_t>(fields));
    connectivity.reserve(static_cast<std::size_t>(fields));
    local_nodes.reserve(static_cast<std::size_t>(fields));
    cells.reserve(static_cast<std::size_t>(fields));
    for (py::ssize_t field = 0; field < fields; ++field) {
        if (components[static_cast<std::size_t>(field)] <= 0)
            throw std::invalid_argument("field component counts must be positive");
        active.emplace_back(py::cast<py::array>(active_nodes_input[field]));
        connectivity.emplace_back(py::cast<py::array>(connectivity_input[field]));
        const auto active_info = active.back().request();
        const auto connectivity_info = connectivity.back().request();
        if (active_info.ndim != 1 || connectivity_info.ndim != 2)
            throw std::invalid_argument("active nodes and connectivity have invalid ranks");
        if (field == 0) {
            cells.push_back(connectivity_info.shape[1]);
        } else if (connectivity_info.shape[1] != cells.front()) {
            throw std::invalid_argument("composite fields have different cell counts");
        }
        local_nodes.push_back(connectivity_info.shape[0]);
        const auto* active_data = static_cast<const std::int64_t*>(active_info.ptr);
        for (py::ssize_t i = 0; i < active_info.shape[0]; ++i) {
            if (active_data[i] < 0 || active_data[i] >= nodes)
                throw std::invalid_argument("active node index is out of bounds");
        }
    }

    std::vector<py::array_t<std::int64_t>> nodal_arrays;
    std::vector<py::array_t<std::int64_t>> element_arrays;
    std::vector<std::int64_t*> nodal_data;
    std::vector<std::int64_t*> element_data;
    std::vector<std::vector<py::ssize_t>> active_position(
        static_cast<std::size_t>(fields), std::vector<py::ssize_t>(
            static_cast<std::size_t>(nodes), -1));
    std::vector<std::vector<std::int64_t>> dof_map(
        static_cast<std::size_t>(fields));
    nodal_arrays.reserve(static_cast<std::size_t>(fields));
    element_arrays.reserve(static_cast<std::size_t>(fields));
    nodal_data.reserve(static_cast<std::size_t>(fields));
    element_data.reserve(static_cast<std::size_t>(fields));
    py::ssize_t next_dof = 0;
    for (py::ssize_t field = 0; field < fields; ++field) {
        const auto count = components[static_cast<std::size_t>(field)];
        const auto active_info = active[static_cast<std::size_t>(field)].request();
        const auto* active_values = static_cast<const std::int64_t*>(active_info.ptr);
        for (py::ssize_t i = 0; i < active_info.shape[0]; ++i) {
            auto& position = active_position[static_cast<std::size_t>(field)]
                [static_cast<std::size_t>(active_values[i])];
            if (position != -1)
                throw std::invalid_argument("active node list contains duplicates");
            position = i;
        }
        nodal_arrays.emplace_back(py::array_t<std::int64_t>(
            std::vector<py::ssize_t>{count, active_info.shape[0]}));
        nodal_data.push_back(nodal_arrays.back().mutable_data());
        const auto cell_count = cells.front();
        const auto dof_rows = local_nodes[static_cast<std::size_t>(field)] * count;
        element_arrays.emplace_back(py::array_t<std::int64_t>(
            std::vector<py::ssize_t>{dof_rows, cell_count}));
        element_data.push_back(element_arrays.back().mutable_data());
        dof_map[static_cast<std::size_t>(field)].assign(
            static_cast<std::size_t>(nodes * count), -1);
    }

    // Preserve the existing global ordering: visit mesh nodes first, then
    // fields, with each field's components contiguous at a node.
    for (py::ssize_t node = 0; node < nodes; ++node) {
        for (py::ssize_t field = 0; field < fields; ++field) {
            const auto field_index = static_cast<std::size_t>(field);
            const auto position = active_position[field_index]
                [static_cast<std::size_t>(node)];
            if (position < 0) continue;
            const auto count = components[field_index];
            auto* nodal = nodal_data[field_index];
            auto& map = dof_map[field_index];
            for (std::int64_t component = 0; component < count; ++component) {
                const auto global = next_dof++;
                nodal[component * (active[field_index].shape(0)) + position] = global;
                map[static_cast<std::size_t>(node * count + component)] = global;
            }
        }
    }

    for (py::ssize_t field = 0; field < fields; ++field) {
        const auto field_index = static_cast<std::size_t>(field);
        const auto count = components[field_index];
        const auto info = connectivity[field_index].request();
        const auto* cell_nodes = static_cast<const std::int64_t*>(info.ptr);
        const auto local_count = local_nodes[field_index];
        const auto cell_count = cells.front();
        auto* output = element_data[field_index];
        const auto& map = dof_map[field_index];
        for (py::ssize_t local = 0; local < local_count; ++local) {
            for (py::ssize_t cell = 0; cell < cell_count; ++cell) {
                const auto node = cell_nodes[local * cell_count + cell];
                if (node < 0 || node >= nodes)
                    throw std::invalid_argument("field connectivity index is out of bounds");
                for (std::int64_t component = 0; component < count; ++component) {
                    const auto global = map[static_cast<std::size_t>(node * count + component)];
                    if (global < 0)
                        throw std::invalid_argument("field connectivity references an inactive node");
                    output[(local * count + component) * cell_count + cell] = global;
                }
            }
        }
    }

    std::vector<std::int64_t> common_nodes;
    common_nodes.reserve(static_cast<std::size_t>(nodes));
    for (py::ssize_t node = 0; node < nodes; ++node) {
        bool common = true;
        for (py::ssize_t field = 0; field < fields; ++field)
            common = common && active_position[static_cast<std::size_t>(field)]
                [static_cast<std::size_t>(node)] >= 0;
        if (common) common_nodes.push_back(node);
    }
    py::ssize_t common_components = 0;
    for (const auto count : components) common_components += count;
    py::array_t<std::int64_t> common_nodal({common_components,
        static_cast<py::ssize_t>(common_nodes.size())});
    auto* common_data = common_nodal.mutable_data();
    py::ssize_t component_offset = 0;
    for (py::ssize_t field = 0; field < fields; ++field) {
        const auto field_index = static_cast<std::size_t>(field);
        const auto count = components[field_index];
        const auto* nodal = nodal_data[field_index];
        const auto active_count = active[field_index].shape(0);
        for (std::size_t column = 0; column < common_nodes.size(); ++column) {
            const auto node = common_nodes[column];
            const auto position = active_position[field_index]
                [static_cast<std::size_t>(node)];
            for (std::int64_t component = 0; component < count; ++component)
                common_data[(component_offset + component) * common_nodes.size() + column] =
                    nodal[component * active_count + position];
        }
        component_offset += count;
    }

    const py::ssize_t composite_rows = layout.shape[0];
    py::array_t<std::int64_t> composite_elements({composite_rows, cells.front()});
    auto* composite_data = composite_elements.mutable_data();
    for (py::ssize_t row = 0; row < composite_rows; ++row) {
        const auto field = layout_data[2 * row];
        const auto dof_row = layout_data[2 * row + 1];
        if (field < 0 || field >= fields || dof_row < 0 ||
            dof_row >= element_arrays[static_cast<std::size_t>(field)].shape(0))
            throw std::invalid_argument("composite local DOF layout is out of bounds");
        const auto* field_element = element_data[static_cast<std::size_t>(field)];
        for (py::ssize_t cell = 0; cell < cells.front(); ++cell)
            composite_data[row * cells.front() + cell] =
                field_element[dof_row * cells.front() + cell];
    }

    py::array_t<double> doflocs({dimension, next_dof});
    auto* dofloc_data = doflocs.mutable_data();
    for (py::ssize_t field = 0; field < fields; ++field) {
        const auto field_index = static_cast<std::size_t>(field);
        const auto count = components[field_index];
        const auto active_info = active[field_index].request();
        const auto* active_values = static_cast<const std::int64_t*>(active_info.ptr);
        const auto* nodal = nodal_data[field_index];
        for (py::ssize_t local = 0; local < active_info.shape[0]; ++local) {
            const auto node = active_values[local];
            for (std::int64_t component = 0; component < count; ++component) {
                const auto global = nodal[component * active_info.shape[0] + local];
                for (py::ssize_t axis = 0; axis < dimension; ++axis)
                    dofloc_data[axis * next_dof + global] =
                        point_data[axis * nodes + node];
            }
        }
    }

    py::tuple nodal_tuple(fields);
    py::tuple element_tuple(fields);
    for (py::ssize_t field = 0; field < fields; ++field) {
        nodal_tuple[field] = std::move(nodal_arrays[static_cast<std::size_t>(field)]);
        element_tuple[field] = std::move(element_arrays[static_cast<std::size_t>(field)]);
    }
    return py::make_tuple(next_dof, std::move(nodal_tuple), std::move(element_tuple),
        std::move(common_nodal), std::move(composite_elements), std::move(doflocs));
}

}  // namespace

namespace native_fem {

void bind_composite_dofs(py::module_& module) {
    module.def("assemble_composite_dofs", &assemble_composite_dofs,
        py::arg("coordinates"), py::arg("active_nodes"),
        py::arg("connectivities"), py::arg("components"),
        py::arg("local_layout"),
        "Build global and element DOF arrays for a composite H1 basis.");
}

}  // namespace native_fem
