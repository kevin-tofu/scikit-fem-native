#pragma once

#include <pybind11/pybind11.h>

namespace native_fem {

void bind_linear_form_assembler(pybind11::module_& module);
void bind_functional_assembler(pybind11::module_& module);
void bind_supermesh_builder(pybind11::module_& module);
void bind_bilinear_form_assembler(pybind11::module_& module);
void bind_cross_bilinear_assembler(pybind11::module_& module);
void bind_cut_form_assemblers(pybind11::module_& module);
void bind_cut_cross_assembler(pybind11::module_& module);
void bind_basis_geometry(pybind11::module_& module);
void bind_edge_topology(pybind11::module_& module);
void bind_runtime(pybind11::module_& module);

}  // namespace native_fem
