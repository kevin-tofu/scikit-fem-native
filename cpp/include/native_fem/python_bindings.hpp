#pragma once

#include <pybind11/pybind11.h>

namespace native_fem {

void bind_h1_assembler(pybind11::module_& module);
void bind_tabulated_assembler(pybind11::module_& module);
void bind_linear_form_assembler(pybind11::module_& module);
void bind_bilinear_form_assembler(pybind11::module_& module);
void bind_cross_bilinear_assembler(pybind11::module_& module);

}  // namespace native_fem
