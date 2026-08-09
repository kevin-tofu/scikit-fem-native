#include <pybind11/pybind11.h>

#include "native_fem/python_bindings.hpp"

PYBIND11_MODULE(_skfn, module) {
    native_fem::bind_runtime(module);
    native_fem::bind_linear_form_assembler(module);
    native_fem::bind_functional_assembler(module);
    native_fem::bind_supermesh_builder(module);
    native_fem::bind_bilinear_form_assembler(module);
    native_fem::bind_cross_bilinear_assembler(module);
    native_fem::bind_cut_form_assemblers(module);
    native_fem::bind_cut_cross_assembler(module);
    native_fem::bind_basis_geometry(module);
}
