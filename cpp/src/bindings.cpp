#include <pybind11/pybind11.h>

#include "native_fem/python_bindings.hpp"

PYBIND11_MODULE(_skfn, module) {
    native_fem::bind_runtime(module);
    native_fem::bind_h1_assembler(module);
    native_fem::bind_tabulated_assembler(module);
    native_fem::bind_linear_form_assembler(module);
    native_fem::bind_functional_assembler(module);
    native_fem::bind_supermesh_builder(module);
    native_fem::bind_bilinear_form_assembler(module);
    native_fem::bind_cross_bilinear_assembler(module);
    native_fem::bind_basis_geometry(module);
    native_fem::bind_j2_material(module);
    native_fem::bind_linear_elastic_material(module);
    native_fem::bind_standard_linear_solid(module);
}
