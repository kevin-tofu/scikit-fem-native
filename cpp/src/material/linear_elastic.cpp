#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <utility>

#include "native_fem/material_assembler.hpp"
#include "native_fem/material_kernel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py=pybind11;

namespace {

using Assembler=native_fem::MaterialGlobalAssembler<
    native_fem::LinearElasticMaterialKernel>;

}  // namespace

void native_fem::bind_linear_elastic_material(py::module_&module){
    py::class_<Assembler>binding(module,"LinearElasticMaterialAssembler");
    binding
        .def(py::init([](
            py::array_t<std::int64_t,
                py::array::c_style|py::array::forcecast>dofs,
            py::array_t<double,
                py::array::c_style|py::array::forcecast>gradients,
            py::array_t<double,
                py::array::c_style|py::array::forcecast>weights,
            double young,double poisson){
            return Assembler(
                std::move(dofs),std::move(gradients),std::move(weights),
                native_fem::LinearElasticMaterialKernel(young,poisson)
            );
        }))
        .def("evaluate",&Assembler::evaluate,py::arg("u"),py::arg("state"),
             py::arg("with_tangent")=true,py::arg("num_threads")=0,
             py::arg("time_step")=0.)
        .def_property_readonly("indptr",&Assembler::indptr)
        .def_property_readonly("indices",&Assembler::indices)
        .def_property_readonly("values",&Assembler::values)
        .def_property_readonly("ndofs",&Assembler::ndofs)
        .def_property_readonly("state_count",&Assembler::state_count)
        .def_property_readonly("state_size",&Assembler::state_size)
        .def_property_readonly("nelements",&Assembler::nelements);
}
