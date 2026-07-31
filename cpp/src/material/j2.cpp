#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "native_fem/material_assembler.hpp"
#include "native_fem/material_kernel.hpp"
#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py=pybind11;

namespace {

using J2GlobalAssembler=native_fem::MaterialGlobalAssembler<
    native_fem::J2MaterialKernel>;

py::tuple evaluate_j2_state(
    py::array_t<double,py::array::c_style|py::array::forcecast>strains,
    py::array_t<double,py::array::c_style>state,
    double young,double poisson,double yield_stress,double hardening,
    int requested_threads){
    const native_fem::J2MaterialKernel kernel(
        young,poisson,yield_stress,hardening
    );
    auto e=strains.request(),s=state.request();
    if(e.ndim!=2||e.shape[1]!=6||s.ndim!=2||s.shape[0]!=e.shape[0]||
       s.shape[1]!=native_fem::J2MaterialKernel::state_size)
        throw std::invalid_argument(
            "J2 arrays must have shapes (n,6) and (n,7)"
        );
    const auto count=static_cast<std::size_t>(e.shape[0]);
    py::array_t<double>stress({e.shape[0],py::ssize_t(6)});
    py::array_t<double>tangent({e.shape[0],py::ssize_t(6),py::ssize_t(6)});
    py::array_t<double>trial_state(
        {e.shape[0],py::ssize_t(native_fem::J2MaterialKernel::state_size)}
    );
    const auto*strain=static_cast<const double*>(e.ptr);
    const auto*committed=static_cast<const double*>(s.ptr);
    auto*out_stress=static_cast<double*>(stress.request().ptr);
    auto*out_tangent=static_cast<double*>(tangent.request().ptr);
    auto*out_state=static_cast<double*>(trial_state.request().ptr);
    {py::gil_scoped_release release;
    native_fem::parallel_for_workers(
        count,requested_threads,
        [&](std::size_t,std::size_t begin,std::size_t end){
        for(std::size_t point=begin;point<end;++point){
            const auto result=kernel.update(
                strain+6*point,
                committed+native_fem::J2MaterialKernel::state_size*point,
                out_state+native_fem::J2MaterialKernel::state_size*point,
                out_tangent+36*point
            );
            std::copy(result.stress.begin(),result.stress.end(),
                      out_stress+6*point);
        }
    });}
    return py::make_tuple(stress,tangent,trial_state);
}

template<class Assembler>
void bind_assembler_properties(py::class_<Assembler>&binding){
    binding
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

}  // namespace

void native_fem::bind_j2_material(py::module_&module){
    module.def("evaluate_j2_state",&evaluate_j2_state,py::arg("strains"),
        py::arg("state"),py::arg("young_modulus"),py::arg("poisson_ratio"),
        py::arg("yield_stress"),py::arg("hardening_modulus"),
        py::arg("num_threads")=0);

    py::class_<J2GlobalAssembler>j2(module,"J2GlobalAssembler");
    j2.def(py::init([](
        py::array_t<std::int64_t,
            py::array::c_style|py::array::forcecast>dofs,
        py::array_t<double,
            py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,
            py::array::c_style|py::array::forcecast>weights,
        double young,double poisson,double yield,double hardening){
        return J2GlobalAssembler(
            std::move(dofs),std::move(gradients),std::move(weights),
            native_fem::J2MaterialKernel(young,poisson,yield,hardening)
        );
    }));
    bind_assembler_properties(j2);

}
