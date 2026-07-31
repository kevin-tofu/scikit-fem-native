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

using Kernel=native_fem::StandardLinearSolidKernel;
using Assembler=native_fem::MaterialGlobalAssembler<Kernel>;

py::tuple evaluate_standard_linear_solid(
    py::array_t<double,py::array::c_style|py::array::forcecast>strains,
    py::array_t<double,py::array::c_style>state,
    double equilibrium,double branch,double poisson,double relaxation_time,
    double time_step,int requested_threads,double evaluation_time_step){
    const Kernel kernel(
        equilibrium,branch,poisson,relaxation_time,time_step
    );
    auto e=strains.request(),s=state.request();
    if(e.ndim!=2||e.shape[1]!=6||s.ndim!=2||s.shape[0]!=e.shape[0]||
       s.shape[1]!=Kernel::state_size)
        throw std::invalid_argument(
            "standard linear solid arrays must both have shape (n,6)"
        );
    const auto count=static_cast<std::size_t>(e.shape[0]);
    py::array_t<double>stress({e.shape[0],py::ssize_t(6)});
    py::array_t<double>tangent({e.shape[0],py::ssize_t(6),py::ssize_t(6)});
    py::array_t<double>trial_state({e.shape[0],py::ssize_t(6)});
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
                strain+6*point,committed+6*point,out_state+6*point,
                out_tangent+36*point,evaluation_time_step
            );
            std::copy(result.stress.begin(),result.stress.end(),
                      out_stress+6*point);
        }
    });}
    return py::make_tuple(stress,tangent,trial_state);
}

}  // namespace

void native_fem::bind_standard_linear_solid(py::module_&module){
    module.def(
        "evaluate_standard_linear_solid",&evaluate_standard_linear_solid,
        py::arg("strains"),py::arg("state"),
        py::arg("equilibrium_modulus"),py::arg("branch_modulus"),
        py::arg("poisson_ratio"),py::arg("relaxation_time"),
        py::arg("time_step"),py::arg("num_threads")=0,
        py::arg("evaluation_time_step")=0.
    );
    py::class_<Assembler>binding(module,"StandardLinearSolidAssembler");
    binding
        .def(py::init([](
            py::array_t<std::int64_t,
                py::array::c_style|py::array::forcecast>dofs,
            py::array_t<double,
                py::array::c_style|py::array::forcecast>gradients,
            py::array_t<double,
                py::array::c_style|py::array::forcecast>weights,
            double equilibrium,double branch,double poisson,
            double relaxation_time,double time_step){
            return Assembler(
                std::move(dofs),std::move(gradients),std::move(weights),
                Kernel(
                    equilibrium,branch,poisson,relaxation_time,time_step
                )
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
