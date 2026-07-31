#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_fem/continuum_kernel.hpp"
#include "native_fem/python_bindings.hpp"
#include "native_fem/parallel.hpp"

namespace py=pybind11;

class LinearFormAssembler {
public:
    LinearFormAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast>shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights){
        auto d=dofs.request(),s=shape.request(),g=gradients.request(),w=weights.request();
        if(d.ndim!=3)throw std::invalid_argument("dofs must have shape (entities, nodes, components)");
        entities_=d.shape[0];nodes_=d.shape[1];components_=d.shape[2];
        if(s.ndim!=3||s.shape[0]!=d.shape[0]||s.shape[2]!=nodes_)
            throw std::invalid_argument("shape values have an invalid shape");
        quadrature_=s.shape[1];
        if(g.ndim!=4||g.shape[0]!=entities_||g.shape[1]!=quadrature_||g.shape[2]!=nodes_)
            throw std::invalid_argument("shape gradients have an invalid shape");
        dimension_=g.shape[3];
        if(w.ndim!=2||w.shape[0]!=entities_||w.shape[1]!=quadrature_)
            throw std::invalid_argument("weights have an invalid shape");
        dofs_.assign((std::int64_t*)d.ptr,(std::int64_t*)d.ptr+d.size);
        shape_.assign((double*)s.ptr,(double*)s.ptr+s.size);
        gradients_.assign((double*)g.ptr,(double*)g.ptr+g.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        for(auto n:dofs_){if(n<0)throw std::invalid_argument("negative dof index");ndofs_=std::max(ndofs_,std::size_t(n+1));}
        result_.resize(ndofs_);
    }
    py::tuple assemble(py::object value_coefficient,py::object gradient_coefficient,
                       int requested_threads){
        const double*value=nullptr;const double*gradient=nullptr;
        bool constant_value=false,constant_gradient=false;
        py::array_t<double,py::array::c_style>value_holder,gradient_holder;
        if(!value_coefficient.is_none()){
            value_holder=py::cast<py::array_t<double,py::array::c_style>>(value_coefficient);
            auto b=value_holder.request();
            constant_value=b.ndim==1&&b.shape[0]==components_;
            if(!constant_value&&(b.ndim!=3||b.shape[0]!=entities_||
               b.shape[1]!=quadrature_||b.shape[2]!=components_))
                throw std::invalid_argument("value coefficient has an invalid shape");
            value=(double*)b.ptr;
        }
        if(!gradient_coefficient.is_none()){
            gradient_holder=py::cast<py::array_t<double,py::array::c_style>>(gradient_coefficient);
            auto b=gradient_holder.request();
            constant_gradient=b.ndim==2&&b.shape[0]==components_&&
                b.shape[1]==dimension_;
            if(!constant_gradient&&(b.ndim!=4||b.shape[0]!=entities_||
               b.shape[1]!=quadrature_||b.shape[2]!=components_||
               b.shape[3]!=dimension_))
                throw std::invalid_argument("gradient coefficient has an invalid shape");
            gradient=(double*)b.ptr;
        }
        if(!value&&!gradient)throw std::invalid_argument("at least one coefficient is required");
        auto start=std::chrono::steady_clock::now();std::fill(result_.begin(),result_.end(),0.);
        {py::gil_scoped_release release;
        const auto workers=native_fem::effective_threads(
            static_cast<std::size_t>(entities_),requested_threads);
        if(workers>1){
            thread_results_.resize(workers);
            for(auto&local:thread_results_){
                local.resize(ndofs_);
                std::fill(local.begin(),local.end(),0.);
            }
            native_fem::parallel_for_workers(
                static_cast<std::size_t>(entities_),requested_threads,
                [&](std::size_t worker,std::size_t begin,std::size_t end){
                auto&output=thread_results_[worker];
                assemble_range(
                    static_cast<int>(begin),static_cast<int>(end),output,
                    value,gradient,constant_value,constant_gradient);
            });
            native_fem::parallel_for_workers(
                ndofs_,requested_threads,
                [&](std::size_t,std::size_t begin,std::size_t end){
                for(std::size_t dof=begin;dof<end;++dof)
                    for(const auto&local:thread_results_)
                        result_[dof]+=local[dof];
            });
        }else{
            assemble_range(0,entities_,result_,value,gradient,
                           constant_value,constant_gradient);
        }
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return py::make_tuple(view(result_),seconds);
    }
    std::size_t ndofs()const{return ndofs_;}int entities()const{return entities_;}
    int quadrature_points()const{return quadrature_;}
private:
    void assemble_range(int begin,int end,std::vector<double>&output,
                        const double*value,const double*gradient,
                        bool constant_value,bool constant_gradient){
        if((!value||constant_value)&&(!gradient||constant_gradient)){
            for(int e=begin;e<end;++e)for(int a=0;a<nodes_;++a)
                for(int c=0;c<components_;++c){
                    double local=0.;
                    for(int q=0;q<quadrature_;++q){
                        double contribution=0.;
                        if(value)contribution+=
                            shape_[(e*quadrature_+q)*nodes_+a]*value[c];
                        if(gradient)for(int j=0;j<dimension_;++j)
                            contribution+=gradients_[
                                ((e*quadrature_+q)*nodes_+a)*dimension_+j]*
                                gradient[c*dimension_+j];
                        local+=weights_[e*quadrature_+q]*contribution;
                    }
                    output[dofs_[(e*nodes_+a)*components_+c]]+=local;
                }
        }else{
            for(int e=begin;e<end;++e)for(int q=0;q<quadrature_;++q)
                for(int a=0;a<nodes_;++a)for(int c=0;c<components_;++c){
                    double contribution=0.;
                    if(value)contribution+=shape_[(e*quadrature_+q)*nodes_+a]*
                        value[constant_value?c:(e*quadrature_+q)*components_+c];
                    if(gradient)for(int j=0;j<dimension_;++j)
                        contribution+=gradients_[
                            ((e*quadrature_+q)*nodes_+a)*dimension_+j]*
                            gradient[constant_gradient?c*dimension_+j:
                                ((e*quadrature_+q)*components_+c)*dimension_+j];
                    output[dofs_[(e*nodes_+a)*components_+c]]+=
                        weights_[e*quadrature_+q]*contribution;
                }
        }
    }
    template<class T>py::array view(std::vector<T>&v){return py::array_t<T>(
        {py::ssize_t(v.size())},{py::ssize_t(sizeof(T))},v.data(),py::cast(this));}
    int entities_,nodes_,components_,quadrature_,dimension_;std::size_t ndofs_{};
    std::vector<std::int64_t>dofs_;std::vector<double>shape_,gradients_,weights_,result_;
    std::vector<std::vector<double>>thread_results_;
};

void native_fem::bind_linear_form_assembler(py::module_&m){py::class_<LinearFormAssembler>(m,"LinearFormAssembler")
    .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>>())
    .def("assemble",&LinearFormAssembler::assemble,py::arg("value_coefficient")=py::none(),py::arg("gradient_coefficient")=py::none(),py::arg("num_threads")=0)
    .def_property_readonly("ndofs",&LinearFormAssembler::ndofs)
    .def_property_readonly("entity_count",&LinearFormAssembler::entities)
    .def_property_readonly("quadrature_point_count",&LinearFormAssembler::quadrature_points);
}
