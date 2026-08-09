#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_fem/python_bindings.hpp"
#include "native_fem/parallel.hpp"

namespace py=pybind11;

class BilinearFormAssembler {
public:
    BilinearFormAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast>shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights){
        auto d=dofs.request(),s=shape.request(),g=gradients.request(),w=weights.request();
        if(d.ndim!=3)throw std::invalid_argument("dofs must have shape (entities, nodes, components)");
        entities_=d.shape[0];nodes_=d.shape[1];components_=d.shape[2];local_dofs_=nodes_*components_;
        if(s.ndim!=3||s.shape[0]!=entities_||s.shape[2]!=nodes_)throw std::invalid_argument("invalid shape values");
        quadrature_=s.shape[1];
        if(g.ndim!=4||g.shape[0]!=entities_||g.shape[1]!=quadrature_||g.shape[2]!=nodes_)
            throw std::invalid_argument("invalid gradients");
        dimension_=g.shape[3];
        if(w.ndim!=2||w.shape[0]!=entities_||w.shape[1]!=quadrature_)throw std::invalid_argument("invalid weights");
        dofs_.assign((std::int64_t*)d.ptr,(std::int64_t*)d.ptr+d.size);
        shape_.assign((double*)s.ptr,(double*)s.ptr+s.size);
        gradients_.assign((double*)g.ptr,(double*)g.ptr+g.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        for(auto n:dofs_)ndofs_=std::max(ndofs_,std::size_t(n+1));
        build_pattern();build_coloring();
    }
    py::tuple assemble(py::object value_coefficient,
                       py::object gradient_coefficient,int requested_threads){
        const double*value=nullptr;const double*gradient=nullptr;
        py::array_t<double,py::array::c_style>vh,gh;
        if(!value_coefficient.is_none()){vh=py::cast<py::array_t<double,py::array::c_style>>(value_coefficient);
            auto b=vh.request();validate_coefficient(b,"value");value=(double*)b.ptr;}
        if(!gradient_coefficient.is_none()){gh=py::cast<py::array_t<double,py::array::c_style>>(gradient_coefficient);
            auto b=gh.request();validate_coefficient(b,"gradient");gradient=(double*)b.ptr;}
        if(!value&&!gradient)throw std::invalid_argument("at least one coefficient is required");
        auto start=std::chrono::steady_clock::now();std::fill(values_.begin(),values_.end(),0.);
        {py::gil_scoped_release release;
        if(native_fem::effective_threads(
               static_cast<std::size_t>(entities_),requested_threads)<=1){
            for(int e=0;e<entities_;++e)
                assemble_element(e,value,gradient);
        }else{
            for(const auto&color:colors_){
                native_fem::parallel_for_workers(
                    color.size(),requested_threads,
                    [&](std::size_t,std::size_t begin,std::size_t end){
                    for(std::size_t index=begin;index<end;++index)
                        assemble_element(color[index],value,gradient);
                });
            }
        }}
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return py::make_tuple(view(values_),seconds);
    }
    py::array indptr(){return view(indptr_);}py::array indices(){return view(indices_);}
    py::array values(){return view(values_);}std::size_t ndofs()const{return ndofs_;}
    std::size_t color_count()const{return colors_.size();}
private:
    void assemble_element(int e,const double*value,const double*gradient){
        for(int q=0;q<quadrature_;++q)
            for(int a=0;a<nodes_;++a)for(int b=0;b<nodes_;++b){
                double entry=0.;const int eq=e*quadrature_+q;
                if(value)entry+=value[eq]*shape_[(eq*nodes_)+a]*shape_[(eq*nodes_)+b];
                if(gradient)for(int j=0;j<dimension_;++j)
                    entry+=gradient[eq]*gradients_[(eq*nodes_+a)*dimension_+j]*
                        gradients_[(eq*nodes_+b)*dimension_+j];
                entry*=weights_[eq];
                for(int c=0;c<components_;++c){
                    int i=a*components_+c,k=b*components_+c;
                    values_[scatter_[(e*local_dofs_+i)*local_dofs_+k]]+=entry;
                }
            }
    }
    void validate_coefficient(const py::buffer_info&b,const char*name){
        if(b.ndim!=2||b.shape[0]!=entities_||b.shape[1]!=quadrature_)
            throw std::invalid_argument(std::string(name)+" coefficient has an invalid shape");}
    template<class T>py::array view(std::vector<T>&v){return py::array_t<T>(
        {py::ssize_t(v.size())},{py::ssize_t(sizeof(T))},v.data(),py::cast(this));}
    void build_pattern(){
        std::vector<std::vector<std::int64_t>>rows(ndofs_);
        for(int e=0;e<entities_;++e)for(int i=0;i<local_dofs_;++i)for(int j=0;j<local_dofs_;++j)
            rows[dofs_[e*local_dofs_+i]].push_back(dofs_[e*local_dofs_+j]);
        indptr_.resize(ndofs_+1);for(std::size_t r=0;r<ndofs_;++r){auto&x=rows[r];std::sort(x.begin(),x.end());
            x.erase(std::unique(x.begin(),x.end()),x.end());indices_.insert(indices_.end(),x.begin(),x.end());indptr_[r+1]=indices_.size();}
        values_.resize(indices_.size());scatter_.resize(entities_*local_dofs_*local_dofs_);
        for(int e=0;e<entities_;++e)for(int i=0;i<local_dofs_;++i){auto row=dofs_[e*local_dofs_+i];
            for(int j=0;j<local_dofs_;++j){auto begin=indices_.begin()+indptr_[row],end=indices_.begin()+indptr_[row+1];
                scatter_[(e*local_dofs_+i)*local_dofs_+j]=std::lower_bound(begin,end,dofs_[e*local_dofs_+j])-indices_.begin();}}}
    void build_coloring(){
        std::vector<std::vector<int>>dof_colors(ndofs_);
        std::vector<int>marks;
        int generation=0;
        for(int e=0;e<entities_;++e){
            ++generation;
            for(int i=0;i<local_dofs_;++i)
                for(const int color:dof_colors[dofs_[e*local_dofs_+i]]){
                    if(color>=static_cast<int>(marks.size()))marks.resize(color+1);
                    marks[color]=generation;
                }
            int color=0;
            while(color<static_cast<int>(marks.size())&&marks[color]==generation)
                ++color;
            if(color==static_cast<int>(colors_.size()))colors_.emplace_back();
            colors_[color].push_back(e);
            for(int i=0;i<local_dofs_;++i)
                dof_colors[dofs_[e*local_dofs_+i]].push_back(color);
        }
    }
    int entities_,nodes_,components_,local_dofs_,quadrature_,dimension_;std::size_t ndofs_{};
    std::vector<std::int64_t>dofs_,indptr_,indices_,scatter_;
    std::vector<double>shape_,gradients_,weights_,values_;
    std::vector<std::vector<int>>colors_;
};

void native_fem::bind_bilinear_form_assembler(py::module_&m){py::class_<BilinearFormAssembler>(m,"BilinearFormAssembler")
    .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>>())
    .def("assemble",&BilinearFormAssembler::assemble,py::arg("value_coefficient")=py::none(),py::arg("gradient_coefficient")=py::none(),py::arg("num_threads")=0)
    .def_property_readonly("indptr",&BilinearFormAssembler::indptr)
    .def_property_readonly("indices",&BilinearFormAssembler::indices)
    .def_property_readonly("values",&BilinearFormAssembler::values)
    .def_property_readonly("ndofs",&BilinearFormAssembler::ndofs);
}
