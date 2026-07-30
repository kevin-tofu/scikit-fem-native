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

namespace py=pybind11;

class TabulatedAssembler {
public:
    TabulatedAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast> element_dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast> gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast> weights,
        std::string kernel,double p1,double p2)
        :p1_(p1),p2_(p2),nonlinear_(kernel=="neo_hookean"){
        if(kernel!="linear_elastic"&&kernel!="neo_hookean")throw std::invalid_argument("unknown kernel");
        auto d=element_dofs.request(),g=gradients.request(),w=weights.request();
        if(d.ndim!=2||d.shape[1]%3)throw std::invalid_argument("element_dofs must have shape (nelement, 3 * nodes)");
        elements_=d.shape[0];local_dofs_=d.shape[1];nodes_=local_dofs_/3;
        if(nodes_>native_fem::max_nodes)throw std::invalid_argument("element exceeds native maximum node count");
        if(g.ndim!=4||g.shape[0]!=d.shape[0]||g.shape[2]!=nodes_||g.shape[3]!=3)
            throw std::invalid_argument("gradients must have shape (nelement, quadrature, nodes, 3)");
        quadrature_=g.shape[1];
        if(w.ndim!=2||w.shape[0]!=d.shape[0]||w.shape[1]!=quadrature_)
            throw std::invalid_argument("weights must have shape (nelement, quadrature)");
        dofs_.assign((std::int64_t*)d.ptr,(std::int64_t*)d.ptr+d.size);
        gradients_.assign((double*)g.ptr,(double*)g.ptr+g.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        for(auto n:dofs_){if(n<0)throw std::invalid_argument("negative dof index");ndofs_=std::max(ndofs_,std::size_t(n+1));}
        build_pattern();if(!nonlinear_)build_linear();
    }
    py::tuple evaluate(py::array_t<double,py::array::c_style>u,py::object load,bool tangent){
        auto b=u.request();validate(b,ndofs_,"u");
        double seconds=assemble((double*)b.ptr,load_ptr(load),residual_.data(),tangent?values_.data():nullptr);
        py::object v=py::none();if(tangent)v=view(values_);
        return py::make_tuple(view(residual_),v,seconds);
    }
    double evaluate_into(py::array_t<double,py::array::c_style>u,
        py::array_t<double,py::array::c_style>residual,py::object tangent,py::object load){
        auto ub=u.request(),rb=residual.request();validate(ub,ndofs_,"u");validate(rb,ndofs_,"residual");
        double*vp=nullptr;py::array_t<double,py::array::c_style>holder;
        if(!tangent.is_none()){holder=py::cast<py::array_t<double,py::array::c_style>>(tangent);
            auto vb=holder.request();validate(vb,values_.size(),"tangent_values");vp=(double*)vb.ptr;}
        return assemble((double*)ub.ptr,load_ptr(load),(double*)rb.ptr,vp);
    }
    py::array indptr(){return view(indptr_);}py::array indices(){return view(indices_);}
    py::array values(){return view(values_);}std::size_t ndofs()const{return ndofs_;}
    std::size_t nelements()const{return elements_;}
private:
    static void validate(const py::buffer_info&b,std::size_t n,const char*name){
        if(b.ndim!=1||std::size_t(b.shape[0])!=n)throw std::invalid_argument(std::string(name)+" has an invalid shape");}
    const double*load_ptr(const py::object&o){if(o.is_none())return nullptr;
        auto a=py::cast<py::array_t<double,py::array::c_style>>(o);auto b=a.request();validate(b,ndofs_,"loads");return(double*)b.ptr;}
    template<class T>py::array view(std::vector<T>&v){return py::array_t<T>(
        {py::ssize_t(v.size())},{py::ssize_t(sizeof(T))},v.data(),py::cast(this));}
    double assemble(const double*u,const double*load,double*r,double*values){
        auto start=std::chrono::steady_clock::now();std::fill(r,r+ndofs_,0.);
        if(values)std::fill(values,values+values_.size(),0.);
        {py::gil_scoped_release release;
        for(std::size_t e=0;e<elements_;++e){
            double ue[native_fem::max_dofs],re[native_fem::max_dofs];
            double ke[native_fem::max_dofs*native_fem::max_dofs];
            std::fill(ue,ue+local_dofs_,0.);std::fill(re,re+local_dofs_,0.);
            std::fill(ke,ke+local_dofs_*local_dofs_,0.);
            for(int i=0;i<local_dofs_;++i)ue[i]=u[dofs_[e*local_dofs_+i]];
            if(nonlinear_)for(int q=0;q<quadrature_;++q)
                native_fem::neo_hookean_qp(&gradients_[((e*quadrature_+q)*nodes_)*3],
                    nodes_,weights_[e*quadrature_+q],ue,p1_,p2_,values,re,ke);
            else{std::copy(linear_.begin()+e*local_dofs_*local_dofs_,
                    linear_.begin()+(e+1)*local_dofs_*local_dofs_,ke);
                for(int i=0;i<local_dofs_;++i)for(int j=0;j<local_dofs_;++j)re[i]+=ke[i*local_dofs_+j]*ue[j];}
            for(int i=0;i<local_dofs_;++i){r[dofs_[e*local_dofs_+i]]+=re[i];
                if(values)for(int j=0;j<local_dofs_;++j)values[scatter_[(e*local_dofs_+i)*local_dofs_+j]]+=ke[i*local_dofs_+j];}
        }if(load)for(std::size_t i=0;i<ndofs_;++i)r[i]-=load[i];}
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
    void build_pattern(){
        std::vector<std::vector<std::int64_t>>rows(ndofs_);
        for(std::size_t e=0;e<elements_;++e)for(int i=0;i<local_dofs_;++i)for(int j=0;j<local_dofs_;++j)
            rows[dofs_[e*local_dofs_+i]].push_back(dofs_[e*local_dofs_+j]);
        indptr_.resize(ndofs_+1);for(std::size_t i=0;i<ndofs_;++i){auto&x=rows[i];std::sort(x.begin(),x.end());
            x.erase(std::unique(x.begin(),x.end()),x.end());indices_.insert(indices_.end(),x.begin(),x.end());indptr_[i+1]=indices_.size();}
        values_.resize(indices_.size());residual_.resize(ndofs_);scatter_.resize(elements_*local_dofs_*local_dofs_);
        for(std::size_t e=0;e<elements_;++e)for(int i=0;i<local_dofs_;++i){auto row=dofs_[e*local_dofs_+i];
            for(int j=0;j<local_dofs_;++j){auto b=indices_.begin()+indptr_[row],z=indices_.begin()+indptr_[row+1];
                scatter_[(e*local_dofs_+i)*local_dofs_+j]=std::lower_bound(b,z,dofs_[e*local_dofs_+j])-indices_.begin();}}}
    void build_linear(){linear_.resize(elements_*local_dofs_*local_dofs_);
        for(std::size_t e=0;e<elements_;++e)for(int q=0;q<quadrature_;++q)
            native_fem::linear_elastic_qp(&gradients_[((e*quadrature_+q)*nodes_)*3],nodes_,
                weights_[e*quadrature_+q],p1_,p2_,&linear_[e*local_dofs_*local_dofs_]);}
    double p1_,p2_;bool nonlinear_;int nodes_,quadrature_,local_dofs_;std::size_t elements_{},ndofs_{};
    std::vector<double>gradients_,weights_,linear_,values_,residual_;
    std::vector<std::int64_t>dofs_,indptr_,indices_,scatter_;
};

void native_fem::bind_tabulated_assembler(py::module_&m){py::class_<TabulatedAssembler>(m,"TabulatedAssembler")
    .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        std::string,double,double>())
    .def("evaluate",&TabulatedAssembler::evaluate,py::arg("u"),py::arg("external_load")=py::none(),py::arg("with_tangent")=true)
    .def("evaluate_into",&TabulatedAssembler::evaluate_into,py::arg("u"),py::arg("residual"),py::arg("tangent_values")=py::none(),py::arg("external_load")=py::none())
    .def_property_readonly("indptr",&TabulatedAssembler::indptr).def_property_readonly("indices",&TabulatedAssembler::indices)
    .def_property_readonly("values",&TabulatedAssembler::values).def_property_readonly("ndofs",&TabulatedAssembler::ndofs)
    .def_property_readonly("nelements",&TabulatedAssembler::nelements);
}

