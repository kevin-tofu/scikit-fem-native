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

class CrossBilinearAssembler {
public:
    CrossBilinearAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>row_dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>column_dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast>row_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>column_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights){
        auto rd=row_dofs.request(),cd=column_dofs.request(),rs=row_shape.request(),cs=column_shape.request(),w=weights.request();
        if(rd.ndim!=3||cd.ndim!=3||rd.shape[0]!=cd.shape[0])
            throw std::invalid_argument("cross dof maps have incompatible shapes");
        entities_=rd.shape[0];row_nodes_=rd.shape[1];column_nodes_=cd.shape[1];
        row_components_=rd.shape[2];column_components_=cd.shape[2];
        row_local_=row_nodes_*row_components_;column_local_=column_nodes_*column_components_;
        if(rs.ndim!=3||rs.shape[0]!=entities_||rs.shape[2]!=row_nodes_)throw std::invalid_argument("invalid row shape values");
        quadrature_=rs.shape[1];
        if(cs.ndim!=3||cs.shape[0]!=entities_||cs.shape[1]!=quadrature_||cs.shape[2]!=column_nodes_)
            throw std::invalid_argument("invalid column shape values");
        if(w.ndim!=2||w.shape[0]!=entities_||w.shape[1]!=quadrature_)throw std::invalid_argument("invalid cross weights");
        row_dofs_.assign((std::int64_t*)rd.ptr,(std::int64_t*)rd.ptr+rd.size);
        column_dofs_.assign((std::int64_t*)cd.ptr,(std::int64_t*)cd.ptr+cd.size);
        row_shape_.assign((double*)rs.ptr,(double*)rs.ptr+rs.size);
        column_shape_.assign((double*)cs.ptr,(double*)cs.ptr+cs.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        for(auto d:row_dofs_)rows_=std::max(rows_,std::size_t(d+1));
        for(auto d:column_dofs_)columns_=std::max(columns_,std::size_t(d+1));
        build_pattern();
    }
    py::tuple assemble(py::object coefficient_object){
        const double*coefficient=nullptr;bool tensor_coefficient=false;
        py::array_t<double,py::array::c_style>holder;
        if(!coefficient_object.is_none()){holder=py::cast<py::array_t<double,py::array::c_style>>(coefficient_object);
            auto b=holder.request();
            if(b.ndim==4&&b.shape[0]==entities_&&b.shape[1]==quadrature_&&
               b.shape[2]==row_components_&&b.shape[3]==column_components_)
                tensor_coefficient=true;
            else if(b.ndim!=2||b.shape[0]!=entities_||b.shape[1]!=quadrature_)
                throw std::invalid_argument("cross coefficient has an invalid shape");
            coefficient=(double*)b.ptr;}
        if(!tensor_coefficient&&row_components_!=column_components_)
            throw std::invalid_argument("scalar cross coefficient requires equal component counts");
        auto start=std::chrono::steady_clock::now();std::fill(values_.begin(),values_.end(),0.);
        {py::gil_scoped_release release;
        for(int e=0;e<entities_;++e)for(int q=0;q<quadrature_;++q){
            int eq=e*quadrature_+q;double scale=weights_[eq];
            for(int a=0;a<row_nodes_;++a)for(int b=0;b<column_nodes_;++b){
                double basis_entry=scale*row_shape_[eq*row_nodes_+a]*column_shape_[eq*column_nodes_+b];
                for(int r=0;r<row_components_;++r)for(int c=0;c<column_components_;++c){
                    double material;
                    if(tensor_coefficient)
                        material=coefficient[((eq*row_components_+r)*column_components_)+c];
                    else
                        material=(r==c?(coefficient?coefficient[eq]:1.):0.);
                    int i=a*row_components_+r,j=b*column_components_+c;
                    values_[scatter_[(e*row_local_+i)*column_local_+j]]+=basis_entry*material;
                }
            }}
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return py::make_tuple(view(values_),seconds);
    }
    py::array indptr(){return view(indptr_);}py::array indices(){return view(indices_);}
    py::array values(){return view(values_);}std::size_t rows()const{return rows_;}
    std::size_t columns()const{return columns_;}
private:
    template<class T>py::array view(std::vector<T>&v){return py::array_t<T>(
        {py::ssize_t(v.size())},{py::ssize_t(sizeof(T))},v.data(),py::cast(this));}
    void build_pattern(){
        std::vector<std::vector<std::int64_t>>pattern(rows_);
        for(int e=0;e<entities_;++e)for(int i=0;i<row_local_;++i)for(int j=0;j<column_local_;++j)
            pattern[row_dofs_[e*row_local_+i]].push_back(column_dofs_[e*column_local_+j]);
        indptr_.resize(rows_+1);for(std::size_t r=0;r<rows_;++r){auto&x=pattern[r];std::sort(x.begin(),x.end());
            x.erase(std::unique(x.begin(),x.end()),x.end());indices_.insert(indices_.end(),x.begin(),x.end());indptr_[r+1]=indices_.size();}
        values_.resize(indices_.size());scatter_.resize(entities_*row_local_*column_local_);
        for(int e=0;e<entities_;++e)for(int i=0;i<row_local_;++i){auto row=row_dofs_[e*row_local_+i];
            for(int j=0;j<column_local_;++j){auto begin=indices_.begin()+indptr_[row],end=indices_.begin()+indptr_[row+1];
                scatter_[(e*row_local_+i)*column_local_+j]=std::lower_bound(begin,end,column_dofs_[e*column_local_+j])-indices_.begin();}}}
    int entities_,row_nodes_,column_nodes_,row_components_,column_components_;
    int row_local_,column_local_,quadrature_;
    std::size_t rows_{},columns_{};std::vector<std::int64_t>row_dofs_,column_dofs_,indptr_,indices_,scatter_;
    std::vector<double>row_shape_,column_shape_,weights_,values_;
};

void native_fem::bind_cross_bilinear_assembler(py::module_&m){py::class_<CrossBilinearAssembler>(m,"CrossBilinearAssembler")
    .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>>())
    .def("assemble",&CrossBilinearAssembler::assemble,py::arg("coefficient")=py::none())
    .def_property_readonly("indptr",&CrossBilinearAssembler::indptr)
    .def_property_readonly("indices",&CrossBilinearAssembler::indices)
    .def_property_readonly("values",&CrossBilinearAssembler::values)
    .def_property_readonly("rows",&CrossBilinearAssembler::rows)
    .def_property_readonly("columns",&CrossBilinearAssembler::columns);}

