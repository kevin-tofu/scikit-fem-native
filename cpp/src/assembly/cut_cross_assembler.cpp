#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_fem/cross_contraction.hpp"
#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py=pybind11;

namespace {
class CutCrossAssembler {
public:
    CutCrossAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>row_dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>column_dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>offsets,
        py::array_t<double,py::array::c_style|py::array::forcecast>row_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>column_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights,
        py::array_t<double,py::array::c_style|py::array::forcecast>row_gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>column_gradients){
        auto rd=row_dofs.request(),cd=column_dofs.request(),o=offsets.request();
        auto rs=row_shape.request(),cs=column_shape.request(),w=weights.request();
        auto rg=row_gradients.request(),cg=column_gradients.request();
        if(rd.ndim!=3||cd.ndim!=3||rd.shape[0]!=cd.shape[0])
            throw std::invalid_argument("cut cross cell dofs have incompatible shapes");
        cells_=int(rd.shape[0]);row_nodes_=int(rd.shape[1]);row_components_=int(rd.shape[2]);
        column_nodes_=int(cd.shape[1]);column_components_=int(cd.shape[2]);
        row_local_=row_nodes_*row_components_;column_local_=column_nodes_*column_components_;
        if(o.ndim!=1||o.shape[0]!=cells_+1)
            throw std::invalid_argument("cut cross offsets have an invalid shape");
        if(rs.ndim!=2||rs.shape[1]!=row_nodes_||cs.ndim!=2||
           cs.shape[0]!=rs.shape[0]||cs.shape[1]!=column_nodes_)
            throw std::invalid_argument("cut cross shape values have an invalid shape");
        points_=int(rs.shape[0]);
        if(w.ndim!=1||w.shape[0]!=points_)
            throw std::invalid_argument("cut cross weights have an invalid shape");
        if(rg.ndim!=3||rg.shape[0]!=points_||rg.shape[1]!=row_nodes_||
           cg.ndim!=3||cg.shape[0]!=points_||cg.shape[1]!=column_nodes_)
            throw std::invalid_argument("cut cross gradients have an invalid shape");
        row_dimension_=int(rg.shape[2]);column_dimension_=int(cg.shape[2]);
        row_dofs_.assign((std::int64_t*)rd.ptr,(std::int64_t*)rd.ptr+rd.size);
        column_dofs_.assign((std::int64_t*)cd.ptr,(std::int64_t*)cd.ptr+cd.size);
        offsets_.assign((std::int64_t*)o.ptr,(std::int64_t*)o.ptr+o.size);
        row_shape_.assign((double*)rs.ptr,(double*)rs.ptr+rs.size);
        column_shape_.assign((double*)cs.ptr,(double*)cs.ptr+cs.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        row_gradients_.assign((double*)rg.ptr,(double*)rg.ptr+rg.size);
        column_gradients_.assign((double*)cg.ptr,(double*)cg.ptr+cg.size);
        if(offsets_.front()!=0||offsets_.back()!=points_)
            throw std::invalid_argument("cut cross offsets do not span points");
        for(auto dof:row_dofs_)rows_=std::max(rows_,std::size_t(dof+1));
        for(auto dof:column_dofs_)columns_=std::max(columns_,std::size_t(dof+1));
        build_pattern();build_coloring();
    }
    void assemble(py::object coefficient_object,const std::string&row_kind,
                  const std::string&column_kind,int requested_threads){
        bool row_gradient=gradient_kind(row_kind),column_gradient=gradient_kind(column_kind);
        py::array_t<double,py::array::c_style>holder;
        const double*coefficient=nullptr;bool tensor=false;
        if(!coefficient_object.is_none()){
            holder=py::cast<py::array_t<double,py::array::c_style>>(coefficient_object);
            auto info=holder.request();tensor=validate_coefficient(info,row_gradient,column_gradient);
            coefficient=(double*)info.ptr;
        }
        validate_contraction(row_gradient,column_gradient,tensor);
        native_fem::CrossBasisView row{
            row_shape_.data(),row_gradients_.data(),row_nodes_,row_dimension_,
            row_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value};
        native_fem::CrossBasisView column{
            column_shape_.data(),column_gradients_.data(),column_nodes_,column_dimension_,
            column_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value};
        native_fem::CrossCoefficientView coefficient_view{
            coefficient,tensor,row_components_,row_dimension_,column_components_,column_dimension_,
            row.kind,column.kind};
        std::fill(values_.begin(),values_.end(),0.);
        {py::gil_scoped_release release;
        for(const auto&color:colors_)native_fem::parallel_for_workers(
            color.size(),requested_threads,[&](std::size_t,std::size_t begin,std::size_t end){
                for(auto index=begin;index<end;++index)
                    assemble_cell(color[index],row,column,coefficient_view,coefficient,tensor);
            });}
    }
    py::array indptr(){return array_view(indptr_);}py::array indices(){return array_view(indices_);}
    py::array values(){return array_view(values_);}std::size_t rows()const{return rows_;}
    std::size_t columns()const{return columns_;}std::size_t color_count()const{return colors_.size();}
private:
    static bool gradient_kind(const std::string&kind){
        if(kind=="value")return false;if(kind=="gradient")return true;
        throw std::invalid_argument("cut cross kind must be value or gradient");
    }
    bool validate_coefficient(const py::buffer_info&info,bool rg,bool cg)const{
        int axis=0;bool tensor=info.ndim==3+int(rg)+int(cg);
        if(tensor){tensor=info.shape[axis++]==points_&&info.shape[axis++]==row_components_;
            if(rg)tensor=tensor&&info.shape[axis++]==row_dimension_;
            tensor=tensor&&info.shape[axis++]==column_components_;
            if(cg)tensor=tensor&&info.shape[axis]==column_dimension_;}
        if(!tensor&&(info.ndim!=1||info.shape[0]!=points_))
            throw std::invalid_argument("cut cross coefficient has an invalid shape");
        return tensor;
    }
    void validate_contraction(bool rg,bool cg,bool tensor)const{
        if(tensor)return;
        if(row_components_!=column_components_||rg!=cg||
           (rg&&row_dimension_!=column_dimension_))
            throw std::invalid_argument("scalar cut cross coefficient has incompatible bases");
    }
    void assemble_cell(int cell,const native_fem::CrossBasisView&row,
        const native_fem::CrossBasisView&column,
        const native_fem::CrossCoefficientView&coefficient_view,
        const double*coefficient,bool tensor){
        for(int q=offsets_[cell];q<offsets_[cell+1];++q)
            for(int a=0;a<row_nodes_;++a)for(int b=0;b<column_nodes_;++b){
                if(!tensor){double entry=weights_[q]*(coefficient?coefficient[q]:1.)*
                    native_fem::contract_scalar_cross_basis(row,column,q,a,b);
                    for(int component=0;component<row_components_;++component){
                        int i=a*row_components_+component,j=b*column_components_+component;
                        values_[scatter_[(cell*row_local_+i)*column_local_+j]]+=entry;}}
                else for(int r=0;r<row_components_;++r)for(int c=0;c<column_components_;++c){
                    double entry=native_fem::contract_cross_basis(
                        row,column,coefficient_view,q,a,b,r,c);
                    int i=a*row_components_+r,j=b*column_components_+c;
                    values_[scatter_[(cell*row_local_+i)*column_local_+j]]+=weights_[q]*entry;}
            }
    }
    void build_pattern(){std::vector<std::vector<std::int64_t>>pattern(rows_);
        for(int cell=0;cell<cells_;++cell)if(offsets_[cell]<offsets_[cell+1])
            for(int i=0;i<row_local_;++i)for(int j=0;j<column_local_;++j)
                pattern[row_dofs_[cell*row_local_+i]].push_back(column_dofs_[cell*column_local_+j]);
        indptr_.resize(rows_+1);for(std::size_t row=0;row<rows_;++row){auto&x=pattern[row];
            std::sort(x.begin(),x.end());x.erase(std::unique(x.begin(),x.end()),x.end());
            indices_.insert(indices_.end(),x.begin(),x.end());indptr_[row+1]=indices_.size();}
        values_.resize(indices_.size());scatter_.resize(cells_*row_local_*column_local_);
        for(int cell=0;cell<cells_;++cell)for(int i=0;i<row_local_;++i){auto row=row_dofs_[cell*row_local_+i];
            for(int j=0;j<column_local_;++j){auto begin=indices_.begin()+indptr_[row],end=indices_.begin()+indptr_[row+1];
                scatter_[(cell*row_local_+i)*column_local_+j]=std::lower_bound(
                    begin,end,column_dofs_[cell*column_local_+j])-indices_.begin();}}}
    void build_coloring(){std::vector<std::vector<int>>dof_colors(rows_);std::vector<int>marks;int generation=0;
        for(int cell=0;cell<cells_;++cell)if(offsets_[cell]<offsets_[cell+1]){++generation;
            for(int i=0;i<row_local_;++i)for(int color:dof_colors[row_dofs_[cell*row_local_+i]]){
                if(color>=int(marks.size()))marks.resize(color+1);marks[color]=generation;}
            int color=0;while(color<int(marks.size())&&marks[color]==generation)++color;
            if(color==int(colors_.size()))colors_.emplace_back();colors_[color].push_back(cell);
            for(int i=0;i<row_local_;++i)dof_colors[row_dofs_[cell*row_local_+i]].push_back(color);}}
    template<class T>py::array array_view(std::vector<T>&values){return py::array_t<T>(
        {py::ssize_t(values.size())},{py::ssize_t(sizeof(T))},values.data(),py::cast(this));}
    int cells_{},points_{},row_nodes_{},column_nodes_{},row_components_{},column_components_;
    int row_local_{},column_local_{},row_dimension_{},column_dimension_{};
    std::size_t rows_{},columns_{};std::vector<std::int64_t>row_dofs_,column_dofs_,offsets_,indptr_,indices_,scatter_;
    std::vector<double>row_shape_,column_shape_,row_gradients_,column_gradients_,weights_,values_;
    std::vector<std::vector<int>>colors_;
};
}

void native_fem::bind_cut_cross_assembler(py::module_&module){
    py::class_<CutCrossAssembler>(module,"CutCrossAssembler")
        .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>>())
        .def("assemble",&CutCrossAssembler::assemble,py::arg("coefficient")=py::none(),
            py::arg("row_kind")="value",py::arg("column_kind")="value",py::arg("num_threads")=0)
        .def_property_readonly("indptr",&CutCrossAssembler::indptr)
        .def_property_readonly("indices",&CutCrossAssembler::indices)
        .def_property_readonly("values",&CutCrossAssembler::values)
        .def_property_readonly("rows",&CutCrossAssembler::rows)
        .def_property_readonly("columns",&CutCrossAssembler::columns)
        .def_property_readonly("color_count",&CutCrossAssembler::color_count);
}
