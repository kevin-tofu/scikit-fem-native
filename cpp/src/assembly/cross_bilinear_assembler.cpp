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
#include "native_fem/cross_contraction.hpp"
#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py=pybind11;

class CrossBilinearAssembler {
public:
    CrossBilinearAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>row_dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>column_dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast>row_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>column_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights,
        py::object row_gradients_object=py::none(),
        py::object column_gradients_object=py::none()){
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
        read_gradients(
            row_gradients_object,row_gradients_,row_dimension_,
            row_nodes_,"row"
        );
        read_gradients(
            column_gradients_object,column_gradients_,column_dimension_,
            column_nodes_,"column"
        );
        for(auto d:row_dofs_)rows_=std::max(rows_,std::size_t(d+1));
        for(auto d:column_dofs_)columns_=std::max(columns_,std::size_t(d+1));
        build_pattern();build_coloring();
    }
    py::tuple assemble(
        py::object coefficient_object,
        const std::string&row_kind="value",
        const std::string&column_kind="value",
        int requested_threads=0){
        const bool row_gradient=kind_is_gradient(row_kind);
        const bool column_gradient=kind_is_gradient(column_kind);
        const double*coefficient=nullptr;bool tensor_coefficient=false;
        py::array_t<double,py::array::c_style>holder;
        if(!coefficient_object.is_none()){holder=py::cast<py::array_t<double,py::array::c_style>>(coefficient_object);
            auto b=holder.request();
            tensor_coefficient=validate_coefficient(
                b,row_gradient,column_gradient
            );
            coefficient=(double*)b.ptr;}
        validate_kinds(row_gradient,column_gradient,tensor_coefficient);
        const auto row_basis=basis_view(
            row_shape_,row_gradients_,row_nodes_,row_dimension_,row_gradient
        );
        const auto column_basis=basis_view(
            column_shape_,column_gradients_,column_nodes_,column_dimension_,
            column_gradient
        );
        const native_fem::CrossCoefficientView coefficient_view{
            coefficient,tensor_coefficient,row_components_,row_dimension_,
            column_components_,column_dimension_,
            row_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value,
            column_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value,
        };
        auto start=std::chrono::steady_clock::now();std::fill(values_.begin(),values_.end(),0.);
        {py::gil_scoped_release release;
        if(native_fem::effective_threads(
               static_cast<std::size_t>(entities_),requested_threads)<=1){
            for(int e=0;e<entities_;++e)assemble_entity(
                e,coefficient,tensor_coefficient,row_basis,column_basis,
                coefficient_view
            );
        }else{
            for(const auto&color:colors_){
                native_fem::parallel_for_workers(
                    color.size(),requested_threads,
                    [&](std::size_t,std::size_t begin,std::size_t end){
                    for(std::size_t index=begin;index<end;++index)
                        assemble_entity(
                            color[index],coefficient,tensor_coefficient,
                            row_basis,column_basis,coefficient_view
                        );
                });
            }
        }}
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return py::make_tuple(view(values_),seconds);
    }
    py::tuple contract_only(
        py::object coefficient_object,
        const std::string&row_kind="value",
        const std::string&column_kind="value"){
        const bool row_gradient=kind_is_gradient(row_kind);
        const bool column_gradient=kind_is_gradient(column_kind);
        const double*coefficient=nullptr;bool tensor_coefficient=false;
        py::array_t<double,py::array::c_style>holder;
        if(!coefficient_object.is_none()){
            holder=py::cast<py::array_t<double,py::array::c_style>>(
                coefficient_object
            );
            auto b=holder.request();
            tensor_coefficient=validate_coefficient(
                b,row_gradient,column_gradient
            );
            coefficient=(double*)b.ptr;
        }
        validate_kinds(row_gradient,column_gradient,tensor_coefficient);
        const auto row_basis=basis_view(
            row_shape_,row_gradients_,row_nodes_,row_dimension_,row_gradient
        );
        const auto column_basis=basis_view(
            column_shape_,column_gradients_,column_nodes_,column_dimension_,
            column_gradient
        );
        const native_fem::CrossCoefficientView coefficient_view{
            coefficient,tensor_coefficient,row_components_,row_dimension_,
            column_components_,column_dimension_,
            row_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value,
            column_gradient?native_fem::CrossBasisKind::gradient:native_fem::CrossBasisKind::value,
        };
        double checksum=0.;
        auto start=std::chrono::steady_clock::now();
        {py::gil_scoped_release release;
        for(int e=0;e<entities_;++e)for(int q=0;q<quadrature_;++q){
            const int eq=e*quadrature_+q;
            for(int a=0;a<row_nodes_;++a)
                for(int b=0;b<column_nodes_;++b){
                    if(!tensor_coefficient){
                        const double material=coefficient?coefficient[eq]:1.;
                        checksum+=row_components_*weights_[eq]*material*
                            native_fem::contract_scalar_cross_basis(
                                row_basis,column_basis,eq,a,b
                            );
                    }else{
                        for(int r=0;r<row_components_;++r)
                            for(int c=0;c<column_components_;++c)
                            checksum+=weights_[eq]*
                                native_fem::contract_cross_basis(
                                    row_basis,column_basis,coefficient_view,
                                    eq,a,b,r,c
                                );
                    }
                }
        }}
        const double seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start
        ).count();
        return py::make_tuple(checksum,seconds);
    }
    void update_tabulation(
        py::array_t<double,py::array::c_style|py::array::forcecast>row_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>column_shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights,
        py::object row_gradients_object=py::none(),
        py::object column_gradients_object=py::none()){
        auto rs=row_shape.request(),cs=column_shape.request(),w=weights.request();
        if(rs.ndim!=3||rs.shape[0]!=entities_||rs.shape[1]!=quadrature_||
           rs.shape[2]!=row_nodes_)
            throw std::invalid_argument("updated row shape values have an invalid shape");
        if(cs.ndim!=3||cs.shape[0]!=entities_||cs.shape[1]!=quadrature_||
           cs.shape[2]!=column_nodes_)
            throw std::invalid_argument("updated column shape values have an invalid shape");
        if(w.ndim!=2||w.shape[0]!=entities_||w.shape[1]!=quadrature_)
            throw std::invalid_argument("updated cross weights have an invalid shape");
        std::vector<double>new_row_gradients,new_column_gradients;
        int new_row_dimension=0,new_column_dimension=0;
        read_gradients(
            row_gradients_object,new_row_gradients,new_row_dimension,
            row_nodes_,"updated row"
        );
        read_gradients(
            column_gradients_object,new_column_gradients,new_column_dimension,
            column_nodes_,"updated column"
        );
        if(new_row_dimension!=row_dimension_||
           new_column_dimension!=column_dimension_)
            throw std::invalid_argument("updated gradient dimensions differ");
        row_shape_.assign((double*)rs.ptr,(double*)rs.ptr+rs.size);
        column_shape_.assign((double*)cs.ptr,(double*)cs.ptr+cs.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        row_gradients_.swap(new_row_gradients);
        column_gradients_.swap(new_column_gradients);
    }
    py::array indptr(){return view(indptr_);}py::array indices(){return view(indices_);}
    py::array values(){return view(values_);}std::size_t rows()const{return rows_;}
    std::size_t columns()const{return columns_;}
private:
    void assemble_entity(
        int e,const double*coefficient,bool tensor_coefficient,
        const native_fem::CrossBasisView&row_basis,
        const native_fem::CrossBasisView&column_basis,
        const native_fem::CrossCoefficientView&coefficient_view){
        for(int q=0;q<quadrature_;++q){
            const int eq=e*quadrature_+q;
            const double scale=weights_[eq];
            for(int a=0;a<row_nodes_;++a)
                for(int b=0;b<column_nodes_;++b){
                    if(!tensor_coefficient){
                        const double material=coefficient?coefficient[eq]:1.;
                        const double entry=scale*material*
                            native_fem::contract_scalar_cross_basis(
                                row_basis,column_basis,eq,a,b
                            );
                        for(int r=0;r<row_components_;++r){
                            const int i=a*row_components_+r;
                            const int j=b*column_components_+r;
                            values_[scatter_[
                                (e*row_local_+i)*column_local_+j
                            ]]+=entry;
                        }
                    }else{
                        for(int r=0;r<row_components_;++r)
                            for(int c=0;c<column_components_;++c){
                                const double entry=
                                    native_fem::contract_cross_basis(
                                        row_basis,column_basis,
                                        coefficient_view,eq,a,b,r,c
                                    );
                                const int i=a*row_components_+r;
                                const int j=b*column_components_+c;
                                values_[scatter_[
                                    (e*row_local_+i)*column_local_+j
                                ]]+=scale*entry;
                            }
                    }
                }
        }
    }
    static native_fem::CrossBasisView basis_view(
        const std::vector<double>&shape,const std::vector<double>&gradient,
        int nodes,int dimension,bool use_gradient){
        return {
            shape.data(),gradient.empty()?nullptr:gradient.data(),
            nodes,dimension,
            use_gradient?native_fem::CrossBasisKind::gradient
                        :native_fem::CrossBasisKind::value,
        };
    }
    static bool kind_is_gradient(const std::string&kind){
        if(kind=="value")return false;
        if(kind=="gradient")return true;
        throw std::invalid_argument("cross basis kind must be value or gradient");
    }
    void read_gradients(
        py::object object,std::vector<double>&output,int&dimension,
        int nodes,const char*name){
        if(object.is_none())return;
        auto array=py::cast<py::array_t<double,py::array::c_style>>(object);
        auto b=array.request();
        if(b.ndim!=4||b.shape[0]!=entities_||b.shape[1]!=quadrature_||
           b.shape[2]!=nodes)
            throw std::invalid_argument(std::string(name)+" gradients have an invalid shape");
        dimension=b.shape[3];
        output.assign((double*)b.ptr,(double*)b.ptr+b.size);
    }
    bool valid_tensor(
        const py::buffer_info&b,bool row_gradient,bool column_gradient)const{
        int axis=0;
        if(b.ndim!=4+int(row_gradient)+int(column_gradient))return false;
        if(b.shape[axis++]!=entities_||b.shape[axis++]!=quadrature_)return false;
        if(b.shape[axis++]!=row_components_)return false;
        if(row_gradient&&b.shape[axis++]!=row_dimension_)return false;
        if(b.shape[axis++]!=column_components_)return false;
        return !column_gradient||b.shape[axis]==column_dimension_;
    }
    bool validate_coefficient(
        const py::buffer_info&b,bool row_gradient,bool column_gradient)const{
        const bool tensor=valid_tensor(
            b,row_gradient,column_gradient
        );
        if(!tensor&&
           (b.ndim!=2||b.shape[0]!=entities_||b.shape[1]!=quadrature_))
            throw std::invalid_argument("cross coefficient has an invalid shape");
        return tensor;
    }
    void validate_kinds(
        bool row_gradient,bool column_gradient,bool tensor_coefficient)const{
        if(row_gradient&&row_gradients_.empty())
            throw std::invalid_argument("row gradients are unavailable");
        if(column_gradient&&column_gradients_.empty())
            throw std::invalid_argument("column gradients are unavailable");
        if(tensor_coefficient)return;
        if(row_components_!=column_components_)
            throw std::invalid_argument("scalar cross coefficient requires equal component counts");
        if(row_gradient!=column_gradient)
            throw std::invalid_argument("value-gradient coupling requires a tensor coefficient");
        if(row_gradient&&row_dimension_!=column_dimension_)
            throw std::invalid_argument("scalar gradient coupling requires equal spatial dimensions");
    }
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
    void build_coloring(){
        std::vector<std::vector<int>>dof_colors(rows_);
        std::vector<int>marks;
        int generation=0;
        for(int e=0;e<entities_;++e){
            ++generation;
            for(int i=0;i<row_local_;++i)
                for(const int color:dof_colors[row_dofs_[e*row_local_+i]]){
                    if(color>=static_cast<int>(marks.size()))
                        marks.resize(color+1);
                    marks[color]=generation;
                }
            int color=0;
            while(color<static_cast<int>(marks.size())&&
                  marks[color]==generation)++color;
            if(color==static_cast<int>(colors_.size()))colors_.emplace_back();
            colors_[color].push_back(e);
            for(int i=0;i<row_local_;++i)
                dof_colors[row_dofs_[e*row_local_+i]].push_back(color);
        }
    }
    int entities_,row_nodes_,column_nodes_,row_components_,column_components_;
    int row_local_,column_local_,quadrature_;
    int row_dimension_{},column_dimension_{};
    std::size_t rows_{},columns_{};std::vector<std::int64_t>row_dofs_,column_dofs_,indptr_,indices_,scatter_;
    std::vector<double>row_shape_,column_shape_,row_gradients_,column_gradients_,weights_,values_;
    std::vector<std::vector<int>>colors_;
};

void native_fem::bind_cross_bilinear_assembler(py::module_&m){py::class_<CrossBilinearAssembler>(m,"CrossBilinearAssembler")
    .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::array_t<double,py::array::c_style|py::array::forcecast>,
        py::object,py::object>(),
        py::arg("row_dofs"),py::arg("column_dofs"),
        py::arg("row_shape"),py::arg("column_shape"),py::arg("weights"),
        py::arg("row_gradients")=py::none(),
        py::arg("column_gradients")=py::none())
    .def("assemble",&CrossBilinearAssembler::assemble,
        py::arg("coefficient")=py::none(),py::arg("row_kind")="value",
        py::arg("column_kind")="value",py::arg("num_threads")=0)
    .def("contract_only",&CrossBilinearAssembler::contract_only,
        py::arg("coefficient")=py::none(),py::arg("row_kind")="value",
        py::arg("column_kind")="value")
    .def("update_tabulation",&CrossBilinearAssembler::update_tabulation,
        py::arg("row_shape"),py::arg("column_shape"),py::arg("weights"),
        py::arg("row_gradients")=py::none(),
        py::arg("column_gradients")=py::none())
    .def_property_readonly("indptr",&CrossBilinearAssembler::indptr)
    .def_property_readonly("indices",&CrossBilinearAssembler::indices)
    .def_property_readonly("values",&CrossBilinearAssembler::values)
    .def_property_readonly("rows",&CrossBilinearAssembler::rows)
    .def_property_readonly("columns",&CrossBilinearAssembler::columns);}
