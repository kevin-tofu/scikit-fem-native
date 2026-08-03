#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py=pybind11;

namespace {

template<class T>py::array view(std::vector<T>&values,py::handle owner){
    return py::array_t<T>(
        {py::ssize_t(values.size())},{py::ssize_t(sizeof(T))},
        values.data(),owner);
}

class CutFormData {
public:
    CutFormData(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>offsets,
        py::array_t<double,py::array::c_style|py::array::forcecast>shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights){
        auto d=dofs.request(),o=offsets.request(),s=shape.request();
        auto g=gradients.request(),w=weights.request();
        if(d.ndim!=3)throw std::invalid_argument(
            "cell dofs must have shape (cells, nodes, components)");
        cells_=static_cast<int>(d.shape[0]);nodes_=static_cast<int>(d.shape[1]);
        components_=static_cast<int>(d.shape[2]);local_dofs_=nodes_*components_;
        if(o.ndim!=1||o.shape[0]!=cells_+1)
            throw std::invalid_argument("cell offsets must have shape (cells + 1,)");
        if(s.ndim!=2||s.shape[1]!=nodes_)
            throw std::invalid_argument("cut shape values have an invalid shape");
        points_=static_cast<int>(s.shape[0]);
        if(g.ndim!=3||g.shape[0]!=points_||g.shape[1]!=nodes_)
            throw std::invalid_argument("cut shape gradients have an invalid shape");
        dimension_=static_cast<int>(g.shape[2]);
        if(w.ndim!=1||w.shape[0]!=points_)
            throw std::invalid_argument("cut weights have an invalid shape");
        dofs_.assign((std::int64_t*)d.ptr,(std::int64_t*)d.ptr+d.size);
        offsets_.assign((std::int64_t*)o.ptr,(std::int64_t*)o.ptr+o.size);
        shape_.assign((double*)s.ptr,(double*)s.ptr+s.size);
        gradients_.assign((double*)g.ptr,(double*)g.ptr+g.size);
        weights_.assign((double*)w.ptr,(double*)w.ptr+w.size);
        if(offsets_.front()!=0||offsets_.back()!=points_)
            throw std::invalid_argument("cell offsets do not span cut points");
        for(int cell=0;cell<cells_;++cell)
            if(offsets_[cell]>offsets_[cell+1])
                throw std::invalid_argument("cell offsets must be nondecreasing");
        for(auto dof:dofs_){
            if(dof<0)throw std::invalid_argument("negative dof index");
            ndofs_=std::max(ndofs_,std::size_t(dof+1));
        }
    }
protected:
    int cells_{},nodes_{},components_{},local_dofs_{},points_{},dimension_{};
    std::size_t ndofs_{};
    std::vector<std::int64_t>dofs_,offsets_;
    std::vector<double>shape_,gradients_,weights_;
};

class CutLinearFormAssembler:public CutFormData {
public:
    using CutFormData::CutFormData;
    py::tuple assemble(py::object value_object,py::object gradient_object,
                       int requested_threads){
        py::array_t<double,py::array::c_style|py::array::forcecast>value_array;
        py::array_t<double,py::array::c_style|py::array::forcecast>gradient_array;
        const double*value=nullptr;const double*gradient=nullptr;
        bool constant_value=false,constant_gradient=false;
        if(!value_object.is_none()){
            value_array=py::cast<decltype(value_array)>(value_object);
            auto info=value_array.request();
            constant_value=info.ndim==1&&info.shape[0]==components_;
            if(!constant_value&&(info.ndim!=2||info.shape[0]!=points_||
               info.shape[1]!=components_))
                throw std::invalid_argument("cut value coefficient has an invalid shape");
            value=(double*)info.ptr;
        }
        if(!gradient_object.is_none()){
            gradient_array=py::cast<decltype(gradient_array)>(gradient_object);
            auto info=gradient_array.request();
            constant_gradient=info.ndim==2&&info.shape[0]==components_&&
                info.shape[1]==dimension_;
            if(!constant_gradient&&(info.ndim!=3||info.shape[0]!=points_||
               info.shape[1]!=components_||info.shape[2]!=dimension_))
                throw std::invalid_argument("cut gradient coefficient has an invalid shape");
            gradient=(double*)info.ptr;
        }
        if(!value&&!gradient)throw std::invalid_argument("at least one coefficient is required");
        auto start=std::chrono::steady_clock::now();result_.assign(ndofs_,0.);
        {py::gil_scoped_release release;
        auto workers=native_fem::effective_threads(cells_,requested_threads);
        if(workers<=1)assemble_range(0,cells_,result_,value,gradient,
            constant_value,constant_gradient);
        else{
            locals_.resize(workers);
            for(auto&local:locals_)local.assign(ndofs_,0.);
            native_fem::parallel_for_workers(cells_,requested_threads,
                [&](std::size_t worker,std::size_t begin,std::size_t end){
                    assemble_range(int(begin),int(end),locals_[worker],value,
                        gradient,constant_value,constant_gradient);
                });
            native_fem::parallel_for_workers(ndofs_,requested_threads,
                [&](std::size_t,std::size_t begin,std::size_t end){
                    for(auto dof=begin;dof<end;++dof)for(const auto&local:locals_)
                        result_[dof]+=local[dof];
                });
        }}
        auto seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
        return py::make_tuple(view(result_,py::cast(this)),seconds);
    }
    std::size_t ndofs()const{return ndofs_;}
    int cell_count()const{return cells_;}int point_count()const{return points_;}
private:
    void assemble_range(int begin,int end,std::vector<double>&output,
        const double*value,const double*gradient,bool constant_value,
        bool constant_gradient){
        for(int cell=begin;cell<end;++cell)for(int q=offsets_[cell];q<offsets_[cell+1];++q)
            for(int node=0;node<nodes_;++node)for(int component=0;component<components_;++component){
                double contribution=0.;
                if(value)contribution+=shape_[q*nodes_+node]*
                    value[constant_value?component:q*components_+component];
                if(gradient)for(int d=0;d<dimension_;++d)
                    contribution+=gradients_[(q*nodes_+node)*dimension_+d]*
                        gradient[constant_gradient?component*dimension_+d:
                            (q*components_+component)*dimension_+d];
                output[dofs_[(cell*nodes_+node)*components_+component]]+=
                    weights_[q]*contribution;
            }
    }
    std::vector<double>result_;std::vector<std::vector<double>>locals_;
};

class CutBilinearFormAssembler:public CutFormData {
public:
    CutBilinearFormAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>dofs,
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>offsets,
        py::array_t<double,py::array::c_style|py::array::forcecast>shape,
        py::array_t<double,py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights):
        CutFormData(dofs,offsets,shape,gradients,weights){build_pattern();build_coloring();}
    void assemble(py::object value_object,py::object gradient_object,int requested_threads){
        py::array_t<double,py::array::c_style|py::array::forcecast>vh,gh;
        const double*value=coefficient(value_object,vh,"value");
        const double*gradient=coefficient(gradient_object,gh,"gradient");
        if(!value&&!gradient)throw std::invalid_argument("at least one coefficient is required");
        std::fill(values_.begin(),values_.end(),0.);
        {py::gil_scoped_release release;
        for(const auto&color:colors_)native_fem::parallel_for_workers(
            color.size(),requested_threads,[&](std::size_t,std::size_t begin,std::size_t end){
                for(auto index=begin;index<end;++index)
                    assemble_cell(color[index],value,gradient);
            });}
    }
    py::array indptr(){return view(indptr_,py::cast(this));}
    py::array indices(){return view(indices_,py::cast(this));}
    py::array values(){return view(values_,py::cast(this));}
    std::size_t ndofs()const{return ndofs_;}std::size_t color_count()const{return colors_.size();}
private:
    const double*coefficient(py::object object,py::array_t<double,py::array::c_style|py::array::forcecast>&holder,const char*name){
        if(object.is_none())return nullptr;
        using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
        holder=py::cast<Array>(object);
        auto info=holder.request();
        if(info.ndim!=1||info.shape[0]!=points_)
            throw std::invalid_argument(std::string("cut ")+name+" coefficient has an invalid shape");
        return (double*)info.ptr;
    }
    void assemble_cell(int cell,const double*value,const double*gradient){
        for(int q=offsets_[cell];q<offsets_[cell+1];++q)for(int a=0;a<nodes_;++a)
            for(int b=0;b<nodes_;++b){
                double entry=0.;
                if(value)entry+=value[q]*shape_[q*nodes_+a]*shape_[q*nodes_+b];
                if(gradient)for(int d=0;d<dimension_;++d)
                    entry+=gradient[q]*gradients_[(q*nodes_+a)*dimension_+d]*
                        gradients_[(q*nodes_+b)*dimension_+d];
                entry*=weights_[q];
                for(int component=0;component<components_;++component){
                    int i=a*components_+component,j=b*components_+component;
                    values_[scatter_[(cell*local_dofs_+i)*local_dofs_+j]]+=entry;
                }
            }
    }
    void build_pattern(){
        std::vector<std::vector<std::int64_t>>rows(ndofs_);
        for(int cell=0;cell<cells_;++cell)if(offsets_[cell]<offsets_[cell+1])
            for(int i=0;i<local_dofs_;++i)for(int j=0;j<local_dofs_;++j)
                rows[dofs_[cell*local_dofs_+i]].push_back(dofs_[cell*local_dofs_+j]);
        indptr_.resize(ndofs_+1);
        for(std::size_t row=0;row<ndofs_;++row){auto&columns=rows[row];
            std::sort(columns.begin(),columns.end());columns.erase(
                std::unique(columns.begin(),columns.end()),columns.end());
            indices_.insert(indices_.end(),columns.begin(),columns.end());
            indptr_[row+1]=indices_.size();}
        values_.resize(indices_.size());scatter_.resize(cells_*local_dofs_*local_dofs_);
        for(int cell=0;cell<cells_;++cell)for(int i=0;i<local_dofs_;++i){
            auto row=dofs_[cell*local_dofs_+i];
            for(int j=0;j<local_dofs_;++j){auto begin=indices_.begin()+indptr_[row];
                auto end=indices_.begin()+indptr_[row+1];
                scatter_[(cell*local_dofs_+i)*local_dofs_+j]=
                    std::lower_bound(begin,end,dofs_[cell*local_dofs_+j])-indices_.begin();}}
    }
    void build_coloring(){
        std::vector<std::vector<int>>dof_colors(ndofs_);std::vector<int>marks;int generation=0;
        for(int cell=0;cell<cells_;++cell)if(offsets_[cell]<offsets_[cell+1]){
            ++generation;for(int i=0;i<local_dofs_;++i)
                for(int color:dof_colors[dofs_[cell*local_dofs_+i]]){
                    if(color>=int(marks.size()))marks.resize(color+1);
                    marks[color]=generation;}
            int color=0;while(color<int(marks.size())&&marks[color]==generation)++color;
            if(color==int(colors_.size()))colors_.emplace_back();
            colors_[color].push_back(cell);
            for(int i=0;i<local_dofs_;++i)dof_colors[dofs_[cell*local_dofs_+i]].push_back(color);
        }
    }
    std::vector<std::int64_t>indptr_,indices_;std::vector<std::size_t>scatter_;
    std::vector<double>values_;std::vector<std::vector<int>>colors_;
};

}

void native_fem::bind_cut_form_assemblers(py::module_&module){
    py::class_<CutLinearFormAssembler>(module,"CutLinearFormAssembler")
        .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>>())
        .def("assemble",&CutLinearFormAssembler::assemble,py::arg("value")=py::none(),
            py::arg("gradient")=py::none(),py::arg("num_threads")=0)
        .def_property_readonly("ndofs",&CutLinearFormAssembler::ndofs)
        .def_property_readonly("cell_count",&CutLinearFormAssembler::cell_count)
        .def_property_readonly("point_count",&CutLinearFormAssembler::point_count);
    py::class_<CutBilinearFormAssembler>(module,"CutBilinearFormAssembler")
        .def(py::init<py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>,
            py::array_t<double,py::array::c_style|py::array::forcecast>>())
        .def("assemble",&CutBilinearFormAssembler::assemble,py::arg("value")=py::none(),
            py::arg("gradient")=py::none(),py::arg("num_threads")=0)
        .def_property_readonly("indptr",&CutBilinearFormAssembler::indptr)
        .def_property_readonly("indices",&CutBilinearFormAssembler::indices)
        .def_property_readonly("values",&CutBilinearFormAssembler::values)
        .def_property_readonly("ndofs",&CutBilinearFormAssembler::ndofs)
        .def_property_readonly("color_count",&CutBilinearFormAssembler::color_count);
}
