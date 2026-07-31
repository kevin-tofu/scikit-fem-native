#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "native_fem/continuum_kernel.hpp"
#include "native_fem/parallel.hpp"

namespace native_fem {

namespace py=pybind11;

template<class MaterialKernel>
class MaterialGlobalAssembler {
public:
    MaterialGlobalAssembler(
        py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>dofs,
        py::array_t<double,py::array::c_style|py::array::forcecast>gradients,
        py::array_t<double,py::array::c_style|py::array::forcecast>weights,
        MaterialKernel kernel)
        :kernel_(std::move(kernel)){
        auto d=dofs.request(),g=gradients.request(),w=weights.request();
        if(d.ndim!=3||d.shape[2]!=3)
            throw std::invalid_argument(
                "dofs must have shape (elements,nodes,3)"
            );
        elements_=d.shape[0];nodes_=d.shape[1];local_dofs_=3*nodes_;
        if(local_dofs_>native_fem::max_dofs)
            throw std::invalid_argument(
                "element exceeds native maximum node count"
            );
        if(g.ndim!=4||g.shape[0]!=elements_||g.shape[2]!=nodes_||
           g.shape[3]!=3)
            throw std::invalid_argument("invalid material gradient shape");
        quadrature_=g.shape[1];
        if(w.ndim!=2||w.shape[0]!=elements_||w.shape[1]!=quadrature_)
            throw std::invalid_argument("invalid material weight shape");
        dofs_.assign(static_cast<std::int64_t*>(d.ptr),
                     static_cast<std::int64_t*>(d.ptr)+d.size);
        gradients_.assign(static_cast<double*>(g.ptr),
                          static_cast<double*>(g.ptr)+g.size);
        weights_.assign(static_cast<double*>(w.ptr),
                        static_cast<double*>(w.ptr)+w.size);
        for(const auto dof:dofs_){
            if(dof<0)throw std::invalid_argument("negative dof index");
            ndofs_=std::max(ndofs_,static_cast<std::size_t>(dof+1));
        }
        build_pattern();build_coloring();
    }

    py::tuple evaluate(
        py::array_t<double,py::array::c_style>u,
        py::array_t<double,py::array::c_style>state,
        bool with_tangent,int requested_threads,double time_step){
        if(time_step<0.)
            throw std::invalid_argument("time_step must be nonnegative");
        auto ub=u.request(),sb=state.request();
        if(ub.ndim!=1||static_cast<std::size_t>(ub.shape[0])!=ndofs_)
            throw std::invalid_argument("u has an invalid shape");
        const auto points=elements_*quadrature_;
        if(sb.ndim!=2||static_cast<std::size_t>(sb.shape[0])!=points||
           sb.shape[1]!=MaterialKernel::state_size)
            throw std::invalid_argument("material state has an invalid shape");
        const auto point_count=static_cast<py::ssize_t>(points);
        py::array_t<double>trial_state(
            {point_count,py::ssize_t(MaterialKernel::state_size)}
        );
        auto*trial=static_cast<double*>(trial_state.request().ptr);
        const auto start=std::chrono::steady_clock::now();
        std::fill(residual_.begin(),residual_.end(),0.);
        if(with_tangent)std::fill(values_.begin(),values_.end(),0.);
        const auto*up=static_cast<const double*>(ub.ptr);
        const auto*committed=static_cast<const double*>(sb.ptr);
        {py::gil_scoped_release release;
        if(native_fem::effective_threads(elements_,requested_threads)<=1){
            for(std::size_t e=0;e<elements_;++e)
                assemble_element(
                    e,up,committed,trial,with_tangent,time_step
                );
        }else{
            for(const auto&color:colors_)
                native_fem::parallel_for_workers(
                    color.size(),requested_threads,
                    [&](std::size_t,std::size_t begin,std::size_t end){
                    for(std::size_t index=begin;index<end;++index)
                        assemble_element(
                            color[index],up,committed,trial,with_tangent,
                            time_step
                        );
                });
        }}
        const double seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
        py::object tangent=py::none();
        if(with_tangent)tangent=view(values_);
        return py::make_tuple(
            view(residual_),tangent,trial_state,seconds
        );
    }

    py::array indptr(){return view(indptr_);}
    py::array indices(){return view(indices_);}
    py::array values(){return view(values_);}
    std::size_t ndofs()const{return ndofs_;}
    std::size_t state_count()const{return elements_*quadrature_;}
    int state_size()const{return MaterialKernel::state_size;}
    std::size_t nelements()const{return elements_;}

private:
    template<class T>py::array view(std::vector<T>&values){
        return py::array_t<T>(
            {py::ssize_t(values.size())},{py::ssize_t(sizeof(T))},
            values.data(),py::cast(this));
    }

    void strain_basis(std::size_t e,int q,int local,double*out)const{
        std::fill(out,out+6,0.);
        const int node=local/3,component=local%3;
        const auto*g=&gradients_[((e*quadrature_+q)*nodes_+node)*3];
        if(component==0){out[0]=g[0];out[3]=.5*g[1];out[5]=.5*g[2];}
        else if(component==1){out[1]=g[1];out[3]=.5*g[0];out[4]=.5*g[2];}
        else{out[2]=g[2];out[4]=.5*g[1];out[5]=.5*g[0];}
    }

    void assemble_element(
        std::size_t e,const double*u,const double*state,
        double*trial_state,bool with_tangent,double time_step){
        double local_u[native_fem::max_dofs]{};
        double local_r[native_fem::max_dofs]{};
        double local_k[native_fem::max_dofs*native_fem::max_dofs];
        if(with_tangent)
            std::fill(local_k,local_k+local_dofs_*local_dofs_,0.);
        for(int local=0;local<local_dofs_;++local)
            local_u[local]=u[dofs_[e*local_dofs_+local]];
        constexpr double metric[6]={1.,1.,1.,2.,2.,2.};
        for(int q=0;q<quadrature_;++q){
            std::array<double,6>strain{};
            for(int local=0;local<local_dofs_;++local){
                double b[6];strain_basis(e,q,local,b);
                for(int i=0;i<6;++i)strain[i]+=b[i]*local_u[local];
            }
            const auto point=e*quadrature_+q;
            double tangent[36];
            const double*committed_point=nullptr;
            double*trial_point=nullptr;
            if constexpr(MaterialKernel::state_size>0){
                committed_point=state+MaterialKernel::state_size*point;
                trial_point=trial_state+MaterialKernel::state_size*point;
            }
            const auto update=kernel_.update(
                strain.data(),committed_point,trial_point,
                with_tangent?tangent:nullptr,time_step
            );
            const double weight=weights_[point];
            for(int row=0;row<local_dofs_;++row){
                double br[6];strain_basis(e,q,row,br);
                for(int i=0;i<6;++i)
                    local_r[row]+=weight*metric[i]*br[i]*update.stress[i];
                if(with_tangent)for(int column=0;column<local_dofs_;++column){
                    double bc[6];strain_basis(e,q,column,bc);
                    double entry=0.;
                    for(int i=0;i<6;++i)for(int j=0;j<6;++j)
                        entry+=weight*metric[i]*br[i]*
                            tangent[i*6+j]*bc[j];
                    local_k[row*local_dofs_+column]+=entry;
                }
            }
        }
        for(int row=0;row<local_dofs_;++row){
            residual_[dofs_[e*local_dofs_+row]]+=local_r[row];
            if(with_tangent)for(int column=0;column<local_dofs_;++column)
                values_[scatter_[(e*local_dofs_+row)*local_dofs_+column]]+=
                    local_k[row*local_dofs_+column];
        }
    }

    void build_pattern(){
        std::vector<std::vector<std::int64_t>>rows(ndofs_);
        for(std::size_t e=0;e<elements_;++e)
            for(int i=0;i<local_dofs_;++i)for(int j=0;j<local_dofs_;++j)
                rows[dofs_[e*local_dofs_+i]].push_back(
                    dofs_[e*local_dofs_+j]);
        indptr_.resize(ndofs_+1);
        for(std::size_t row=0;row<ndofs_;++row){
            auto&columns=rows[row];
            std::sort(columns.begin(),columns.end());
            columns.erase(
                std::unique(columns.begin(),columns.end()),columns.end()
            );
            indices_.insert(indices_.end(),columns.begin(),columns.end());
            indptr_[row+1]=indices_.size();
        }
        values_.resize(indices_.size());residual_.resize(ndofs_);
        scatter_.resize(elements_*local_dofs_*local_dofs_);
        for(std::size_t e=0;e<elements_;++e)for(int i=0;i<local_dofs_;++i){
            const auto row=dofs_[e*local_dofs_+i];
            for(int j=0;j<local_dofs_;++j){
                const auto begin=indices_.begin()+indptr_[row];
                const auto end=indices_.begin()+indptr_[row+1];
                scatter_[(e*local_dofs_+i)*local_dofs_+j]=
                    std::lower_bound(begin,end,dofs_[e*local_dofs_+j])-
                    indices_.begin();
            }
        }
    }

    void build_coloring(){
        std::vector<std::vector<int>>dof_colors(ndofs_);
        std::vector<int>marks;int generation=0;
        for(std::size_t e=0;e<elements_;++e){
            ++generation;
            for(int i=0;i<local_dofs_;++i)
                for(const int color:dof_colors[dofs_[e*local_dofs_+i]]){
                    if(color>=static_cast<int>(marks.size()))
                        marks.resize(color+1);
                    marks[color]=generation;
                }
            int color=0;
            while(color<static_cast<int>(marks.size())&&
                  marks[color]==generation)++color;
            if(color==static_cast<int>(colors_.size()))colors_.emplace_back();
            colors_[color].push_back(e);
            for(int i=0;i<local_dofs_;++i)
                dof_colors[dofs_[e*local_dofs_+i]].push_back(color);
        }
    }

    MaterialKernel kernel_;
    int nodes_{},local_dofs_{},quadrature_{};
    std::size_t elements_{},ndofs_{};
    std::vector<std::int64_t>dofs_,indptr_,indices_,scatter_;
    std::vector<double>gradients_,weights_,values_,residual_;
    std::vector<std::vector<int>>colors_;
};

}  // namespace native_fem
