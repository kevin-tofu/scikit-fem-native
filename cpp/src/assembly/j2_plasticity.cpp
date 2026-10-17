#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py = pybind11;
namespace {

using Tensor = std::array<std::array<double, 3>, 3>;
using Fourth = std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3>;

struct J2PointResult {
    Tensor stress{};
    Fourth tangent{};
    std::array<double,7> state{};
    double equivalent_stress=0.;
    double incremental_energy=0.;
    bool plastic=false;
};

J2PointResult update_j2_point(const Tensor& strain,const double* committed,
                              double lambda,double mu,double yield_stress,
                              double hardening) {
    J2PointResult result;
    Tensor plastic_strain{};
    plastic_strain[0][0]=committed[0]; plastic_strain[1][1]=committed[1];
    plastic_strain[2][2]=committed[2];
    plastic_strain[0][1]=plastic_strain[1][0]=committed[3];
    plastic_strain[1][2]=plastic_strain[2][1]=committed[4];
    plastic_strain[0][2]=plastic_strain[2][0]=committed[5];
    Tensor elastic_strain{},trial_stress{},deviator{};
    double trace=0.,norm_squared=0.;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        elastic_strain[i][j]=strain[i][j]-plastic_strain[i][j];
        if(i==j) trace+=elastic_strain[i][j];
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        trial_stress[i][j]=2.*mu*elastic_strain[i][j]+(i==j?lambda*trace:0.);
        deviator[i][j]=trial_stress[i][j]-(i==j?(2.*mu+3.*lambda)*trace/3.:0.);
        norm_squared+=deviator[i][j]*deviator[i][j];
    }
    const double norm=std::sqrt(norm_squared),q_trial=std::sqrt(1.5)*norm;
    const double alpha=committed[6];
    const double yield_function=q_trial-(yield_stress+hardening*alpha);
    result.plastic=yield_function>1e-12*yield_stress;
    const double delta_gamma=result.plastic?yield_function/(3.*mu+hardening):0.;
    const double scale=result.plastic?std::max(0.,1.-3.*mu*delta_gamma/q_trial):1.;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        result.stress[i][j]=scale*deviator[i][j]+
                            (i==j?(2.*mu+3.*lambda)*trace/3.:0.);
        if(result.plastic)
            plastic_strain[i][j]+=delta_gamma*1.5*deviator[i][j]/q_trial;
    }
    result.equivalent_stress=scale*q_trial;
    for(int i=0;i<3;++i) for(int j=i;j<3;++j) {
        const int index=(i==j)?i:(i==0&&j==1?3:(i==1&&j==2?4:5));
        result.state[index]=plastic_strain[i][j];
    }
    result.state[6]=alpha+delta_gamma;
    double elastic_energy=0.,elastic_trace=0.;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        const double elastic=strain[i][j]-plastic_strain[i][j];
        elastic_energy+=mu*elastic*elastic;
        if(i==j) elastic_trace+=elastic;
    }
    result.incremental_energy=elastic_energy+0.5*lambda*elastic_trace*elastic_trace
        +0.5*hardening*result.state[6]*result.state[6]
        +yield_stress*delta_gamma;
    const double bulk=lambda+2.*mu/3.;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
      for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
        const double id=0.5*((i==k&&j==l)+(i==l&&j==k))-
                         (i==j&&k==l)/3.;
        result.tangent[i][j][k][l]=bulk*(i==j&&k==l)+2.*mu*id;
        if(result.plastic) {
            const double nij=deviator[i][j]/norm,nkl=deviator[k][l]/norm;
            result.tangent[i][j][k][l]-=
                (6.*mu*mu/(3.*mu+hardening))*nij*nkl+
                (6.*mu*mu*delta_gamma/q_trial)*(id-nij*nkl);
        }
      }
    return result;
}

py::tuple evaluate_j2_material(
    py::array_t<double,py::array::c_style|py::array::forcecast> strains,
    py::array_t<double,py::array::c_style|py::array::forcecast> committed,
    double lambda,double mu,double yield_stress,double hardening) {
    const auto e=strains.request(),s=committed.request();
    if(e.ndim!=3||e.shape[1]!=3||e.shape[2]!=3)
        throw std::invalid_argument("strains must have shape (points, 3, 3)");
    const py::ssize_t n=e.shape[0];
    if(s.ndim!=2||s.shape[0]!=n||s.shape[1]!=7)
        throw std::invalid_argument("committed_state must have shape (points, 7)");
    if(!(std::isfinite(lambda)&&std::isfinite(mu)&&mu>0.&&lambda+2.*mu/3.>0.&&
         std::isfinite(yield_stress)&&yield_stress>0.&&
         std::isfinite(hardening)&&hardening>=0.))
        throw std::invalid_argument("material parameters are outside their valid range");
    const auto* ep=static_cast<const double*>(e.ptr);
    const auto* sp=static_cast<const double*>(s.ptr);
    py::array_t<double> stress(py::array::ShapeContainer{
        n,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> tangent(py::array::ShapeContainer{
        n,py::ssize_t{3},py::ssize_t{3},py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> trial(py::array::ShapeContainer{n,py::ssize_t{7}});
    py::array_t<std::int64_t> plastic(py::array::ShapeContainer{n});
    py::array_t<double> equivalent(py::array::ShapeContainer{n});
    py::array_t<double> energy(py::array::ShapeContainer{n});
    auto* so=stress.mutable_data();auto* co=tangent.mutable_data();
    auto* to=trial.mutable_data();auto* po=plastic.mutable_data();
    auto* qo=equivalent.mutable_data();
    auto* eo=energy.mutable_data();
    {
      py::gil_scoped_release release;
      native_fem::parallel_for(static_cast<std::size_t>(n),[&](std::size_t first,std::size_t last){
        for(py::ssize_t p=static_cast<py::ssize_t>(first);p<static_cast<py::ssize_t>(last);++p) {
          Tensor strain{};
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)strain[i][j]=ep[(p*3+i)*3+j];
          const auto point=update_j2_point(strain,sp+p*7,lambda,mu,yield_stress,hardening);
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)so[(p*3+i)*3+j]=point.stress[i][j];
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)
            co[((((p*3+i)*3+j)*3+k)*3+l)]=point.tangent[i][j][k][l];
          std::copy(point.state.begin(),point.state.end(),to+p*7);
          po[p]=point.plastic?1:0;qo[p]=point.equivalent_stress;
          eo[p]=point.incremental_energy;
        }
      });
    }
    return py::make_tuple(stress,tangent,trial,plastic,equivalent,energy);
}

py::tuple evaluate_j2_plasticity(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> dofs,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> displacement,
    py::array_t<double, py::array::c_style | py::array::forcecast> committed,
    double lambda, double mu, double yield_stress, double hardening) {
    const auto d=dofs.request(), g=gradients.request(), w=weights.request();
    const auto u=displacement.request(), s=committed.request();
    if (d.ndim!=3 || d.shape[2]!=3)
        throw std::invalid_argument("dofs must have shape (elements, nodes, 3)");
    const py::ssize_t ne=d.shape[0], nn=d.shape[1], nd=3*nn;
    if (g.ndim!=4 || g.shape[0]!=ne || g.shape[2]!=nn || g.shape[3]!=3)
        throw std::invalid_argument("gradients must have shape (elements, quadrature, nodes, 3)");
    const py::ssize_t nq=g.shape[1];
    if (w.ndim!=2 || w.shape[0]!=ne || w.shape[1]!=nq)
        throw std::invalid_argument("weights must have shape (elements, quadrature)");
    if (u.ndim!=1 || s.ndim!=3 || s.shape[0]!=ne || s.shape[1]!=nq || s.shape[2]!=7)
        throw std::invalid_argument("committed_state must have shape (elements, quadrature, 7)");
    if (!(std::isfinite(lambda) && std::isfinite(mu) && mu>0. && lambda+2.*mu/3.>0. &&
          std::isfinite(yield_stress) && yield_stress>0. &&
          std::isfinite(hardening) && hardening>=0.))
        throw std::invalid_argument("material parameters are outside their valid range");
    const auto* di=static_cast<const std::int64_t*>(d.ptr);
    const auto* gr=static_cast<const double*>(g.ptr);
    const auto* wt=static_cast<const double*>(w.ptr);
    const auto* uv=static_cast<const double*>(u.ptr);
    const auto* st=static_cast<const double*>(s.ptr);
    for (py::ssize_t i=0;i<d.size;++i)
        if (di[i]<0 || di[i]>=u.shape[0]) throw std::invalid_argument("dofs contain an out-of-range index");
    py::array_t<double> residual(py::array::ShapeContainer{ne,nd});
    py::array_t<double> tangent(py::array::ShapeContainer{ne,nd,nd});
    py::array_t<double> trial(py::array::ShapeContainer{ne,nq,py::ssize_t{7}});
    py::array_t<double> stress(py::array::ShapeContainer{ne,nq,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> equivalent_stress(py::array::ShapeContainer{ne,nq});
    py::array_t<double> energy(py::array::ShapeContainer{ne,nq});
    py::array_t<std::int64_t> plastic(py::array::ShapeContainer{ne,nq});
    auto* ro=residual.mutable_data(); auto* ko=tangent.mutable_data();
    auto* so=trial.mutable_data(); auto* po=plastic.mutable_data();
    auto* sigma_out=stress.mutable_data(); auto* q_out=equivalent_stress.mutable_data();
    auto* energy_out=energy.mutable_data();
    std::fill(ro,ro+residual.size(),0.); std::fill(ko,ko+tangent.size(),0.);
    std::copy(st,st+s.size,so);
    {
        py::gil_scoped_release release;
        native_fem::parallel_for(static_cast<std::size_t>(ne),[&](std::size_t first,std::size_t last) {
          for (py::ssize_t e=static_cast<py::ssize_t>(first);e<static_cast<py::ssize_t>(last);++e)
            for (py::ssize_t q=0;q<nq;++q) {
              Tensor eps{};
              for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
                for (py::ssize_t a=0;a<nn;++a) {
                  const double ga=gr[((e*nq+q)*nn+a)*3+j];
                  const double gb=gr[((e*nq+q)*nn+a)*3+i];
                  eps[i][j]+=0.5*(uv[di[(e*nn+a)*3+i]]*ga+uv[di[(e*nn+a)*3+j]]*gb);
                }
              }
              const auto si=(e*nq+q)*7;
              const auto point=update_j2_point(
                  eps,st+si,lambda,mu,yield_stress,hardening);
              for(int i=0;i<3;++i)for(int j=0;j<3;++j)
                sigma_out[((e*nq+q)*3+i)*3+j]=point.stress[i][j];
              q_out[e*nq+q]=point.equivalent_stress;
              energy_out[e*nq+q]=point.incremental_energy;
              po[e*nq+q]=point.plastic?1:0;
              std::copy(point.state.begin(),point.state.end(),so+si);
              const auto& sigma=point.stress;
              const auto& c=point.tangent;
              const double weight=wt[e*nq+q];
              for(py::ssize_t a=0;a<nn;++a) for(int i=0;i<3;++i) {
                double rv=0.; for(int j=0;j<3;++j) rv+=sigma[i][j]*gr[((e*nq+q)*nn+a)*3+j];
                ro[e*nd+a*3+i]+=weight*rv;
                for(py::ssize_t b=0;b<nn;++b) for(int k=0;k<3;++k) {
                  double kv=0.; for(int j=0;j<3;++j) for(int l=0;l<3;++l)
                    kv+=gr[((e*nq+q)*nn+a)*3+j]*c[i][j][k][l]*gr[((e*nq+q)*nn+b)*3+l];
                  ko[e*nd*nd+(a*3+i)*nd+b*3+k]+=weight*kv;
                }
              }
            }
        });
    }
    return py::make_tuple(residual,tangent,trial,plastic,stress,equivalent_stress,energy);
}
}

void native_fem::bind_j2_plasticity(py::module_& module) {
    module.def("evaluate_j2_material",&evaluate_j2_material,
      py::arg("strains"),py::arg("committed_state"),py::arg("lambda"),
      py::arg("mu"),py::arg("yield_stress"),py::arg("hardening"));
    module.def("evaluate_j2_plasticity",&evaluate_j2_plasticity,
      py::arg("dofs"),py::arg("gradients"),py::arg("weights"),py::arg("displacement"),
      py::arg("committed_state"),py::arg("lambda"),py::arg("mu"),py::arg("yield_stress"),py::arg("hardening"));
}
