#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

using Matrix3 = std::array<std::array<double, 3>, 3>;
using Tensor4 = std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3>;

struct NeoHookeanPointResult {
    Matrix3 stress{};
    Matrix3 cauchy_stress{};
    Tensor4 tangent{};
    Matrix3 inverse_transpose{};
    double energy=0.;
    double equivalent_stress=0.;
    double tangent_geometric_scale=0.;
};

double determinant(const Matrix3& a) {
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
         - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
         + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

Matrix3 inverse(const Matrix3& a, double det) {
    Matrix3 result{};
    result[0][0] = (a[1][1]*a[2][2] - a[1][2]*a[2][1]) / det;
    result[0][1] = (a[0][2]*a[2][1] - a[0][1]*a[2][2]) / det;
    result[0][2] = (a[0][1]*a[1][2] - a[0][2]*a[1][1]) / det;
    result[1][0] = (a[1][2]*a[2][0] - a[1][0]*a[2][2]) / det;
    result[1][1] = (a[0][0]*a[2][2] - a[0][2]*a[2][0]) / det;
    result[1][2] = (a[0][2]*a[1][0] - a[0][0]*a[1][2]) / det;
    result[2][0] = (a[1][0]*a[2][1] - a[1][1]*a[2][0]) / det;
    result[2][1] = (a[0][1]*a[2][0] - a[0][0]*a[2][1]) / det;
    result[2][2] = (a[0][0]*a[1][1] - a[0][1]*a[1][0]) / det;
    return result;
}

NeoHookeanPointResult evaluate_neo_hookean_point(
    const Matrix3& f,double mu,double lambda,bool compute_tangent=true) {
    const double j=determinant(f);
    if(!(std::isfinite(j)&&j>0.))
        throw std::domain_error("Neo-Hookean material encountered nonpositive det(F)");
    const Matrix3 inv_f=inverse(f,j);
    Matrix3 inv_ft{};
    double i1=0.;
    for(int i=0;i<3;++i)for(int k=0;k<3;++k) {
        inv_ft[i][k]=inv_f[k][i];
        i1+=f[i][k]*f[i][k];
    }
    const double log_j=std::log(j);
    NeoHookeanPointResult result;
    result.inverse_transpose=inv_ft;
    result.tangent_geometric_scale=mu-lambda*log_j;
    result.energy=mu*.5*(i1-3.)-mu*log_j+lambda*.5*log_j*log_j;
    for(int i=0;i<3;++i)for(int k=0;k<3;++k)
        result.stress[i][k]=mu*(f[i][k]-inv_ft[i][k])+
                            lambda*log_j*inv_ft[i][k];
    double cauchy_trace=0.;
    for(int i=0;i<3;++i)for(int k=0;k<3;++k) {
        for(int jidx=0;jidx<3;++jidx)
            result.cauchy_stress[i][k]+=result.stress[i][jidx]*f[k][jidx]/j;
        if(i==k)cauchy_trace+=result.cauchy_stress[i][k];
    }
    double deviatoric_norm_squared=0.;
    for(int i=0;i<3;++i)for(int k=0;k<3;++k) {
        const double dev=result.cauchy_stress[i][k]-(i==k?cauchy_trace/3.:0.);
        deviatoric_norm_squared+=dev*dev;
    }
    result.equivalent_stress=std::sqrt(1.5*deviatoric_norm_squared);
    if(compute_tangent)
        for(int i=0;i<3;++i)for(int jidx=0;jidx<3;++jidx)
          for(int k=0;k<3;++k)for(int l=0;l<3;++l)
            result.tangent[i][jidx][k][l]=
                mu*(i==k&&jidx==l?1.:0.)+
                (mu-lambda*log_j)*inv_ft[i][l]*inv_ft[k][jidx]+
                lambda*inv_ft[i][jidx]*inv_ft[k][l];
    return result;
}

py::tuple evaluate_neo_hookean_material(
    py::array_t<double,py::array::c_style|py::array::forcecast> deformation_gradients,
    double mu,double lambda) {
    const auto f=deformation_gradients.request();
    if(f.ndim!=3||f.shape[1]!=3||f.shape[2]!=3)
        throw std::invalid_argument("deformation_gradients must have shape (points, 3, 3)");
    if(!(std::isfinite(mu)&&mu>0.&&std::isfinite(lambda)&&lambda>=0.))
        throw std::invalid_argument("material parameters must be finite with mu > 0 and lambda >= 0");
    const py::ssize_t n=f.shape[0];
    const auto* fd=static_cast<const double*>(f.ptr);
    py::array_t<double> stress(py::array::ShapeContainer{n,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> cauchy(py::array::ShapeContainer{n,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> tangent(py::array::ShapeContainer{
        n,py::ssize_t{3},py::ssize_t{3},py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> energy(py::array::ShapeContainer{n});
    py::array_t<double> equivalent(py::array::ShapeContainer{n});
    auto* sd=stress.mutable_data();auto* xd=cauchy.mutable_data();
    auto* cd=tangent.mutable_data();auto* ed=energy.mutable_data();
    auto* qd=equivalent.mutable_data();
    {
      py::gil_scoped_release release;
      native_fem::parallel_for(static_cast<std::size_t>(n),[&](std::size_t first,std::size_t last){
        for(py::ssize_t p=static_cast<py::ssize_t>(first);p<static_cast<py::ssize_t>(last);++p) {
          Matrix3 fp{};
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)fp[i][j]=fd[(p*3+i)*3+j];
          const auto point=evaluate_neo_hookean_point(fp,mu,lambda);
          ed[p]=point.energy;
          qd[p]=point.equivalent_stress;
          for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            sd[(p*3+i)*3+j]=point.stress[i][j];
            xd[(p*3+i)*3+j]=point.cauchy_stress[i][j];
          }
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)
            cd[((((p*3+i)*3+j)*3+k)*3+l)]=point.tangent[i][j][k][l];
        }
      });
    }
    return py::make_tuple(stress,tangent,energy,cauchy,equivalent);
}

py::tuple evaluate_neo_hookean(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> dofs,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradients,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> displacement,
    double mu,
    double lambda) {
    const auto d = dofs.request();
    const auto g = gradients.request();
    const auto w = weights.request();
    const auto u = displacement.request();
    if (d.ndim != 3 || d.shape[2] != 3)
        throw std::invalid_argument("dofs must have shape (elements, nodes, 3)");
    const py::ssize_t elements = d.shape[0];
    const py::ssize_t nodes = d.shape[1];
    const py::ssize_t local_dofs = nodes * 3;
    if (g.ndim != 4 || g.shape[0] != elements || g.shape[2] != nodes || g.shape[3] != 3)
        throw std::invalid_argument("gradients must have shape (elements, quadrature, nodes, 3)");
    const py::ssize_t quadrature = g.shape[1];
    if (w.ndim != 2 || w.shape[0] != elements || w.shape[1] != quadrature)
        throw std::invalid_argument("weights must have shape (elements, quadrature)");
    if (u.ndim != 1)
        throw std::invalid_argument("displacement must be one-dimensional");
    if (!(std::isfinite(mu) && mu > 0.0))
        throw std::invalid_argument("mu must be positive and finite");
    if (!(std::isfinite(lambda) && lambda >= 0.0))
        throw std::invalid_argument("lambda must be nonnegative and finite");

    const auto* dof_data = static_cast<const std::int64_t*>(d.ptr);
    const auto* grad_data = static_cast<const double*>(g.ptr);
    const auto* weight_data = static_cast<const double*>(w.ptr);
    const auto* u_data = static_cast<const double*>(u.ptr);
    for (py::ssize_t index = 0; index < d.size; ++index)
        if (dof_data[index] < 0 || dof_data[index] >= u.shape[0])
            throw std::invalid_argument("dofs contain an out-of-range index");

    py::array_t<double> residual({elements, local_dofs});
    py::array_t<double> tangent({elements, local_dofs, local_dofs});
    py::array_t<double> energy({elements});
    py::array_t<double> stress(py::array::ShapeContainer{
        elements,quadrature,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> cauchy(py::array::ShapeContainer{
        elements,quadrature,py::ssize_t{3},py::ssize_t{3}});
    py::array_t<double> equivalent_stress(py::array::ShapeContainer{elements,quadrature});
    py::array_t<double> deformation_gradients(py::array::ShapeContainer{
        elements,quadrature,py::ssize_t{3},py::ssize_t{3}});
    auto* r_out = residual.mutable_data();
    auto* k_out = tangent.mutable_data();
    auto* e_out = energy.mutable_data();
    auto* s_out=stress.mutable_data();
    auto* x_out=cauchy.mutable_data();
    auto* q_out=equivalent_stress.mutable_data();
    auto* f_out=deformation_gradients.mutable_data();
    std::fill(r_out,r_out+residual.size(),0.0);
    std::fill(k_out,k_out+tangent.size(),0.0);

    {
        py::gil_scoped_release release;
        native_fem::parallel_for(static_cast<std::size_t>(elements),
            [&](std::size_t begin, std::size_t end) {
            for (py::ssize_t e = static_cast<py::ssize_t>(begin);
                 e < static_cast<py::ssize_t>(end); ++e) {
                double element_energy = 0.0;
                std::vector<double> transformed_gradient(
                    static_cast<std::size_t>(nodes)*3);
                std::vector<double> gradient_dot(
                    static_cast<std::size_t>(nodes)*nodes);
                for (py::ssize_t q = 0; q < quadrature; ++q) {
                    Matrix3 f{};
                    for (int i = 0; i < 3; ++i) f[i][i] = 1.0;
                    for (py::ssize_t a = 0; a < nodes; ++a)
                        for (int j = 0; j < 3; ++j) {
                            const double grad = grad_data[((e*quadrature+q)*nodes+a)*3+j];
                            for (int i = 0; i < 3; ++i) {
                                const auto dof = dof_data[(e*nodes+a)*3+i];
                                f[i][j] += u_data[dof] * grad;
                            }
                        }
                    const auto point=evaluate_neo_hookean_point(f,mu,lambda,false);
                    for(py::ssize_t a=0;a<nodes;++a) {
                        const double* grad_a=grad_data+
                            ((e*quadrature+q)*nodes+a)*3;
                        for(int i=0;i<3;++i) {
                            double value=0.;
                            for(int j=0;j<3;++j)
                                value+=point.inverse_transpose[i][j]*grad_a[j];
                            transformed_gradient[a*3+i]=value;
                        }
                        for(py::ssize_t b=0;b<nodes;++b) {
                            const double* grad_b=grad_data+
                                ((e*quadrature+q)*nodes+b)*3;
                            double value=0.;
                            for(int j=0;j<3;++j)value+=grad_a[j]*grad_b[j];
                            gradient_dot[a*nodes+b]=value;
                        }
                    }
                    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
                        const auto index=((e*quadrature+q)*3+i)*3+j;
                        s_out[index]=point.stress[i][j];
                        x_out[index]=point.cauchy_stress[i][j];
                        f_out[index]=f[i][j];
                    }
                    q_out[e*quadrature+q]=point.equivalent_stress;
                    const double weight = weight_data[e*quadrature+q];
                    element_energy += weight * point.energy;

                    for (py::ssize_t a = 0; a < nodes; ++a)
                        for (int i = 0; i < 3; ++i) {
                            double value = 0.0;
                            for (int j = 0; j < 3; ++j)
                                value += point.stress[i][j]
                                    * grad_data[((e*quadrature+q)*nodes+a)*3+j];
                            r_out[e*local_dofs+a*3+i] += weight * value;
                        }
                    for (py::ssize_t a = 0; a < nodes; ++a)
                        for (int i = 0; i < 3; ++i)
                            for (py::ssize_t b = 0; b < nodes; ++b)
                                for (int k = 0; k < 3; ++k) {
                                    const double* ta=transformed_gradient.data()+a*3;
                                    const double* tb=transformed_gradient.data()+b*3;
                                    const double value=mu*(i==k?gradient_dot[a*nodes+b]:0.)+
                                        point.tangent_geometric_scale*tb[i]*ta[k]+
                                        lambda*ta[i]*tb[k];
                                    k_out[e*local_dofs*local_dofs
                                          +(a*3+i)*local_dofs+b*3+k]
                                        += weight * value;
                                }
                }
                e_out[e] = element_energy;
            }
        });
    }
    return py::make_tuple(residual,tangent,energy,stress,deformation_gradients,
                          cauchy,equivalent_stress);
}

}  // namespace

void native_fem::bind_neo_hookean(py::module_& module) {
    module.def("evaluate_neo_hookean_material", &evaluate_neo_hookean_material,
        py::arg("deformation_gradients"),py::arg("mu"),py::arg("lambda"));
    module.def("evaluate_neo_hookean", &evaluate_neo_hookean,
        py::arg("dofs"), py::arg("gradients"), py::arg("weights"),
        py::arg("displacement"), py::arg("mu"), py::arg("lambda"));
}
