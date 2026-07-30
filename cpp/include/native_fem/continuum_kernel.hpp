#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "native_fem/neo_hookean_tet4.hpp"

namespace native_fem {

constexpr int max_nodes = 27;
constexpr int max_dofs = 3 * max_nodes;

inline void linear_elastic_qp(
    const double* grad, int nodes, double dv, double young, double poisson,
    double* tangent) {
    const double lambda=young*poisson/((1.+poisson)*(1.-2.*poisson));
    const double mu=young/(2.*(1.+poisson));
    double b[6][max_dofs]{}, d[6][6]{};
    for(int a=0;a<nodes;++a) {
        const int c=3*a; const double* g=grad+3*a;
        b[0][c]=g[0]; b[1][c+1]=g[1]; b[2][c+2]=g[2];
        b[3][c]=g[1]; b[3][c+1]=g[0];
        b[4][c+1]=g[2]; b[4][c+2]=g[1];
        b[5][c]=g[2]; b[5][c+2]=g[0];
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        d[i][j]=lambda+(i==j?2.*mu:0.);
    d[3][3]=d[4][4]=d[5][5]=mu;
    const int ndof=3*nodes;
    for(int i=0;i<ndof;++i) for(int j=0;j<ndof;++j)
        for(int p=0;p<6;++p) for(int q=0;q<6;++q)
            tangent[i*ndof+j]+=dv*b[p][i]*d[p][q]*b[q][j];
}

inline void neo_hookean_qp(
    const double* grad, int nodes, double dv, const double* u,
    double mu, double lambda, bool with_tangent,
    double* residual, double* tangent) {
    double f[3][3]={{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
    for(int a=0;a<nodes;++a) for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        f[i][j]+=u[3*a+i]*grad[3*a+j];
    double finv[3][3]; const double jac=inverse3(f,finv);
    const double logj=std::log(jac);
    double p[3][3];
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        const double fit=finv[j][i];
        p[i][j]=mu*(f[i][j]-fit)+lambda*logj*fit;
    }
    for(int a=0;a<nodes;++a) for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        residual[3*a+i]+=dv*p[i][j]*grad[3*a+j];
    if(!with_tangent) return;
    const int ndof=3*nodes;
    for(int a=0;a<nodes;++a) for(int i=0;i<3;++i)
        for(int b=0;b<nodes;++b) for(int k=0;k<3;++k)
            for(int j=0;j<3;++j) for(int l=0;l<3;++l) {
                const double modulus=
                    mu*(i==k&&j==l?1.:0.)
                    +(mu-lambda*logj)*finv[l][i]*finv[j][k]
                    +lambda*finv[j][i]*finv[l][k];
                tangent[(3*a+i)*ndof+3*b+k]+=
                    dv*grad[3*a+j]*modulus*grad[3*b+l];
            }
}

}  // namespace native_fem
