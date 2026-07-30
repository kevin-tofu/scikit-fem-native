#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace native_fem {

struct NeoHookeanTet4Result {
    std::array<double, 12> residual{};
    std::array<double, 144> tangent{};
};

inline double inverse3(
    const double f[3][3], double inv[3][3], bool allow_negative = false) {
    const double det =
        f[0][0]*(f[1][1]*f[2][2]-f[1][2]*f[2][1])
        - f[0][1]*(f[1][0]*f[2][2]-f[1][2]*f[2][0])
        + f[0][2]*(f[1][0]*f[2][1]-f[1][1]*f[2][0]);
    if ((!allow_negative && !(det > 0.0)) || std::abs(det) < 1e-14)
        throw std::invalid_argument("deformation or element Jacobian is non-positive");
    const double s = 1.0 / det;
    inv[0][0]=(f[1][1]*f[2][2]-f[1][2]*f[2][1])*s;
    inv[0][1]=(f[0][2]*f[2][1]-f[0][1]*f[2][2])*s;
    inv[0][2]=(f[0][1]*f[1][2]-f[0][2]*f[1][1])*s;
    inv[1][0]=(f[1][2]*f[2][0]-f[1][0]*f[2][2])*s;
    inv[1][1]=(f[0][0]*f[2][2]-f[0][2]*f[2][0])*s;
    inv[1][2]=(f[0][2]*f[1][0]-f[0][0]*f[1][2])*s;
    inv[2][0]=(f[1][0]*f[2][1]-f[1][1]*f[2][0])*s;
    inv[2][1]=(f[0][1]*f[2][0]-f[0][0]*f[2][1])*s;
    inv[2][2]=(f[0][0]*f[1][1]-f[0][1]*f[1][0])*s;
    return det;
}

inline NeoHookeanTet4Result neo_hookean_tet4(
    const double grad[4][3], double volume, const double* u,
    double mu, double lambda, bool with_tangent) {
    double f[3][3] = {{1.,0.,0.}, {0.,1.,0.}, {0.,0.,1.}};
    for (int a=0; a<4; ++a) for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) f[i][j] += u[3*a+i]*grad[a][j];
    double finv[3][3];
    const double jacobian = inverse3(f, finv);
    const double logj = std::log(jacobian);
    double p[3][3];
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        const double fit = finv[j][i];
        p[i][j] = mu*(f[i][j]-fit) + lambda*logj*fit;
    }
    NeoHookeanTet4Result out;
    for (int a=0; a<4; ++a) for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            out.residual[3*a+i] += volume*p[i][j]*grad[a][j];
    if (!with_tangent) return out;
    for (int a=0; a<4; ++a) for (int i=0; i<3; ++i)
        for (int b=0; b<4; ++b) for (int k=0; k<3; ++k)
            for (int j=0; j<3; ++j) for (int l=0; l<3; ++l) {
                const double modulus =
                    mu*(i==k && j==l ? 1. : 0.)
                    + (mu-lambda*logj)*finv[l][i]*finv[j][k]
                    + lambda*finv[j][i]*finv[l][k];
                out.tangent[(3*a+i)*12+3*b+k] +=
                    volume*grad[a][j]*modulus*grad[b][l];
            }
    return out;
}

}  // namespace native_fem
