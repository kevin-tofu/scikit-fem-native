#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace native_fem {

struct Tet4Result {
    std::array<double, 12 * 12> tangent{};
    double volume{};
};

inline Tet4Result linear_elastic_tet4(
    const double* x, double young_modulus, double poisson_ratio) {
    if (!(young_modulus > 0.0))
        throw std::invalid_argument("young_modulus must be positive");
    if (!(poisson_ratio > -1.0 && poisson_ratio < 0.5))
        throw std::invalid_argument("poisson_ratio must be in (-1, 0.5)");

    const double a00 = x[3] - x[0], a01 = x[6] - x[0], a02 = x[9] - x[0];
    const double a10 = x[4] - x[1], a11 = x[7] - x[1], a12 = x[10] - x[1];
    const double a20 = x[5] - x[2], a21 = x[8] - x[2], a22 = x[11] - x[2];
    const double det =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    if (std::abs(det) < 1e-14)
        throw std::invalid_argument("Tet4 has a singular Jacobian");

    const double invdet = 1.0 / det;
    const double inv[3][3] = {
        {(a11*a22-a12*a21)*invdet, (a02*a21-a01*a22)*invdet, (a01*a12-a02*a11)*invdet},
        {(a12*a20-a10*a22)*invdet, (a00*a22-a02*a20)*invdet, (a02*a10-a00*a12)*invdet},
        {(a10*a21-a11*a20)*invdet, (a01*a20-a00*a21)*invdet, (a00*a11-a01*a10)*invdet}
    };
    constexpr double dndr[4][3] = {
        {-1., -1., -1.}, {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}
    };
    double grad[4][3]{};
    for (int n = 0; n < 4; ++n)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                grad[n][j] += dndr[n][k] * inv[k][j];

    double b[6][12]{};
    for (int n = 0; n < 4; ++n) {
        const int c = 3 * n;
        b[0][c] = grad[n][0];
        b[1][c+1] = grad[n][1];
        b[2][c+2] = grad[n][2];
        b[3][c] = grad[n][1]; b[3][c+1] = grad[n][0];
        b[4][c+1] = grad[n][2]; b[4][c+2] = grad[n][1];
        b[5][c] = grad[n][2]; b[5][c+2] = grad[n][0];
    }
    const double lambda = young_modulus * poisson_ratio /
                          ((1. + poisson_ratio) * (1. - 2. * poisson_ratio));
    const double mu = young_modulus / (2. * (1. + poisson_ratio));
    double d[6][6]{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d[i][j] = lambda + (i == j ? 2. * mu : 0.);
    d[3][3] = d[4][4] = d[5][5] = mu;

    Tet4Result result;
    result.volume = std::abs(det) / 6.;
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            for (int p = 0; p < 6; ++p)
                for (int q = 0; q < 6; ++q)
                    result.tangent[i*12+j] +=
                        result.volume * b[p][i] * d[p][q] * b[q][j];
    return result;
}

}  // namespace native_fem
