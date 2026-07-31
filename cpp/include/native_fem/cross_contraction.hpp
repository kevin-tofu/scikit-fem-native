#pragma once

#include <cstddef>

namespace native_fem {

enum class CrossBasisKind {
    value,
    gradient,
};

struct CrossBasisView {
    const double* shape;
    const double* gradient;
    int nodes;
    int dimension;
    CrossBasisKind kind;

    int directions() const {
        return kind == CrossBasisKind::gradient ? dimension : 1;
    }

    double at(int entity_quadrature, int node, int direction) const {
        if (kind == CrossBasisKind::value)
            return shape[entity_quadrature * nodes + node];
        return gradient[
            (entity_quadrature * nodes + node) * dimension + direction
        ];
    }
};

struct CrossCoefficientView {
    const double* data;
    bool tensor;
    int row_components;
    int row_dimension;
    int column_components;
    int column_dimension;
    CrossBasisKind row_kind;
    CrossBasisKind column_kind;

    double at(
        int entity_quadrature, int row_component, int row_direction,
        int column_component, int column_direction
    ) const {
        if (!tensor) {
            if (row_component != column_component)
                return 0.;
            if (
                row_kind == CrossBasisKind::gradient
                && row_direction != column_direction
            )
                return 0.;
            return data ? data[entity_quadrature] : 1.;
        }
        std::size_t index = entity_quadrature;
        index = index * row_components + row_component;
        if (row_kind == CrossBasisKind::gradient)
            index = index * row_dimension + row_direction;
        index = index * column_components + column_component;
        if (column_kind == CrossBasisKind::gradient)
            index = index * column_dimension + column_direction;
        return data[index];
    }
};

inline double contract_cross_basis(
    const CrossBasisView& row, const CrossBasisView& column,
    const CrossCoefficientView& coefficient, int entity_quadrature,
    int row_node, int column_node, int row_component,
    int column_component
) {
    double result = 0.;
    for (int i = 0; i < row.directions(); ++i)
        for (int j = 0; j < column.directions(); ++j)
            result +=
                row.at(entity_quadrature, row_node, i)
                * coefficient.at(
                    entity_quadrature, row_component, i,
                    column_component, j
                )
                * column.at(entity_quadrature, column_node, j);
    return result;
}

inline double contract_scalar_cross_basis(
    const CrossBasisView& row, const CrossBasisView& column,
    int entity_quadrature, int row_node, int column_node
) {
    if (row.kind == CrossBasisKind::value)
        return row.at(entity_quadrature, row_node, 0)
            * column.at(entity_quadrature, column_node, 0);
    double result = 0.;
    for (int direction = 0; direction < row.dimension; ++direction)
        result += row.at(entity_quadrature, row_node, direction)
            * column.at(entity_quadrature, column_node, direction);
    return result;
}

}  // namespace native_fem
