#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

void integrate_rt0(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> divergences,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style | py::array::forcecast> value_coefficient,
    py::array_t<double, py::array::c_style | py::array::forcecast> divergence_coefficient,
    bool include_value,
    bool include_divergence,
    py::array_t<double, py::array::c_style> output) {
    const auto v = values.request();
    const auto d = divergences.request();
    const auto measure = dx.request();
    const auto vc = value_coefficient.request();
    const auto dc = divergence_coefficient.request();
    const auto out = output.request();
    if (v.ndim != 4 || d.ndim != 3 || measure.ndim != 2 ||
        vc.ndim != 2 || dc.ndim != 2 || out.ndim != 3)
        throw std::invalid_argument("invalid RT0 integration input rank");
    const py::ssize_t cells = v.shape[0];
    const py::ssize_t local = v.shape[1];
    const py::ssize_t dim = v.shape[2];
    const py::ssize_t points = v.shape[3];
    if ((dim != 2 || local != 3) && (dim != 3 || local != 4))
        throw std::invalid_argument("RT0 supports triangles and tetrahedra");
    if (d.shape[0] != cells || d.shape[1] != local || d.shape[2] != points ||
        measure.shape[0] != cells || measure.shape[1] != points ||
        vc.shape[0] != cells || vc.shape[1] != points ||
        dc.shape[0] != cells || dc.shape[1] != points ||
        out.shape[0] != cells || out.shape[1] != local || out.shape[2] != local)
        throw std::invalid_argument("incompatible RT0 integration shapes");

    const auto* value_data = static_cast<const double*>(v.ptr);
    const auto* divergence_data = static_cast<const double*>(d.ptr);
    const auto* dx_data = static_cast<const double*>(measure.ptr);
    const auto* value_coefficient_data = static_cast<const double*>(vc.ptr);
    const auto* divergence_coefficient_data = static_cast<const double*>(dc.ptr);
    auto* output_data = static_cast<double*>(out.ptr);
    {
        py::gil_scoped_release release;
        native_fem::parallel_for(static_cast<std::size_t>(cells),
            [&](std::size_t begin, std::size_t end) {
                for (py::ssize_t cell = static_cast<py::ssize_t>(begin);
                     cell < static_cast<py::ssize_t>(end); ++cell) {
                    for (py::ssize_t row = 0; row < local; ++row) {
                        for (py::ssize_t column = 0; column < local; ++column) {
                            double result = 0.;
                            for (py::ssize_t q = 0; q < points; ++q) {
                                double value_dot = 0.;
                                if (include_value) {
                                    for (py::ssize_t component = 0; component < dim; ++component) {
                                        const auto ri = ((cell * local + row) * dim + component) * points + q;
                                        const auto ci = ((cell * local + column) * dim + component) * points + q;
                                        value_dot += value_data[ri] * value_data[ci];
                                    }
                                }
                                const auto ri = (cell * local + row) * points + q;
                                const auto ci = (cell * local + column) * points + q;
                                const double div_dot = include_divergence
                                    ? divergence_data[ri] * divergence_data[ci] : 0.;
                                const auto qi = cell * points + q;
                                result += dx_data[qi] * (
                                    value_coefficient_data[qi] * value_dot +
                                    divergence_coefficient_data[qi] * div_dot);
                            }
                            output_data[(cell * local + row) * local + column] = result;
                        }
                    }
                }
            });
    }
}

void integrate_rt0_vector_load(
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> field,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style> output) {
    const auto v = values.request();
    const auto f = field.request();
    const auto measure = dx.request();
    const auto out = output.request();
    if (v.ndim != 4 || f.ndim != 3 || measure.ndim != 2 || out.ndim != 2)
        throw std::invalid_argument("invalid RT0 load input rank");
    const py::ssize_t cells = v.shape[0];
    const py::ssize_t local = v.shape[1];
    const py::ssize_t dim = v.shape[2];
    const py::ssize_t points = v.shape[3];
    if (f.shape[0] != dim || f.shape[1] != cells || f.shape[2] != points ||
        measure.shape[0] != cells || measure.shape[1] != points ||
        out.shape[0] != cells || out.shape[1] != local)
        throw std::invalid_argument("incompatible RT0 load shapes");
    const auto* value_data = static_cast<const double*>(v.ptr);
    const auto* field_data = static_cast<const double*>(f.ptr);
    const auto* dx_data = static_cast<const double*>(measure.ptr);
    auto* output_data = static_cast<double*>(out.ptr);
    {
        py::gil_scoped_release release;
        native_fem::parallel_for(static_cast<std::size_t>(cells),
            [&](std::size_t begin, std::size_t end) {
                for (py::ssize_t cell = static_cast<py::ssize_t>(begin);
                     cell < static_cast<py::ssize_t>(end); ++cell) {
                    for (py::ssize_t basis = 0; basis < local; ++basis) {
                        double result = 0.;
                        for (py::ssize_t q = 0; q < points; ++q) {
                            double dot = 0.;
                            for (py::ssize_t component = 0; component < dim; ++component) {
                                const auto vi = ((cell * local + basis) * dim + component) * points + q;
                                const auto fi = (component * cells + cell) * points + q;
                                dot += value_data[vi] * field_data[fi];
                            }
                            result += dx_data[cell * points + q] * dot;
                        }
                        output_data[cell * local + basis] = result;
                    }
                }
            });
    }
}

void integrate_rt0_dg0_divergence(
    py::array_t<double, py::array::c_style | py::array::forcecast> rt0_divergences,
    py::array_t<double, py::array::c_style | py::array::forcecast> scalar_shapes,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx,
    py::array_t<double, py::array::c_style | py::array::forcecast> coefficient,
    bool rt0_is_test,
    py::array_t<double, py::array::c_style> output) {
    const auto div = rt0_divergences.request();
    const auto shape = scalar_shapes.request();
    const auto measure = dx.request();
    const auto weight = coefficient.request();
    const auto out = output.request();
    if (div.ndim != 3 || shape.ndim != 3 || measure.ndim != 2 ||
        weight.ndim != 2 || out.ndim != 3)
        throw std::invalid_argument("invalid RT0-DG0 integration input rank");
    const py::ssize_t cells = div.shape[0];
    const py::ssize_t rt0_local = div.shape[1];
    const py::ssize_t points = div.shape[2];
    const py::ssize_t scalar_local = shape.shape[1];
    const py::ssize_t rows = rt0_is_test ? rt0_local : scalar_local;
    const py::ssize_t columns = rt0_is_test ? scalar_local : rt0_local;
    if (shape.shape[0] != cells || shape.shape[2] != points ||
        measure.shape[0] != cells || measure.shape[1] != points ||
        weight.shape[0] != cells || weight.shape[1] != points ||
        out.shape[0] != cells || out.shape[1] != rows ||
        out.shape[2] != columns)
        throw std::invalid_argument("incompatible RT0-DG0 integration shapes");
    const auto* div_data = static_cast<const double*>(div.ptr);
    const auto* shape_data = static_cast<const double*>(shape.ptr);
    const auto* dx_data = static_cast<const double*>(measure.ptr);
    const auto* coefficient_data = static_cast<const double*>(weight.ptr);
    auto* output_data = static_cast<double*>(out.ptr);
    {
        py::gil_scoped_release release;
        native_fem::parallel_for(static_cast<std::size_t>(cells),
            [&](std::size_t begin, std::size_t end) {
                for (py::ssize_t cell = static_cast<py::ssize_t>(begin);
                     cell < static_cast<py::ssize_t>(end); ++cell) {
                    for (py::ssize_t row = 0; row < rows; ++row) {
                        for (py::ssize_t column = 0; column < columns; ++column) {
                            const py::ssize_t rt0_index = rt0_is_test ? row : column;
                            const py::ssize_t scalar_index = rt0_is_test ? column : row;
                            double result = 0.;
                            for (py::ssize_t q = 0; q < points; ++q) {
                                const auto di = (cell * rt0_local + rt0_index) * points + q;
                                const auto si = (cell * scalar_local + scalar_index) * points + q;
                                const auto qi = cell * points + q;
                                result += dx_data[qi] * coefficient_data[qi] *
                                    div_data[di] * shape_data[si];
                            }
                            output_data[(cell * rows + row) * columns + column] = result;
                        }
                    }
                }
            });
    }
}

}  // namespace

void native_fem::bind_hdiv_assembler(py::module_& module) {
    module.def("integrate_rt0", &integrate_rt0);
    module.def("integrate_rt0_vector_load", &integrate_rt0_vector_load);
    module.def(
        "integrate_rt0_dg0_divergence",
        &integrate_rt0_dg0_divergence
    );
}
