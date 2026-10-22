#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "native_fem/python_bindings.hpp"

namespace py = pybind11;

namespace {

template <typename Index>
void accumulate_local_blocks(
    const std::int64_t* dofs,
    const double* local_residual,
    const double* local_tangent,
    const Index* tangent_scatter,
    py::ssize_t ne,
    py::ssize_t nd,
    py::ssize_t ndofs,
    py::ssize_t nnz,
    double* residual,
    double* tangent_data) {
    for (py::ssize_t e = 0; e < ne; ++e) {
        for (py::ssize_t a = 0; a < nd; ++a) {
            const auto local = e * nd + a;
            residual[dofs[local]] += local_residual[local];
        }
        const auto offset = e * nd * nd;
        for (py::ssize_t ij = 0; ij < nd * nd; ++ij) {
            tangent_data[tangent_scatter[offset + ij]] += local_tangent[offset + ij];
        }
    }
    (void) ndofs;
    (void) nnz;
}

py::tuple assemble_nonlinear_global(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> dofs,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_residual,
    py::array_t<double, py::array::c_style | py::array::forcecast> local_tangent,
    py::array tangent_scatter,
    py::ssize_t ndofs,
    py::ssize_t nnz) {
    const auto d = dofs.request();
    const auto r = local_residual.request();
    const auto k = local_tangent.request();
    const auto s = tangent_scatter.request();
    if (d.ndim != 2 || r.ndim != 2 || d.shape[0] != r.shape[0] ||
        d.shape[1] != r.shape[1])
        throw std::invalid_argument("dofs and local_residual must have matching shape (elements, local_dofs)");
    const py::ssize_t ne = d.shape[0];
    const py::ssize_t nd = d.shape[1];
    if (k.ndim != 3 || k.shape[0] != ne || k.shape[1] != nd || k.shape[2] != nd)
        throw std::invalid_argument("local_tangent must have shape (elements, local_dofs, local_dofs)");
    if (s.ndim != 3 || s.shape[0] != ne || s.shape[1] != nd || s.shape[2] != nd)
        throw std::invalid_argument("tangent_scatter must match local_tangent shape");
    if ((tangent_scatter.flags() & py::array::c_style) == 0)
        throw std::invalid_argument("tangent_scatter must be C-contiguous");
    if (ndofs < 0 || nnz < 0)
        throw std::invalid_argument("global dimensions must be nonnegative");

    const auto* di = static_cast<const std::int64_t*>(d.ptr);
    const auto* rv = static_cast<const double*>(r.ptr);
    const auto* kv = static_cast<const double*>(k.ptr);
    for (py::ssize_t i = 0; i < d.size; ++i)
        if (di[i] < 0 || di[i] >= ndofs)
            throw std::invalid_argument("dofs contain an out-of-range index");

    py::array_t<double> residual(py::array::ShapeContainer{ndofs});
    py::array_t<double> tangent_data(py::array::ShapeContainer{nnz});
    auto* ro = residual.mutable_data();
    auto* to = tangent_data.mutable_data();
    std::fill(ro, ro + ndofs, 0.);
    std::fill(to, to + nnz, 0.);

    auto scatter = [&](const auto* map) {
        for (py::ssize_t i = 0; i < s.size; ++i)
            if (map[i] < 0 || map[i] >= nnz)
                throw std::invalid_argument("tangent_scatter contains an out-of-range index");
        py::gil_scoped_release release;
        accumulate_local_blocks(di, rv, kv, map, ne, nd, ndofs, nnz, ro, to);
    };
    if (tangent_scatter.dtype().is(py::dtype::of<std::int32_t>())) {
        scatter(static_cast<const std::int32_t*>(s.ptr));
    } else if (tangent_scatter.dtype().is(py::dtype::of<std::int64_t>())) {
        scatter(static_cast<const std::int64_t*>(s.ptr));
    } else {
        throw std::invalid_argument("tangent_scatter must use int32 or int64 indices");
    }
    return py::make_tuple(residual, tangent_data);
}

}  // namespace

void native_fem::bind_nonlinear_sparse_scatter(py::module_& module) {
    module.def("assemble_nonlinear_global", &assemble_nonlinear_global,
        py::arg("dofs"), py::arg("local_residual"), py::arg("local_tangent"),
        py::arg("tangent_scatter"), py::arg("ndofs"), py::arg("nnz"));
}
