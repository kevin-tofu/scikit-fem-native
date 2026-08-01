import pytest
import numpy as np

import skfemntv
from skfemntv.helpers import ddot,dot,grad


def test_native_thread_count_is_configurable():
    original=skfemntv.get_num_threads()
    try:
        skfemntv.set_num_threads(2)
        assert skfemntv.get_num_threads()==min(2,skfemntv.available_num_threads())
        with pytest.raises(ValueError,match="positive"):
            skfemntv.set_num_threads(0)
    finally:
        skfemntv.set_num_threads(original)


def test_parallel_basis_geometry_matches_single_thread():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,40),np.linspace(0.,1.,40)
    )
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    original=skfemntv.get_num_threads()
    try:
        skfemntv.set_num_threads(1)
        serial=skfemntv.Basis(mesh,element)
        skfemntv.set_num_threads(4)
        parallel=skfemntv.Basis(mesh,element)
    finally:
        skfemntv.set_num_threads(original)
    np.testing.assert_array_equal(
        parallel.tabulated_shape,serial.tabulated_shape
    )
    np.testing.assert_array_equal(
        parallel.tabulated_gradients,serial.tabulated_gradients
    )
    np.testing.assert_array_equal(parallel.dx,serial.dx)
    np.testing.assert_array_equal(
        parallel.global_coordinates,serial.global_coordinates
    )


def test_native_capability_branch_and_temporary_thread_limit():
    assert skfemntv.BACKEND=="skfemntv-native"
    assert skfemntv.has_capability("native_threads")
    assert skfemntv.has_capability("parallel_linear_assembly")
    assert skfemntv.has_capability("parallel_bilinear_assembly")
    assert not skfemntv.has_capability("imaginary_extension")
    original=skfemntv.get_num_threads()
    with skfemntv.thread_limit(10**6) as effective:
        assert effective==skfemntv.available_num_threads()
        assert skfemntv.get_num_threads()==effective
    assert skfemntv.get_num_threads()==original


def test_per_call_parallel_linear_assembly_matches_serial():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,80),np.linspace(0.,1.,80)
    )
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )

    @skfemntv.LinearForm
    def load(v,w):
        return dot(w.scale,v)

    scale=np.array([2.])
    serial=skfemntv.asm(load,basis,scale=scale,num_threads=1)
    parallel=skfemntv.asm(load,basis,scale=scale,num_threads=4)
    np.testing.assert_allclose(parallel,serial,rtol=2e-15,atol=2e-15)


def test_per_call_parallel_bilinear_assembly_matches_serial():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,80),np.linspace(0.,1.,80)
    )
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )

    @skfemntv.BilinearForm
    def laplace(u,v,w):
        return ddot(grad(u),grad(v))

    serial=skfemntv.asm(laplace,basis,num_threads=1)
    serial_values=serial.toarray()
    parallel=skfemntv.asm(laplace,basis,num_threads=4)
    np.testing.assert_allclose(
        parallel.toarray(),serial_values,rtol=2e-15,atol=2e-15
    )


def test_parallel_fused_neo_hookean_matches_serial():
    axis=np.linspace(0.,1.,5)
    mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3)
    )
    dofs=basis.element_dofs.T.reshape(mesh.nelements,4,3)
    assembler=skfemntv.NativeAssembler(
        mesh.p.T,mesh.t.T,dofs,skfemntv.NeoHookean(mu=3.,lmbda=5.)
    )
    u=np.zeros(assembler.ndofs,dtype=np.float64)
    serial=assembler.assemble(u,num_threads=1)
    serial_residual=serial.residual.copy()
    serial_tangent=serial.tangent.copy()
    parallel=assembler.assemble(u,num_threads=4)
    np.testing.assert_allclose(
        parallel.residual,serial_residual,rtol=2e-15,atol=2e-15
    )
    np.testing.assert_allclose(
        parallel.tangent.toarray(),serial_tangent.toarray(),
        rtol=2e-15,atol=2e-15,
    )


def test_native_assembler_reports_parallel_coloring():
    axis=np.linspace(0.,1.,4)
    mesh=skfemntv.MeshHex.init_tensor(axis,axis,axis)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementHex1(),dim=3)
    )
    diagnostics=skfemntv.NativeAssembler.from_basis(
        basis,skfemntv.NeoHookean(3.,5.)
    ).parallel_diagnostics
    assert diagnostics=={
        "color_count":8,
        "min_color_size":1,
        "max_color_size":8,
        "explicit_thread_threshold":128,
        "parallel_eligible_color_count":0,
    }
