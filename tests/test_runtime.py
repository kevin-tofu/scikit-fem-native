import pytest
import numpy as np

import skfn
from skfn.helpers import ddot,dot,grad


def test_native_thread_count_is_configurable():
    original=skfn.get_num_threads()
    try:
        skfn.set_num_threads(2)
        assert skfn.get_num_threads()==min(2,skfn.available_num_threads())
        with pytest.raises(ValueError,match="positive"):
            skfn.set_num_threads(0)
    finally:
        skfn.set_num_threads(original)


def test_parallel_basis_geometry_matches_single_thread():
    mesh=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,40),np.linspace(0.,1.,40)
    )
    element=skfn.ElementVector(skfn.ElementTriP1(),dim=1)
    original=skfn.get_num_threads()
    try:
        skfn.set_num_threads(1)
        serial=skfn.Basis(mesh,element)
        skfn.set_num_threads(4)
        parallel=skfn.Basis(mesh,element)
    finally:
        skfn.set_num_threads(original)
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
    assert skfn.BACKEND=="skfn-native"
    assert skfn.has_capability("native_threads")
    assert skfn.has_capability("parallel_linear_assembly")
    assert skfn.has_capability("parallel_bilinear_assembly")
    assert not skfn.has_capability("imaginary_extension")
    original=skfn.get_num_threads()
    with skfn.thread_limit(10**6) as effective:
        assert effective==skfn.available_num_threads()
        assert skfn.get_num_threads()==effective
    assert skfn.get_num_threads()==original


def test_per_call_parallel_linear_assembly_matches_serial():
    mesh=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,80),np.linspace(0.,1.,80)
    )
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP1(),dim=1)
    )

    @skfn.LinearForm
    def load(v,w):
        return dot(w.scale,v)

    scale=np.array([2.])
    serial=skfn.asm(load,basis,scale=scale,num_threads=1)
    parallel=skfn.asm(load,basis,scale=scale,num_threads=4)
    np.testing.assert_allclose(parallel,serial,rtol=2e-15,atol=2e-15)


def test_per_call_parallel_bilinear_assembly_matches_serial():
    mesh=skfn.MeshTri.init_tensor(
        np.linspace(0.,1.,80),np.linspace(0.,1.,80)
    )
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTriP1(),dim=1)
    )

    @skfn.BilinearForm
    def laplace(u,v,w):
        return ddot(grad(u),grad(v))

    serial=skfn.asm(laplace,basis,num_threads=1)
    serial_values=serial.toarray()
    parallel=skfn.asm(laplace,basis,num_threads=4)
    np.testing.assert_allclose(
        parallel.toarray(),serial_values,rtol=2e-15,atol=2e-15
    )


def test_parallel_fused_neo_hookean_matches_serial():
    axis=np.linspace(0.,1.,5)
    mesh=skfn.MeshTet.init_tensor(axis,axis,axis)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementTetP1(),dim=3)
    )
    dofs=basis.element_dofs.T.reshape(mesh.nelements,4,3)
    assembler=skfn.NativeAssembler(
        mesh.p.T,mesh.t.T,dofs,skfn.NeoHookean(mu=3.,lmbda=5.)
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
