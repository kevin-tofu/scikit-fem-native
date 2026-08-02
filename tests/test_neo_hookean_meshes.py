from pathlib import Path
import sys

import numpy as np
import pytest
import skfem

import skfemntv


sys.path.insert(0,str(Path(__file__).parents[1]/"benchmarks"))
from skfem_neo_hookean import forms as reference_forms


def distort(points):
    points=points.copy();x,y,z=points.copy()
    points[0]=x+.045*y*z
    points[1]=y-.035*x*z
    points[2]=z+.025*x*y
    return points


def native_space(topology,intorder=4,*,multi=False):
    if topology=="tet10":
        linear=(
            skfemntv.MeshTet.init_tensor([0.,1.],[0.,1.],[0.,1.])
            if multi else skfemntv.MeshTet()
        )
        mesh=skfemntv.MeshTet2.from_mesh(linear)
        element=skfemntv.ElementTetP2()
    elif topology=="hex27":
        linear=(
            skfemntv.MeshHex.init_tensor([0.,.5,1.],[0.,1.],[0.,1.])
            if multi else skfemntv.MeshHex()
        )
        mesh=skfemntv.MeshHex2.from_mesh(linear)
        element=skfemntv.ElementHex2()
    elif topology=="wedge6":
        mesh=(
            skfemntv.MeshWedge1.init_tensor([0.,1.],[0.,1.],[0.,1.])
            if multi else skfemntv.MeshWedge1()
        )
        element=skfemntv.ElementWedge1()
    elif topology=="pyramid5":
        mesh=skfemntv.MeshPyramid1()
        if multi:
            points=np.hstack((mesh.p,mesh.p+np.array([[1.5],[0.],[0.]])))
            cells=np.hstack((mesh.t,mesh.t+mesh.p.shape[1]))
            mesh=skfemntv.MeshPyramid1(points,cells)
        element=skfemntv.ElementPyramid1()
    else:
        raise ValueError(topology)
    mesh=type(mesh)(distort(mesh.p),mesh.t)
    return skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=3),intorder=intorder
    )


def reference_space(topology,native):
    if topology=="tet10":
        mesh=skfem.MeshTet2(native.mesh.p,native.mesh.t)
        element=skfem.ElementTetP2()
    elif topology=="hex27":
        template=skfem.MeshHex2.from_mesh(skfem.MeshHex())
        mesh=skfem.MeshHex2(distort(template.p),template.t)
        element=skfem.ElementHex2()
    elif topology=="wedge6":
        mesh=skfem.MeshWedge1(native.mesh.p,native.mesh.t)
        element=skfem.ElementWedge1()
    else:
        raise ValueError(topology)
    return skfem.Basis(
        mesh,skfem.ElementVector(element),quadrature=(native.X,native.W)
    )


def permutation(native,reference):
    lookup={
        tuple(np.round(reference.doflocs[:,3*node],13)):node
        for node in range(reference.N//3)
    }
    result=[]
    for node in range(native.N//3):
        other=lookup[tuple(np.round(native.doflocs[:,3*node],13))]
        result.extend(3*other+component for component in range(3))
    return np.asarray(result)


@pytest.mark.parametrize("topology",["tet10","hex27","wedge6"])
@pytest.mark.parametrize("intorder",[2,4,6])
def test_neo_hookean_matches_scikit_fem_forms(topology,intorder):
    native=native_space(topology,intorder)
    reference=reference_space(topology,native)
    order=permutation(native,reference)
    mu,lmbda=7.,11.
    rng=np.random.default_rng(120+intorder)
    u=np.ascontiguousarray(rng.normal(scale=1.5e-3,size=native.N))
    reference_u=np.empty(reference.N);reference_u[order]=u
    field=reference.interpolate(reference_u)
    residual_form,tangent_form=reference_forms(mu,lmbda)
    expected_residual=residual_form.assemble(
        reference,displacement=field
    )[order]
    expected_tangent=tangent_form.assemble(
        reference,displacement=field
    )[order][:,order]
    actual=skfemntv.NativeAssembler.from_basis(
        native,skfemntv.NeoHookean(mu,lmbda)
    ).assemble(u,num_threads=4)
    np.testing.assert_allclose(
        actual.residual,expected_residual,rtol=3e-11,atol=3e-11
    )
    np.testing.assert_allclose(
        actual.tangent.toarray(),expected_tangent.toarray(),
        rtol=3e-11,atol=3e-11,
    )


@pytest.mark.parametrize(
    "topology",["tet10","hex27","wedge6","pyramid5"]
)
def test_neo_hookean_tangent_and_parallel_on_native_meshes(topology):
    space=native_space(topology,4,multi=True)
    assert space.mesh.nelements>1
    assembler=skfemntv.NativeAssembler.from_basis(
        space,skfemntv.NeoHookean(8.,12.)
    )
    rng=np.random.default_rng(143)
    u=np.ascontiguousarray(rng.normal(scale=1.5e-3,size=space.N))
    direction=rng.normal(size=space.N)
    serial=assembler.assemble(u,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    parallel=assembler.assemble(u,num_threads=4)
    np.testing.assert_allclose(
        parallel.residual,residual,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        parallel.tangent.data,tangent.data,rtol=3e-13,atol=3e-13
    )
    step=1e-7
    plus=assembler.assemble(
        np.ascontiguousarray(u+step*direction),mode="residual",num_threads=4
    ).residual.copy()
    minus=assembler.assemble(
        np.ascontiguousarray(u-step*direction),mode="residual",num_threads=4
    ).residual.copy()
    np.testing.assert_allclose(
        tangent@direction,(plus-minus)/(2.*step),rtol=4e-7,atol=4e-7
    )


@pytest.mark.parametrize("intorder",[2,4,6])
def test_pyramid5_neo_hookean_integration_order(intorder):
    space=native_space("pyramid5",intorder)
    assembler=skfemntv.NativeAssembler.from_basis(
        space,skfemntv.NeoHookean(6.,9.)
    )
    rng=np.random.default_rng(190+intorder)
    u=np.ascontiguousarray(rng.normal(scale=1e-3,size=space.N))
    direction=rng.normal(size=space.N)
    actual=assembler.assemble(u)
    step=1e-7
    plus=assembler.assemble(
        np.ascontiguousarray(u+step*direction),mode="residual"
    ).residual.copy()
    minus=assembler.assemble(
        np.ascontiguousarray(u-step*direction),mode="residual"
    ).residual.copy()
    np.testing.assert_allclose(
        actual.tangent@direction,(plus-minus)/(2.*step),
        rtol=4e-7,atol=4e-7,
    )


@pytest.mark.parametrize(
    "topology",["tet10","hex27","wedge6","pyramid5"]
)
def test_neo_hookean_rejects_inverted_deformation(topology):
    space=native_space(topology,2)
    assembler=skfemntv.NativeAssembler.from_basis(
        space,skfemntv.NeoHookean(5.,8.)
    )
    u=np.zeros(space.N)
    u[0::3]=-2.*space.doflocs[0,0::3]
    with pytest.raises(ValueError,match="non-positive"):
        assembler.assemble(u,num_threads=1)


@pytest.mark.parametrize(
    "topology",["tet10","hex27","wedge6","pyramid5"]
)
def test_native_basis_rejects_singular_geometry(topology):
    space=native_space(topology,2)
    points=space.mesh.p.copy();points[2]=0.
    mesh=type(space.mesh)(points,space.mesh.t)
    scalar=type(space.elem.elem)()
    with pytest.raises(
        ValueError,match="reason=near_singular_or_non_finite"
    ):
        skfemntv.Basis(mesh,skfemntv.ElementVector(scalar),intorder=2)
