from pathlib import Path
import sys

import numpy as np
import pytest
import skfem

import skfn


sys.path.insert(0,str(Path(__file__).parents[1]/"benchmarks"))
from skfem_j2 import forms as reference_forms,update as reference_j2


def spaces(topology,distorted=False,intorder=4):
    axis=np.linspace(0.,1.,2)
    if topology=="tet10":
        linear=skfn.MeshTet.init_tensor(axis,axis,axis)
        mesh=skfn.MeshTet2.from_mesh(linear)
        element=skfn.ElementTetP2()
        reference_mesh=skfem.MeshTet2.from_mesh(
            skfem.MeshTet(linear.p,linear.t)
        )
        reference_element=skfem.ElementTetP2()
    else:
        linear=skfn.MeshHex.init_tensor(axis,axis,axis)
        mesh=skfn.MeshHex2.from_mesh(linear)
        element=skfn.ElementHex2()
        reference_mesh=skfem.MeshHex2.from_mesh(
            skfem.MeshHex(linear.p,linear.t)
        )
        reference_element=skfem.ElementHex2()
    if distorted:
        points=mesh.p.copy();x,y,z=points
        points[0]=x+.07*y*z+.025*np.sin(np.pi*y)*np.sin(np.pi*z)
        points[1]=y-.05*x*z+.02*np.sin(np.pi*x)*np.sin(np.pi*z)
        points[2]=z+.04*x*y+.015*np.sin(np.pi*x)*np.sin(np.pi*y)
        mesh=type(mesh)(points,mesh.t)
        reference_mesh=type(reference_mesh)(points,mesh.t)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(element,dim=3),intorder=intorder
    )
    reference=skfem.Basis(
        reference_mesh,skfem.ElementVector(reference_element),
        quadrature=(basis.X,basis.W),
    )
    return basis,reference


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


def strain(field):
    return np.stack((
        field.grad[0,0],field.grad[1,1],field.grad[2,2],
        .5*(field.grad[0,1]+field.grad[1,0]),
        .5*(field.grad[1,2]+field.grad[2,1]),
        .5*(field.grad[0,2]+field.grad[2,0]),
    ))


@pytest.mark.parametrize("topology",["tet10","hex27"])
def test_high_order_j2_matches_scikit_fem(topology):
    basis,reference=spaces(topology)
    material=skfn.J2Plasticity(210.,.3,.25,2.)
    assembler=skfn.MaterialAssembler(basis,material)
    state=assembler.initial_state()
    coordinates=basis.doflocs
    u=np.zeros(basis.N)
    u[0::3]=.004*coordinates[0,0::3]
    u[1::3]=-.002*coordinates[1,1::3]
    u[2::3]=-.002*coordinates[2,2::3]
    actual=assembler.assemble(u,state,num_threads=4)
    residual=actual.residual.copy();tangent=actual.tangent.copy()

    order=permutation(basis,reference)
    reference_u=np.empty(reference.N);reference_u[order]=u
    field=reference.interpolate(reference_u)
    stress,constitutive,plastic,alpha=reference_j2(
        strain(field),material.young_modulus,material.poisson_ratio,
        material.yield_stress,material.hardening_modulus,
    )
    residual_form,tangent_form=reference_forms()
    expected_residual=residual_form.assemble(reference,stress=stress)[order]
    expected_tangent=tangent_form.assemble(
        reference,constitutive=constitutive
    )[order][:,order]
    np.testing.assert_allclose(
        residual,expected_residual,rtol=8e-11,atol=8e-11
    )
    np.testing.assert_allclose(
        tangent.toarray(),expected_tangent.toarray(),rtol=8e-11,atol=8e-11
    )
    np.testing.assert_allclose(
        actual.trial_state.plastic_strain,
        np.moveaxis(plastic,0,-1).reshape(-1,6),rtol=8e-11,atol=8e-11,
    )
    np.testing.assert_allclose(
        actual.trial_state.equivalent_plastic_strain,alpha.reshape(-1),
        rtol=8e-11,atol=8e-11,
    )


@pytest.mark.parametrize("topology",["tet10","hex27"])
@pytest.mark.parametrize("material",[
    skfn.J2Plasticity(210.,.3,.25,2.),
    skfn.StandardLinearSolid(100.,60.,.25,2.,.2),
])
def test_distorted_high_order_material_tangent_and_parallel(topology,material):
    basis,_=spaces(topology,distorted=True)
    assembler=skfn.MaterialAssembler(basis,material)
    state=assembler.initial_state()
    rng=np.random.default_rng(32)
    u=np.ascontiguousarray(rng.normal(scale=8e-4,size=basis.N))
    direction=rng.normal(size=basis.N)
    serial=assembler.assemble(u,state,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    trial=serial.trial_state.storage.copy()
    parallel=assembler.assemble(u,state,num_threads=4)
    np.testing.assert_allclose(
        parallel.residual,residual,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        parallel.tangent.data,tangent.data,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        parallel.trial_state.storage,trial,rtol=3e-13,atol=3e-13
    )
    step=1e-7
    plus=assembler.assemble(
        np.ascontiguousarray(u+step*direction),state,
        mode="residual",num_threads=4,
    ).residual.copy()
    minus=assembler.assemble(
        np.ascontiguousarray(u-step*direction),state,
        mode="residual",num_threads=4,
    ).residual.copy()
    np.testing.assert_allclose(
        tangent@direction,(plus-minus)/(2.*step),rtol=8e-5,atol=8e-7
    )


@pytest.mark.parametrize("topology",["tet10","hex27"])
@pytest.mark.parametrize("intorder",[2,4,6])
def test_high_order_material_integration_order_sweep(topology,intorder):
    basis,_=spaces(topology,distorted=True,intorder=intorder)
    material=skfn.StandardLinearSolid(100.,60.,.25,2.,.2)
    assembler=skfn.MaterialAssembler(basis,material)
    state=assembler.initial_state()
    assert assembler.state_count==basis.mesh.nelements*basis.X.shape[1]
    u=np.ascontiguousarray(
        np.random.default_rng(81+intorder).normal(scale=1e-4,size=basis.N)
    )
    actual=assembler.assemble(u,state,num_threads=4)
    residual=actual.residual.copy();tangent=actual.tangent.copy()
    factor=1./(1.+material.time_step/material.relaxation_time)
    equivalent=skfn.LinearElasticity(
        material.equilibrium_modulus+factor*material.branch_modulus,
        material.poisson_ratio,
    )
    reference=skfn.NativeAssembler.from_basis(basis,equivalent).assemble(
        u,num_threads=1
    )
    np.testing.assert_allclose(
        residual,reference.residual,rtol=8e-13,atol=8e-13
    )
    np.testing.assert_allclose(
        tangent.data,reference.tangent.data,rtol=8e-13,atol=8e-13
    )
