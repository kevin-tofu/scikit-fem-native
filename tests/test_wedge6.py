from pathlib import Path
import sys

import numpy as np
import pytest
import skfem
from skfem.models.elasticity import linear_elasticity

import skfn


sys.path.insert(0,str(Path(__file__).parents[1]/"benchmarks"))
from skfem_j2 import forms as reference_forms,update as reference_j2


def spaces(*,distorted=False,intorder=4):
    mesh=skfn.MeshWedge1.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,2),np.linspace(0.,1.,3)
    )
    if distorted:
        points=mesh.p.copy();x,y,z=points
        points[0]=x+.08*y*z
        points[1]=y-.05*x*z
        points[2]=z+.04*x*y
        mesh=skfn.MeshWedge1(points,mesh.t)
    basis=skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementWedge1()),intorder=intorder
    )
    reference=skfem.Basis(
        skfem.MeshWedge1(mesh.p,mesh.t),
        skfem.ElementVector(skfem.ElementWedge1()),
        quadrature=(basis.X,basis.W),
    )
    return basis,reference


def strain(field):
    return np.stack((
        field.grad[0,0],field.grad[1,1],field.grad[2,2],
        .5*(field.grad[0,1]+field.grad[1,0]),
        .5*(field.grad[1,2]+field.grad[2,1]),
        .5*(field.grad[0,2]+field.grad[2,0]),
    ))


@pytest.mark.parametrize("intorder",[2,4,6])
def test_wedge6_linear_elasticity_matches_scikit_fem(intorder):
    basis,reference=spaces(distorted=True,intorder=intorder)
    young=130.;poisson=.27
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    mu=young/(2.*(1.+poisson))
    expected=linear_elasticity(Lambda=lmbda,Mu=mu).assemble(reference)
    actual=skfn.NativeAssembler.from_basis(
        basis,skfn.LinearElasticity(young,poisson)
    ).assemble(np.zeros(basis.N),num_threads=4).tangent
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=5e-13,atol=5e-13
    )


def test_wedge6_j2_matches_scikit_fem():
    basis,reference=spaces(distorted=True)
    material=skfn.J2Plasticity(210.,.3,.25,2.)
    assembler=skfn.MaterialAssembler(basis,material)
    state=assembler.initial_state()
    coordinates=basis.doflocs
    u=np.zeros(basis.N)
    u[0::3]=.004*coordinates[0,0::3]
    u[1::3]=-.002*coordinates[1,1::3]
    u[2::3]=-.002*coordinates[2,2::3]
    actual=assembler.assemble(u,state,num_threads=4)

    field=reference.interpolate(u)
    stress,constitutive,plastic,alpha=reference_j2(
        strain(field),material.young_modulus,material.poisson_ratio,
        material.yield_stress,material.hardening_modulus,
    )
    residual_form,tangent_form=reference_forms()
    expected_residual=residual_form.assemble(reference,stress=stress)
    expected_tangent=tangent_form.assemble(
        reference,constitutive=constitutive
    )
    np.testing.assert_allclose(
        actual.residual,expected_residual,rtol=8e-11,atol=8e-11
    )
    np.testing.assert_allclose(
        actual.tangent.toarray(),expected_tangent.toarray(),
        rtol=8e-11,atol=8e-11,
    )
    np.testing.assert_allclose(
        actual.trial_state.plastic_strain,
        np.moveaxis(plastic,0,-1).reshape(-1,6),rtol=8e-11,atol=8e-11,
    )
    np.testing.assert_allclose(
        actual.trial_state.equivalent_plastic_strain,alpha.reshape(-1),
        rtol=8e-11,atol=8e-11,
    )


@pytest.mark.parametrize("material",[
    skfn.J2Plasticity(210.,.3,.25,2.),
    skfn.StandardLinearSolid(100.,60.,.25,2.,.2),
])
def test_wedge6_material_tangent_and_parallel(material):
    basis,_=spaces(distorted=True,intorder=4)
    assembler=skfn.MaterialAssembler(basis,material)
    state=assembler.initial_state()
    rng=np.random.default_rng(74)
    u=np.ascontiguousarray(rng.normal(scale=8e-4,size=basis.N))
    direction=rng.normal(size=basis.N)
    serial=assembler.assemble(u,state,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    parallel=assembler.assemble(u,state,num_threads=4)
    np.testing.assert_allclose(parallel.residual,residual,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(
        parallel.tangent.data,tangent.data,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        parallel.trial_state.storage,serial.trial_state.storage,
        rtol=3e-13,atol=3e-13,
    )
    step=1e-7
    plus=assembler.assemble(
        np.ascontiguousarray(u+step*direction),state,mode="residual",
        num_threads=4,
    ).residual.copy()
    minus=assembler.assemble(
        np.ascontiguousarray(u-step*direction),state,mode="residual",
        num_threads=4,
    ).residual.copy()
    np.testing.assert_allclose(
        tangent@direction,(plus-minus)/(2.*step),rtol=8e-5,atol=8e-7
    )


def test_wedge_facets_fail_explicitly_until_mixed_topology_is_supported():
    mesh=skfn.MeshWedge1()
    with pytest.raises(NotImplementedError,match="mixed triangle"):
        mesh.boundary_facets()
