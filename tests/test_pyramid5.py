import numpy as np
import pytest

import skfn
from skfn.helpers import dot


def basis(*,distorted=False,intorder=4):
    mesh=skfn.MeshPyramid1()
    if distorted:
        points=mesh.p.copy()
        points[:,4]+=(.07,-.04,.08)
        points[:,1]+=(.03,.02,0.)
        mesh=skfn.MeshPyramid1(points,mesh.t)
    return skfn.Basis(
        mesh,skfn.ElementVector(skfn.ElementPyramid1()),intorder=intorder
    )


@pytest.mark.parametrize("intorder",[2,4,6])
def test_pyramid5_reference_volume_and_partition_of_unity(intorder):
    space=basis(intorder=intorder)
    np.testing.assert_allclose(space.dx.sum(),1./3.,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(
        space.tabulated_shape.sum(axis=2),1.,rtol=2e-14,atol=2e-14
    )
    np.testing.assert_allclose(
        space.tabulated_gradients.sum(axis=2),0.,rtol=2e-13,atol=2e-13
    )


@pytest.mark.parametrize("distorted",[False,True])
def test_pyramid5_constant_strain_patch(distorted):
    space=basis(distorted=distorted,intorder=6)
    young=120.;poisson=.26
    assembler=skfn.NativeAssembler.from_basis(
        space,skfn.LinearElasticity(young,poisson)
    )
    matrix=np.array([[.013,.002,-.001],[.004,-.007,.003],[.001,.005,.009]])
    displacement=np.empty(space.N)
    for component in range(3):
        displacement[component::3]=(
            matrix[component]@space.doflocs[:,component::3]
        )
    tangent=assembler.assemble(np.zeros(space.N)).tangent
    strain=.5*(matrix+matrix.T)
    lmbda=young*poisson/((1.+poisson)*(1.-2.*poisson))
    mu=young/(2.*(1.+poisson))
    stress=lmbda*np.trace(strain)*np.eye(3)+2.*mu*strain
    expected=np.sum(strain*stress)*space.dx.sum()
    np.testing.assert_allclose(
        displacement@(tangent@displacement),expected,
        rtol=8e-13,atol=8e-13,
    )


@pytest.mark.parametrize("material",[
    skfn.J2Plasticity(210.,.3,.25,2.),
    skfn.StandardLinearSolid(100.,60.,.25,2.,.2),
])
def test_pyramid5_material_tangent_and_parallel(material):
    space=basis(distorted=True,intorder=4)
    assembler=skfn.MaterialAssembler(space,material)
    state=assembler.initial_state()
    rng=np.random.default_rng(95)
    u=np.ascontiguousarray(rng.normal(scale=8e-4,size=space.N))
    direction=rng.normal(size=space.N)
    serial=assembler.assemble(u,state,num_threads=1)
    residual=serial.residual.copy();tangent=serial.tangent.copy()
    trial=serial.trial_state.storage.copy()
    parallel=assembler.assemble(u,state,num_threads=4)
    np.testing.assert_allclose(parallel.residual,residual,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(
        parallel.tangent.data,tangent.data,rtol=3e-13,atol=3e-13
    )
    np.testing.assert_allclose(
        parallel.trial_state.storage,trial,rtol=3e-13,atol=3e-13
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


def test_pyramid_mixed_facets_area_normals_and_functional():
    mesh=skfn.MeshPyramid1()
    facets=skfn.FacetBasis(
        mesh,skfn.ElementVector(skfn.ElementPyramid1()),intorder=6
    )
    np.testing.assert_allclose(
        facets.dx.sum(),1.+np.sqrt(5.),rtol=2e-14,atol=2e-14
    )
    np.testing.assert_allclose(
        np.sum(facets.normals*facets.dx[:,:,None],axis=(0,1)),0.,
        atol=3e-14,
    )
    assert sorted(mesh._facet_sizes.tolist())==[3,3,3,3,4]

    @skfn.Functional
    def moment(w):
        return 1.+w.x[1]+w.n[0]**2

    value=skfn.asm(moment,facets)
    direct=np.sum(
        (1.+facets.global_coordinates[:,:,1]+facets.normals[:,:,0]**2)
        *facets.dx
    )
    np.testing.assert_allclose(value,direct,rtol=2e-14,atol=2e-14)

    @skfn.LinearForm
    def normal_load(v,w):
        return dot(w.n,v)

    resultant=np.array([
        skfn.asm(normal_load,facets)[component::3].sum()
        for component in range(3)
    ])
    np.testing.assert_allclose(resultant,0.,atol=3e-14)
