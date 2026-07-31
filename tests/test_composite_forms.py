import numpy as np
import skfem
from skfem.helpers import div as reference_div
from skfem.helpers import ddot as reference_ddot
from skfem.helpers import dot as reference_dot
from skfem.helpers import grad as reference_grad

import skfn
from skfn.helpers import ddot,div,dot,grad


def test_element_multiplication_creates_composite_basis():
    mesh=skfn.MeshTet()
    element=skfn.ElementTetP1()*skfn.ElementTetP1()
    basis=skfn.Basis(mesh,element,intorder=2)
    assert isinstance(element,skfn.ElementComposite)
    assert len(element.elems)==2
    assert len(basis.subbases)==2
    assert basis.N==2*mesh.p.shape[1]
    np.testing.assert_array_equal(
        basis.subbases[0].nodal_dofs[0],np.arange(0,basis.N,2)
    )
    np.testing.assert_array_equal(
        basis.subbases[1].nodal_dofs[0],np.arange(1,basis.N,2)
    )


def test_composite_form_signature_and_all_scalar_blocks_match_skfem():
    mesh=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
    )
    basis=skfn.Basis(
        mesh,skfn.ElementTetP1()*skfn.ElementTetP1(),intorder=2
    )

    @skfn.BilinearForm
    def form(u1,u2,v1,v2,w):
        return (
            (1.+w.x[0])*u1*v1
            +2.*u2*v2+.3*u2*v1-.4*u1*v2
            +w.diffusion*dot(grad(u1),grad(v1))
        )

    actual=skfn.asm(form,basis,diffusion=.6)

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementTetP1()*skfem.ElementTetP1(),
        intorder=2,
    )

    @skfem.BilinearForm
    def reference(u1,u2,v1,v2,w):
        return (
            (1.+w.x[0])*u1*v1
            +2.*u2*v2+.3*u2*v1-.4*u1*v2
            +w.diffusion*reference_dot(
                reference_grad(u1),reference_grad(v1)
            )
        )

    expected=skfem.asm(reference,reference_basis,diffusion=.6)
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=4e-13,atol=4e-13
    )


def test_repeated_composite_assembly_reuses_native_blocks():
    basis=skfn.Basis(
        skfn.MeshTet(),
        skfn.ElementTetP1()*skfn.ElementTetP1(),
    )

    @skfn.BilinearForm
    def form(u1,u2,v1,v2,w):
        return w.a*u1*v1+w.b*u2*v2

    first=skfn.asm(form,basis,a=1.,b=2.)
    native=form._native_cache[basis]
    ids={key:id(value) for key,value in native._assemblers.items()}
    second=skfn.asm(form,basis,a=3.,b=4.)
    assert ids=={
        key:id(value) for key,value in native._assemblers.items()
    }
    assert first.shape==second.shape==(basis.N,basis.N)


def test_vector_scalar_divergence_blocks_match_skfem():
    mesh=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,3),
    )
    element=skfn.ElementVector(skfn.ElementTetP1())*skfn.ElementTetP1()
    basis=skfn.Basis(mesh,element,intorder=2)

    @skfn.BilinearForm
    def form(u,p,v,q,w):
        return (
            w.mu*ddot(grad(u),grad(v))
            -p*div(v)
            -q*div(u)
            +w.stabilization*p*q
        )

    actual=skfn.asm(form,basis,mu=.7,stabilization=.05)

    reference_mesh=skfem.MeshTet(mesh.p,mesh.t)
    reference_element=(
        skfem.ElementVector(skfem.ElementTetP1())*skfem.ElementTetP1()
    )
    reference_basis=skfem.Basis(
        reference_mesh,reference_element,intorder=2
    )

    @skfem.BilinearForm
    def reference(u,p,v,q,w):
        return (
            w.mu*reference_ddot(reference_grad(u),reference_grad(v))
            -p*reference_div(v)
            -q*reference_div(u)
            +w.stabilization*p*q
        )

    expected=skfem.asm(
        reference,reference_basis,mu=.7,stabilization=.05
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=5e-13,atol=5e-13
    )


def test_vector_scalar_divergence_blocks_are_transposes():
    basis=skfn.Basis(
        skfn.MeshTet(),
        skfn.ElementVector(skfn.ElementTetP1())*skfn.ElementTetP1(),
        intorder=2,
    )

    @skfn.BilinearForm
    def form(u,p,v,q,w):
        return p*div(v)+q*div(u)

    matrix=skfn.asm(form,basis).toarray()
    vector_dofs=basis.subbases[0].nodal_dofs.reshape(-1,order="F")
    scalar_dofs=basis.subbases[1].nodal_dofs.reshape(-1,order="F")
    upper=matrix[np.ix_(vector_dofs,scalar_dofs)]
    lower=matrix[np.ix_(scalar_dofs,vector_dofs)]
    np.testing.assert_allclose(upper,lower.T,rtol=0.,atol=2e-16)


def test_taylor_hood_p2_p1_form_matches_skfem():
    linear=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfn.MeshTet2.from_mesh(linear)
    element=(
        skfn.ElementVector(skfn.ElementTetP2())
        *skfn.ElementTetP1()
    )
    basis=skfn.Basis(mesh,element,intorder=4)

    @skfn.BilinearForm
    def form(u,p,v,q,w):
        return (
            w.mu*ddot(grad(u),grad(v))
            -p*div(v)-q*div(u)
        )

    actual=skfn.asm(form,basis,mu=.65)

    reference_linear=skfem.MeshTet(linear.p,linear.t)
    reference_mesh=skfem.MeshTet2.from_mesh(reference_linear)
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2())
        *skfem.ElementTetP1(),
        intorder=4,
    )
    assert basis.nodal_dofs.shape==reference_basis.nodal_dofs.shape

    @skfem.BilinearForm
    def reference(u,p,v,q,w):
        return (
            w.mu*reference_ddot(reference_grad(u),reference_grad(v))
            -p*reference_div(v)-q*reference_div(u)
        )

    expected=skfem.asm(reference,reference_basis,mu=.65)
    reference_fields=reference_basis.split_indices()
    native_fields=(
        basis.subbases[0].nodal_dofs.reshape(-1,order="F"),
        basis.subbases[1].nodal_dofs.reshape(-1,order="F"),
    )
    permutation=np.empty(basis.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        native_fields,reference_fields,(3,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference_basis.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,native_dof in enumerate(native_dofs):
            node=offset//components
            component=offset%components
            coordinate=tuple(np.round(
                basis.doflocs[:,native_dof],14
            ))
            permutation[native_dof]=lookup[coordinate][component]
    expected=expected[permutation][:,permutation]
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-12,atol=2e-12
    )


def test_taylor_hood_composite_linear_form_matches_skfem():
    linear=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfn.MeshTet2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(skfn.ElementTetP2())
        *skfn.ElementTetP1(),
        intorder=4,
    )

    @skfn.LinearForm
    def load(v,q,w):
        return (
            dot(w.force,v)
            +w.source*q
            +ddot(w.flux,grad(v))
        )

    force=np.array([1.2,-.4,.7])[:,None,None]
    flux=np.array([
        [.2,-.1,.3],
        [.4,.5,-.2],
        [-.3,.1,.6],
    ])[:,:,None,None]
    actual=skfn.asm(
        load,basis,force=force,source=.35,flux=flux
    )
    native=load._native_cache[basis]
    assembler_ids={
        field:id(assembler)
        for field,assembler in native._assemblers.items()
    }
    repeated=skfn.asm(
        load,basis,force=2.*force,source=.7,flux=2.*flux
    )
    assert assembler_ids=={
        field:id(assembler)
        for field,assembler in native._assemblers.items()
    }
    np.testing.assert_allclose(repeated,2.*actual,rtol=2e-15,atol=2e-15)

    reference_mesh=skfem.MeshTet2.from_mesh(
        skfem.MeshTet(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2())
        *skfem.ElementTetP1(),
        intorder=4,
    )

    @skfem.LinearForm
    def reference(v,q,w):
        return (
            reference_dot(w.force,v)
            +w.source*q
            +reference_ddot(w.flux,reference_grad(v))
        )

    expected=skfem.asm(
        reference,reference_basis,
        force=force,source=.35,flux=flux,
    )
    reference_fields=reference_basis.split_indices()
    native_fields=(
        basis.subbases[0].nodal_dofs.reshape(-1,order="F"),
        basis.subbases[1].nodal_dofs.reshape(-1,order="F"),
    )
    permutation=np.empty(basis.N,dtype=np.int64)
    for native_dofs,reference_dofs,components in zip(
        native_fields,reference_fields,(3,1)
    ):
        lookup={}
        for dof in reference_dofs:
            coordinate=tuple(np.round(reference_basis.doflocs[:,dof],14))
            lookup.setdefault(coordinate,[]).append(int(dof))
        for offset,native_dof in enumerate(native_dofs):
            coordinate=tuple(np.round(
                basis.doflocs[:,native_dof],14
            ))
            permutation[native_dof]=lookup[coordinate][offset%components]
    np.testing.assert_allclose(
        actual,expected[permutation],rtol=2e-12,atol=2e-12
    )


def test_taylor_hood_split_and_boundary_dofs_match_skfem():
    linear=skfn.MeshTet.init_tensor(
        np.linspace(0.,1.,3),
        np.linspace(0.,1.,2),
        np.linspace(0.,1.,2),
    )
    mesh=skfn.MeshTet2.from_mesh(linear)
    basis=skfn.Basis(
        mesh,
        skfn.ElementVector(skfn.ElementTetP2())
        *skfn.ElementTetP1(),
        intorder=4,
    )
    velocity,pressure=basis.split_bases()
    velocity_indices,pressure_indices=basis.split_indices()

    assert velocity.N==3*mesh.p.shape[1]
    assert pressure.N==len(np.unique(mesh.t[:4]))
    np.testing.assert_array_equal(
        np.sort(np.concatenate((velocity_indices,pressure_indices))),
        np.arange(basis.N),
    )

    velocity_boundary_local=velocity.get_dofs().all()
    pressure_boundary_local=pressure.get_dofs().all()
    velocity_boundary=velocity_indices[velocity_boundary_local]
    pressure_boundary=pressure_indices[pressure_boundary_local]
    composite_boundary=basis.get_dofs().all()
    np.testing.assert_array_equal(
        composite_boundary,
        np.sort(np.concatenate((
            velocity_boundary,pressure_boundary
        ))),
    )

    reference_mesh=skfem.MeshTet2.from_mesh(
        skfem.MeshTet(linear.p,linear.t)
    )
    reference_basis=skfem.Basis(
        reference_mesh,
        skfem.ElementVector(skfem.ElementTetP2())
        *skfem.ElementTetP1(),
        intorder=4,
    )
    reference_split=reference_basis.split_bases()
    assert tuple(part.N for part in (velocity,pressure))==tuple(
        part.N for part in reference_split
    )
    for native_part,reference_part in zip(
        (velocity,pressure),reference_split
    ):
        native_coordinates=np.round(
            native_part.doflocs[:,native_part.get_dofs().all()].T,14
        )
        reference_coordinates=np.round(
            reference_part.doflocs[
                :,reference_part.get_dofs().all()
            ].T,14
        )
        np.testing.assert_array_equal(
            np.unique(native_coordinates,axis=0),
            np.unique(reference_coordinates,axis=0),
        )
