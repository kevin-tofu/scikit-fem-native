import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import ddot,dot,grad


@pytest.mark.parametrize("mesh,element",[
    (skfemntv.MeshTri(),skfemntv.ElementTriP1()),
    (skfemntv.MeshTet(),skfemntv.ElementTetP1()),
])
def test_cut_basis_interpolates_affine_field_and_gradient(mesh,element):
    level_set=skfemntv.LevelSet(lambda x:x.sum(axis=0)-.6,tolerance=0.)
    quadrature=level_set.cut_quadrature(mesh,intorder=2)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(element,dim=1)
    )
    cut_basis=skfemntv.CutCellBasis(basis,quadrature)
    coefficients=2.+3.*mesh.p[0]
    if mesh.dim()>1:
        coefficients=coefficients-4.*mesh.p[1]
    field=cut_basis.interpolate(coefficients)

    expected=2.+3.*quadrature.points[:,0]
    gradient=np.zeros(mesh.dim());gradient[0]=3.
    if mesh.dim()>1:
        expected-=4.*quadrature.points[:,1]
        gradient[1]=-4.
    np.testing.assert_allclose(field.value,expected,atol=1.e-14)
    np.testing.assert_allclose(
        field.grad,np.broadcast_to(gradient[:,None],field.grad.shape),
        atol=1.e-14,
    )
    assert cut_basis.integrate(field.value)==pytest.approx(
        quadrature.weights@expected
    )


def test_cut_basis_preserves_csr_cells_and_global_dofs():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(-1.,1.,5),np.linspace(0.,1.,3)
    )
    quadrature=skfemntv.LevelSet(lambda x:x[0]-.1).cut_quadrature(mesh)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=2)
    )
    cut_basis=skfemntv.CutCellBasis(basis,quadrature)

    assert cut_basis.npoints==len(quadrature.weights)
    assert cut_basis.nelems==np.count_nonzero(np.diff(quadrature.cell_offsets))
    for point,cell in enumerate(quadrature.cells):
        np.testing.assert_array_equal(
            cut_basis.quadrature_dofs[point],basis.element_dofs[:,cell]
        )
    assert not cut_basis.shape.flags.writeable
    assert not cut_basis.gradients.flags.writeable
    assert not cut_basis.quadrature_dofs.flags.writeable


def test_cut_basis_supports_restricted_parent_basis_when_cells_are_present():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(-1.,1.,4),np.linspace(0.,1.,2)
    )
    level_set=skfemntv.LevelSet(lambda x:x[0]-.2)
    quadrature=level_set.cut_quadrature(mesh)
    active=level_set.classify(mesh).active
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1),
        elements=active,
    )

    cut_basis=skfemntv.CutCellBasis(basis,quadrature)
    np.testing.assert_array_equal(cut_basis.tind,active)


def test_cut_basis_rejects_unsupported_elements_and_bad_values():
    mesh=skfemntv.MeshTri()
    quadrature=skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(mesh)
    with pytest.raises(NotImplementedError,match="TriP1 and TetP1"):
        skfemntv.CutCellBasis(
            skfemntv.Basis(
                skfemntv.MeshTri2(),
                skfemntv.ElementVector(skfemntv.ElementTriP2(),dim=1),
            ),
            skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(mesh),
        )
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    cut_basis=skfemntv.CutCellBasis(basis,quadrature)
    with pytest.raises(ValueError,match="coefficients"):
        cut_basis.interpolate(np.ones(basis.N+1))
    with pytest.raises(ValueError,match="integrand"):
        cut_basis.integrate(np.ones(cut_basis.npoints+1))


def test_cut_basis_assembles_functional_linear_and_bilinear_forms_natively():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    quadrature=level_set.cut_quadrature(mesh,intorder=2)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    cut_basis=skfemntv.CutCellBasis(basis,quadrature)

    @skfemntv.Functional
    def first_moment(w):
        return w.x[0]+2.*w.x[1]

    @skfemntv.LinearForm
    def source(v,w):
        return dot(w.force,v)

    @skfemntv.BilinearForm
    def reaction_diffusion(u,v,w):
        return (1.+w.x[1])*dot(u,v)+ddot(grad(u),grad(v))

    assert skfemntv.asm(first_moment,cut_basis)==pytest.approx(
        quadrature.weights@(quadrature.points[:,0]+2.*quadrature.points[:,1])
    )
    vector=skfemntv.asm(source,cut_basis,force=np.array([1.]),num_threads=2)
    matrix=skfemntv.asm(reaction_diffusion,cut_basis,num_threads=2)
    assert type(source._native_cache[cut_basis]._native).__name__==(
        "CutLinearFormAssembler"
    )
    assert type(reaction_diffusion._native_cache[cut_basis]._native).__name__==(
        "CutBilinearFormAssembler"
    )
    assert vector.shape==(basis.N,)
    assert matrix.shape==(basis.N,basis.N)
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T)
    assert np.all(np.linalg.eigvalsh(matrix.toarray())>0.)


def test_full_domain_cut_assembly_matches_regular_basis():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    regular=skfemntv.Basis(mesh,element,intorder=2)
    cut=skfemntv.CutCellBasis(
        regular,
        skfemntv.LevelSet(-np.ones(mesh.p.shape[1]),tolerance=0.)
        .cut_quadrature(mesh,intorder=2),
    )

    @skfemntv.LinearForm
    def load(v,w):
        return dot(w.force,v)

    @skfemntv.BilinearForm
    def mass_stiffness(u,v,w):
        return (1.+w.x[0])*dot(u,v)+ddot(grad(u),grad(v))

    np.testing.assert_allclose(
        skfemntv.asm(load,cut,force=np.array([1.5])),
        skfemntv.asm(load,regular,force=np.array([1.5])),atol=2.e-15
    )
    np.testing.assert_allclose(
        skfemntv.asm(mass_stiffness,cut).toarray(),
        skfemntv.asm(mass_stiffness,regular).toarray(),atol=2.e-14,
    )


def test_cut_assembly_serial_parallel_agree():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(-1.,1.,12),np.linspace(-1.,1.,11)
    )
    regular=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    cut=skfemntv.CutCellBasis(
        regular,skfemntv.LevelSet(
            lambda x:x[0]**2+x[1]**2-.7**2
        ).cut_quadrature(mesh,intorder=2),
    )

    @skfemntv.BilinearForm
    def form(u,v,w):
        return dot(u,v)+ddot(grad(u),grad(v))

    serial=skfemntv.asm(form,cut,num_threads=1)
    parallel=skfemntv.asm(form,cut,num_threads=4)
    np.testing.assert_allclose(parallel.toarray(),serial.toarray(),atol=1.e-14)
