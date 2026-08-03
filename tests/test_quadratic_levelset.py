import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import ddot,dot,grad


def _mesh():
    return skfemntv.MeshTri2.from_mesh(skfemntv.MeshTri())


def _basis(mesh):
    return skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP2(),dim=1)
    )


def test_tri6_full_domain_measure_and_parent_reference_coordinates():
    mesh=_mesh()
    quadrature=skfemntv.LevelSet(
        -np.ones(mesh.p.shape[1]),tolerance=0.
    ).cut_quadrature(mesh,intorder=2)

    assert quadrature.diagnostics.total_measure==pytest.approx(.5)
    assert len(quadrature.weights)==12
    assert np.all(quadrature.reference_points>=0.)
    assert np.all(quadrature.reference_points.sum(axis=1)<=1.)


def test_tri6_linear_interface_matches_tri3_reconstruction():
    linear=skfemntv.MeshTri()
    quadratic=skfemntv.MeshTri2.from_mesh(linear)
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.6,tolerance=0.)

    first=level_set.interface_quadrature(linear,intorder=3)
    second=level_set.interface_quadrature(quadratic,intorder=3)

    assert second.diagnostics.total_measure==pytest.approx(
        first.diagnostics.total_measure
    )
    np.testing.assert_allclose(second.weights.sum(),np.sqrt(2.)*.6)


def test_cut_cell_basis_reproduces_quadratic_value_and_gradient():
    mesh=_mesh()
    quadrature=skfemntv.LevelSet(
        lambda x:x[0]+x[1]-.7,tolerance=0.
    ).cut_quadrature(mesh,intorder=3)
    basis=_basis(mesh)
    cut=skfemntv.CutCellBasis(basis,quadrature)
    coefficients=(
        mesh.p[0]**2+2.*mesh.p[0]*mesh.p[1]+3.*mesh.p[1]**2
        +4.*mesh.p[0]-2.*mesh.p[1]+1.
    )

    field=cut.interpolate(coefficients)
    x=cut.points[:,0];y=cut.points[:,1]
    np.testing.assert_allclose(
        field.value,x*x+2.*x*y+3.*y*y+4.*x-2.*y+1.,atol=2.e-14
    )
    np.testing.assert_allclose(field.grad[0],2.*x+2.*y+4.,atol=3.e-14)
    np.testing.assert_allclose(field.grad[1],2.*x+6.*y-2.,atol=3.e-14)


def test_tri6_curved_geometry_is_rejected_with_cell_diagnostics():
    mesh=_mesh()
    mesh.p[1,mesh.t[3,0]]+=.01
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)

    with pytest.raises(NotImplementedError,match="curved Tri6.*cell 0"):
        level_set.cut_quadrature(mesh)
    with pytest.raises(NotImplementedError,match="curved Tri6.*cell 0"):
        level_set.interface_quadrature(mesh)


def test_tri6_full_domain_native_assembly_matches_regular_p2_basis():
    mesh=_mesh()
    regular=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP2(),dim=1),
        intorder=6,
    )
    cut=skfemntv.CutCellBasis(
        regular,
        skfemntv.LevelSet(-np.ones(mesh.p.shape[1]),tolerance=0.)
        .cut_quadrature(mesh,intorder=6),
    )

    @skfemntv.BilinearForm
    def mass_stiffness(u,v,w):
        return (1.+w.x[0])*dot(u,v)+ddot(grad(u),grad(v))

    expected=skfemntv.asm(mass_stiffness,regular)
    actual=skfemntv.asm(mass_stiffness,cut,num_threads=2)
    np.testing.assert_allclose(actual.toarray(),expected.toarray(),atol=3.e-14)


def test_tri6_quadratic_level_set_improves_circle_perimeter():
    exact=2.*np.pi*.7
    errors=[]
    for points in (9,17):
        linear=skfemntv.MeshTri.init_tensor(
            np.linspace(-1.,1.,points),np.linspace(-1.,1.,points)
        )
        quadratic=skfemntv.MeshTri2.from_mesh(linear)
        level_set=skfemntv.LevelSet(
            lambda x:x[0]**2+x[1]**2-.7**2,tolerance=0.
        )
        p1=level_set.interface_quadrature(linear,intorder=2)
        p2=level_set.interface_quadrature(quadratic,intorder=2)
        errors.append((
            abs(p1.diagnostics.total_measure-exact),
            abs(p2.diagnostics.total_measure-exact),
        ))

    assert errors[0][1]<errors[0][0]
    assert errors[1][1]<errors[1][0]
    assert errors[1][1]<errors[0][1]
