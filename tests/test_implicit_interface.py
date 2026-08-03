import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import avg,ddot,dot,grad,jump,normal_grad


def _numpy_trace_block(test,trial,row_kind,column_kind):
    row=(test.shape if row_kind=="value" else test.gradients
         if row_kind=="gradient" else
         np.einsum("qnd,qd->qn",test.gradients,test.normal_vectors))
    column=(trial.shape if column_kind=="value" else trial.gradients
            if column_kind=="gradient" else
            np.einsum("qnd,qd->qn",trial.gradients,trial.normal_vectors))
    result=np.zeros((test.N,trial.N))
    for cell in test.tind:
        selection=test.cell_slice(int(cell))
        local=(
            np.einsum(
                "q,qad,qbd->ab",test.weights[selection],
                row[selection],column[selection],
            )
            if row_kind==column_kind=="gradient" else
            np.einsum(
                "q,qa,qb->ab",test.weights[selection],
                row[selection],column[selection],
            )
        )
        dofs_test=test.cell_dofs[cell,:,0]
        dofs_trial=trial.cell_dofs[cell,:,0]
        result[np.ix_(dofs_test,dofs_trial)]+=local
    return result


def _numpy_two_sided(negative,positive,row_weights,column_weights,kinds):
    sides=(negative,positive);rows=[]
    for row,test in enumerate(sides):
        blocks=[]
        for column,trial in enumerate(sides):
            blocks.append(
                row_weights[row]*column_weights[column]
                *_numpy_trace_block(test,trial,*kinds)
            )
        rows.append(np.hstack(blocks))
    return np.vstack(rows)


def test_tri3_interface_integrates_length_and_linear_field():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)

    assert rule.weights.sum()==pytest.approx(np.sqrt(.5))
    assert rule.weights@rule.points[:,0]==pytest.approx(np.sqrt(.5)*.25)
    np.testing.assert_allclose(
        rule.normals,
        np.broadcast_to(
            np.array([1.,1.])[None,:]/np.sqrt(2.),rule.normals.shape
        ),
    )
    assert np.all(rule.weights>0.)
    assert rule.side=="interface"


def test_tet4_interface_integrates_area_and_linear_field():
    mesh=skfemntv.MeshTet()
    level_set=skfemntv.LevelSet(lambda x:x.sum(axis=0)-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)
    area=np.sqrt(3.)/8.

    assert rule.weights.sum()==pytest.approx(area)
    assert rule.weights@rule.points[:,0]==pytest.approx(area/6.)
    np.testing.assert_allclose(
        rule.normals,np.ones((len(rule.weights),3))/np.sqrt(3.)
    )


def test_tet4_quadrilateral_interface_is_partitioned_without_padding():
    mesh=skfemntv.MeshTet()
    # Plane x + y = .5 cuts four tetrahedron edges and forms a quadrilateral.
    rule=skfemntv.LevelSet(
        lambda x:x[0]+x[1]-.5,tolerance=0.
    ).interface_quadrature(mesh,intorder=2)

    assert len(rule.weights)==6
    assert rule.cell_offsets.tolist()==[0,6]
    assert np.all(rule.weights>0.)
    assert not rule.points.flags.writeable


def test_implicit_facet_basis_interpolates_and_assembles_surface_forms():
    mesh=skfemntv.MeshTri()
    level_set=skfemntv.LevelSet(lambda x:x[0]+x[1]-.5,tolerance=0.)
    rule=level_set.interface_quadrature(mesh,intorder=2)
    parent=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    )
    basis=skfemntv.ImplicitFacetBasis(parent,rule)
    field=basis.interpolate(1.+2.*mesh.p[0]-mesh.p[1])
    np.testing.assert_allclose(
        field.value,1.+2.*rule.points[:,0]-rule.points[:,1]
    )

    @skfemntv.Functional
    def normal_moment(w):
        return w.x[0]*w.n[0]

    @skfemntv.BilinearForm
    def mass(u,v,w):
        return dot(u,v)

    assert skfemntv.asm(normal_moment,basis)==pytest.approx(
        rule.weights@(rule.points[:,0]*rule.normals[:,0])
    )
    matrix=skfemntv.asm(mass,basis,num_threads=2)
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T)
    assert matrix.sum()==pytest.approx(rule.weights.sum())


def test_interface_quadrature_rejects_ambiguous_and_unsupported_geometry():
    with pytest.raises(ValueError,match="not unique in cell 0"):
        skfemntv.LevelSet(np.zeros(3),tolerance=0.).interface_quadrature(
            skfemntv.MeshTri()
        )


def test_two_sided_implicit_traces_have_opposite_normals_and_block_dofs():
    mesh=skfemntv.MeshTri()
    rule=skfemntv.LevelSet(
        lambda x:x[0]+x[1]-.5,tolerance=0.
    ).interface_quadrature(mesh,intorder=2)
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    negative=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="negative"
    )
    positive=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="positive"
    )
    interface=skfemntv.ImplicitInterfacePair(negative,positive)

    np.testing.assert_allclose(
        negative.normal_vectors,-positive.normal_vectors
    )

    @skfemntv.BilinearForm
    def penalty(u,v,w):
        return 3.*dot(jump(u),jump(v))

    matrix=skfemntv.asm(
        penalty,negative,positive,integration=interface
    )
    assert all(
        type(assembler._native).__name__=="CutCrossAssembler"
        for assembler in interface._cross_cache.values()
    )
    assert matrix.shape==(negative.N+positive.N,)*2
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T)
    np.testing.assert_allclose(
        matrix.toarray(),
        3.*_numpy_two_sided(
            negative,positive,(1.,-1.),(1.,-1.),("value","value")
        ),
        atol=2.e-14,
    )
    np.testing.assert_allclose(
        matrix@np.ones(negative.N+positive.N),0.,atol=2.e-14
    )


def test_two_sided_normal_flux_and_linear_jump_are_user_composable():
    mesh=skfemntv.MeshTri()
    rule=skfemntv.LevelSet(
        lambda x:x[0]+x[1]-.5,tolerance=0.
    ).interface_quadrature(mesh,intorder=2)
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    negative=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="negative"
    )
    positive=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="positive"
    )
    interface=skfemntv.ImplicitInterfacePair(negative,positive)

    @skfemntv.BilinearForm
    def consistency(u,v,w):
        return dot(avg(normal_grad(u)),jump(v))

    @skfemntv.LinearForm
    def jump_load(v,w):
        return dot(w.load,jump(v))

    matrix=skfemntv.asm(
        consistency,negative,positive,integration=interface
    )
    vector=skfemntv.asm(
        jump_load,negative,positive,integration=interface,load=2.
    )
    assert matrix.shape==(6,6)
    assert vector.shape==(6,)
    np.testing.assert_allclose(vector[:3],-vector[3:],atol=2.e-14)
    np.testing.assert_allclose(
        matrix.toarray(),
        _numpy_two_sided(
            negative,positive,(1.,-1.),(.5,.5),
            ("value","normal_gradient"),
        ),
        atol=2.e-14,
    )
    with pytest.raises(NotImplementedError,match="Tri3 and Tet4"):
        skfemntv.LevelSet(lambda x:x[0]).interface_quadrature(
            skfemntv.MeshQuad()
        )


def test_segmented_cross_gradient_jump_matches_independent_numpy_oracle():
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,4),np.linspace(0.,1.,3)
    )
    rule=skfemntv.LevelSet(
        lambda x:x[0]-.45,tolerance=0.
    ).interface_quadrature(mesh,intorder=4)
    element=skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1)
    negative=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="negative"
    )
    positive=skfemntv.ImplicitFacetBasis(
        skfemntv.Basis(mesh,element),rule,side="positive"
    )
    interface=skfemntv.ImplicitInterfacePair(negative,positive)

    @skfemntv.BilinearForm
    def gradient_penalty(u,v,w):
        return 1.7*ddot(jump(grad(u)),jump(grad(v)))

    actual=skfemntv.asm(
        gradient_penalty,negative,positive,integration=interface
    )
    expected=1.7*_numpy_two_sided(
        negative,positive,(1.,-1.),(1.,-1.),("gradient","gradient")
    )
    np.testing.assert_allclose(actual.toarray(),expected,atol=3.e-14)
    with pytest.raises(TypeError,match="ImplicitInterfaceQuadrature"):
        skfemntv.ImplicitFacetBasis(
            skfemntv.Basis(
                skfemntv.MeshTri(),
                skfemntv.ElementVector(skfemntv.ElementTriP1(),dim=1),
            ),
            skfemntv.LevelSet(lambda x:x[0]).cut_quadrature(
                skfemntv.MeshTri()
            ),
        )


@pytest.mark.parametrize("resolution",[2,4,8])
def test_planar_interface_measure_is_stable_under_mesh_refinement(resolution):
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,resolution+1),
        np.linspace(0.,1.,resolution+1),
    )
    rule=skfemntv.LevelSet(
        lambda x:x[0]-.37,tolerance=0.
    ).interface_quadrature(mesh,intorder=4)

    assert rule.weights.sum()==pytest.approx(1.,abs=2.e-14)
    assert rule.diagnostics.nonempty_cell_count==2*resolution
