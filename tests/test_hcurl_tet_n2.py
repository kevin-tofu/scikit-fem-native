"""Orientation, assembly, and evaluation checks for tetrahedral Nedelec N2."""

from itertools import permutations

import numpy as np

import skfemntv
from skfemntv.evaluation import evaluate_basis_at
from skfemntv.helpers import curl,dot


def _single_tet(order=(0,1,2,3)):
    points=np.array(((0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)))
    return skfemntv.MeshTet(points,np.asarray(order,dtype=np.int64)[:,None])


def _assemble_local(basis,local):
    from scipy.sparse import coo_matrix
    dofs=basis.element_dofs.T
    rows=np.repeat(dofs,dofs.shape[1],axis=1).ravel()
    cols=np.tile(dofs,(1,dofs.shape[1])).ravel()
    return coo_matrix(
        (local.ravel(),(rows,cols)),shape=(basis.N,basis.N),
    ).toarray()


def test_tet_n2_entity_transformations_preserve_single_cell_operators():
    reference=None
    for order in permutations(range(4)):
        basis=skfemntv.AffineTetN2Basis(_single_tet(order),intorder=6)
        mass=_assemble_local(basis,basis.element_mass_matrices()[0])
        curl_curl=_assemble_local(basis,basis.element_curl_curl_matrices()[0])
        if reference is None:
            reference=(mass,curl_curl)
        else:
            np.testing.assert_allclose(mass,reference[0],rtol=2e-13,atol=2e-13)
            np.testing.assert_allclose(curl_curl,reference[1],rtol=2e-13,atol=2e-13)


def test_tet_n2_shared_face_tangential_trace_is_continuous():
    mesh=skfemntv.MeshTet(
        np.array(((0.,1.,0.,0.,0.),(0.,0.,1.,0.,0.),(0.,0.,0.,1.,-1.))),
        np.array(((0,0),(1,2),(2,1),(3,4)),dtype=np.int64),
    )
    basis=skfemntv.AffineTetN2Basis(mesh,intorder=6)
    shared=set(basis.element_dofs[:,0]) & set(basis.element_dofs[:,1])
    assert len(shared)==8  # three edge pairs and the shared face pair
    coefficients=np.random.default_rng(24).standard_normal(basis.N)
    point=np.array((.2,.3,0.))
    values=[]
    for cell in range(2):
        vertices=mesh.p[:,mesh.t[:4,cell]]
        jacobian=vertices[:,1:]-vertices[:,:1]
        reference=np.linalg.solve(jacobian,point-vertices[:,0])
        reference_values=skfemntv._nedelec_reference.tet_n2_basis(reference[:,None])
        reference_values=np.einsum(
            "ab,biq->aiq",basis.transformations[cell],reference_values,
        )
        mapped=skfemntv._hcurl_mapping.covariant_piola(
            reference_values[:,:,None,:],basis.jacobians[:,:,cell,None],
        )[:,:,0,0]
        values.append(coefficients[basis.element_dofs[:,cell]]@mapped)
    np.testing.assert_allclose(values[0][:2],values[1][:2],atol=5e-13)


def test_tet_n2_public_basis_assembly_and_evaluation():
    mesh=skfemntv.MeshTet.init_tensor(
        np.array((0.,1.)),np.array((0.,1.)),np.array((0.,1.)),
    )
    basis=skfemntv.Basis(mesh,skfemntv.ElementTetN2(),intorder=6)

    @skfemntv.BilinearForm
    def maxwell(u,v,w):
        return dot(u,v)+.2*dot(curl(u),curl(v))

    matrix=skfemntv.asm(maxwell,basis).tocsr()
    direct=skfemntv.TetN2Assembler(basis._hcurl_basis).assemble_maxwell(
        curl_coefficient=.2,
    ).copy()
    np.testing.assert_allclose(matrix.toarray(),direct.toarray(),atol=4e-13)
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T,atol=4e-13)
    force=np.ones_like(basis._hcurl_basis.global_coordinates)

    @skfemntv.LinearForm
    def load(v,w):
        return dot(w.force,v)

    assembled_load=skfemntv.asm(load,basis,force=force)
    direct_load=skfemntv.TetN2LinearAssembler(
        basis._hcurl_basis,
    ).assemble_vector_load(force)
    np.testing.assert_allclose(assembled_load,direct_load,atol=4e-13)
    assert basis.N==(
        2*basis._hcurl_basis.dof_map.topology.edges.shape[1]
        +2*mesh.facets.shape[1]
    )
    assert len(basis.get_dofs().all())==60

    coefficients=np.linspace(-.2,.3,basis.N)
    field=basis.interpolate(coefficients)
    assert field.value.shape==(3,mesh.nelements,len(basis.W))
    assert field.curl.shape==(3,mesh.nelements,len(basis.W))
    point=evaluate_basis_at(
        basis,np.array(((.2,),(.3,),(.4,))),coefficients,
    )
    assert point.value.shape==(3,1)
    assert point.grad.shape==(3,3,1)
    assert point.curl.shape==(3,1)
    projected=basis.project(lambda x:np.stack((x[1]*0.+1.,x[2]*0.+2.,x[0]*0.-1.)))
    assert projected.shape==(basis.N,)
    assert np.all(np.isfinite(projected))
    plot_mesh,plot_values=basis.refinterp(coefficients,nrefs=1)
    assert plot_mesh.t.shape[0]==4
    assert plot_values.shape==(3*plot_mesh.p.shape[1],)


def test_tet_n2_edge_face_moment_interpolation_reproduces_affine_fields():
    mesh=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3),np.linspace(0.,1.,3),
    )
    basis=skfemntv.Basis(mesh,skfemntv.ElementTetN2(),intorder=6)
    matrix=np.array(((.2,-.4,.7),(1.1,.3,-.2),(-.6,.8,.5)))
    offset=np.array((.3,-.7,.2))

    def affine_field(points):
        return matrix@points+offset[:,None]

    coefficients=basis.interpolate_moments(affine_field)
    interpolated=basis.interpolate(coefficients)
    expected=affine_field(basis._hcurl_basis.global_coordinates.reshape(3,-1))
    expected=expected.reshape(interpolated.value.shape)
    np.testing.assert_allclose(interpolated.value,expected,atol=2e-12)
    expected_curl=np.broadcast_to(
        np.array((matrix[2,1]-matrix[1,2],
                  matrix[0,2]-matrix[2,0],
                  matrix[1,0]-matrix[0,1]))[:,None,None],
        interpolated.curl.shape,
    )
    np.testing.assert_allclose(interpolated.curl,expected_curl,atol=3e-12)
