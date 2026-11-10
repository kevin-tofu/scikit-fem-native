"""Orientation, assembly, and interpolation checks for tetrahedral Nedelec N3."""

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


def test_tet_n3_entity_transforms_preserve_trace_operator_for_all_vertex_orders():
    reference=None
    for order in permutations(range(4)):
        basis=skfemntv.AffineTetN3Basis(_single_tet(order),intorder=6)
        local=(basis.element_mass_matrices()[0]
               +.1*basis.element_curl_curl_matrices()[0])
        operator=_assemble_local(basis,local)
        trace_dofs=42
        operator=(operator[:trace_dofs,:trace_dofs]
                  -operator[:trace_dofs,trace_dofs:]
                  @np.linalg.solve(
                      operator[trace_dofs:,trace_dofs:],
                      operator[trace_dofs:,:trace_dofs],
                  ))
        if reference is None:
            reference=operator
        else:
            np.testing.assert_allclose(operator,reference,rtol=4e-12,atol=4e-12)


def test_tet_n3_shared_face_tangential_trace_is_continuous():
    mesh=skfemntv.MeshTet(
        np.array(((0.,1.,0.,0.,0.),(0.,0.,1.,0.,0.),(0.,0.,0.,1.,-1.))),
        np.array(((0,0),(1,2),(2,1),(3,4)),dtype=np.int64),
    )
    basis=skfemntv.AffineTetN3Basis(mesh,intorder=6)
    assert len(set(basis.element_dofs[:,0])&set(basis.element_dofs[:,1]))==15
    coefficients=np.random.default_rng(35).standard_normal(basis.N)
    point=np.array((.2,.3,0.))
    values=[]
    for cell in range(2):
        vertices=mesh.p[:,mesh.t[:4,cell]]
        jacobian=vertices[:,1:]-vertices[:,:1]
        reference=np.linalg.solve(jacobian,point-vertices[:,0])
        reference_values=skfemntv._nedelec_reference.tet_n3_basis(reference[:,None])
        reference_values=np.einsum(
            "ab,biq->aiq",basis.transformations[cell],reference_values,
        )
        mapped=skfemntv._hcurl_mapping.covariant_piola(
            reference_values[:,:,None,:],basis.jacobians[:,:,cell,None],
        )[:,:,0,0]
        values.append(coefficients[basis.element_dofs[:,cell]]@mapped)
    np.testing.assert_allclose(values[0][:2],values[1][:2],atol=2e-12)


def test_tet_n3_public_assembly_moment_interpolation_and_evaluation():
    mesh=skfemntv.MeshTet.init_tensor(
        np.array((0.,1.)),np.array((0.,1.)),np.array((0.,1.)),
    )
    basis=skfemntv.Basis(mesh,skfemntv.ElementTetN3(),intorder=8)

    @skfemntv.BilinearForm
    def maxwell(u,v,w):
        return dot(u,v)+.1*dot(curl(u),curl(v))

    matrix=skfemntv.asm(maxwell,basis).tocsr()
    direct=skfemntv.TetN3Assembler(basis._hcurl_basis).assemble_maxwell(
        curl_coefficient=.1,
    ).copy()
    np.testing.assert_allclose(matrix.toarray(),direct.toarray(),atol=3e-12)
    np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T,atol=3e-12)

    linear=np.array(((.2,-.4,.7),(1.1,.3,-.2),(-.6,.8,.5)))
    offset=np.array((.3,-.7,.2))
    field=lambda points:linear@points+offset[:,None]
    coefficients=basis.interpolate_moments(field)
    numerical=basis.interpolate(coefficients)
    exact=field(basis._hcurl_basis.global_coordinates.reshape(3,-1))
    np.testing.assert_allclose(
        numerical.value,exact.reshape(numerical.value.shape),atol=3e-12,
    )
    point=evaluate_basis_at(
        basis,np.array(((.2,),(.3,),(.4,))),coefficients,
    )
    assert point.value.shape==(3,1)
    assert point.grad.shape==(3,3,1)
    assert point.curl.shape==(3,1)
    plot_mesh,plot_values=basis.refinterp(coefficients,nrefs=1)
    assert plot_values.shape==(3*plot_mesh.p.shape[1],)


def test_tet_n3_assembly_memory_preflight():
    mesh=skfemntv.MeshTet.init_tensor(
        np.linspace(0.,1.,3),np.linspace(0.,1.,3),np.linspace(0.,1.,3),
    )
    basis=skfemntv.AffineTetN3Basis(mesh,intorder=6)
    estimate=skfemntv.estimate_tet_n3_assembly_memory(basis)
    assert estimate.kind=="hcurl_tet_n3"
    assert estimate.row_local_dofs==estimate.column_local_dofs==45
    assert estimate.entity_count==mesh.nelements
    assert estimate.basis_bytes==sum(
        array.nbytes for array in (
            basis._element_values,basis._element_curls,basis.dx,
            basis.element_dofs,
        )
    )
    required=int(np.ceil(estimate.construction_peak_total_bytes_upper_bound*1.25))
    skfemntv.TetN3Assembler(
        basis,memory_limit_bytes=required,memory_safety_factor=1.25,
    )
    try:
        skfemntv.TetN3Assembler(
            basis,memory_limit_bytes=required-1,memory_safety_factor=1.25,
        )
    except skfemntv.AssemblyMemoryBudgetError as error:
        assert error.required_bytes==required
    else:
        raise AssertionError("an insufficient TetN3 memory budget was accepted")
