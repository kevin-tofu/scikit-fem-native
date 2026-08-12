import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv


def _distorted_mesh(divisions,*,mixed_orientation=False):
    mesh=skfemntv.MeshTri.init_tensor(
        np.linspace(0.,1.,divisions+1),np.linspace(0.,1.,divisions+1)
    )
    points=mesh.p.copy()
    h=1./divisions
    interior=(
        (points[0]>0.)&(points[0]<1.)
        &(points[1]>0.)&(points[1]<1.)
    )
    x=points[0,interior].copy()
    y=points[1,interior].copy()
    points[0,interior]+=.18*h*np.sin(2.*np.pi*y)*np.sin(np.pi*x)
    points[1,interior]+=.14*h*np.sin(2.*np.pi*x)*np.sin(np.pi*y)
    connectivity=mesh.t.copy()
    vertices=points[:,connectivity[:3]]
    first=vertices[:,1]-vertices[:,0]
    second=vertices[:,2]-vertices[:,0]
    negative=first[0]*second[1]-first[1]*second[0]<0.
    temporary=connectivity[0,negative].copy()
    connectivity[0,negative]=connectivity[1,negative]
    connectivity[1,negative]=temporary
    if mixed_orientation:
        selected=np.arange(connectivity.shape[1])%3==1
        temporary=connectivity[0,selected].copy()
        connectivity[0,selected]=connectivity[1,selected]
        connectivity[1,selected]=temporary
    result=skfemntv.MeshTri(points,connectivity)
    # MeshTri construction applies its own canonical cell ordering.  Restore
    # the deliberate orientation fixture after construction so this test owns
    # the exact signed-connectivity pattern under examination.
    result.t=np.ascontiguousarray(connectivity)
    return result


def _exact(points):
    x,y=points
    return np.array((np.sin(np.pi*y),np.sin(np.pi*x)))


def _exact_curl(points):
    x,y=points
    return np.pi*(np.cos(np.pi*x)-np.cos(np.pi*y))


def _solve(mesh,beta=.2):
    basis=skfemntv.AffineTriN1Basis(mesh,intorder=5,max_aspect_ratio=4.)
    matrix=skfemntv.TriN1Assembler(basis).assemble_maxwell(
        mass_coefficient=1.,curl_coefficient=beta
    ).copy()
    rhs=skfemntv.TriN1LinearAssembler(basis).assemble_vector_load(
        lambda x:(1.+beta*np.pi**2)*_exact(x)
    ).copy()
    boundary=basis.boundary_dofs()
    free=np.setdiff1d(np.arange(basis.N),boundary)
    solution=np.zeros(basis.N)
    solution[free]=spsolve(matrix[free][:,free],rhs[free])
    points=basis.global_coordinates
    value_error=basis.evaluate(solution)-_exact(points)
    curl_error=basis.evaluate_curl(solution)-_exact_curl(points)
    l2=np.sqrt(np.einsum("ieq,ieq,eq->",value_error,value_error,basis.dx))
    curl_l2=np.sqrt(np.einsum("eq,eq,eq->",curl_error,curl_error,basis.dx))
    return basis,matrix,rhs,solution,np.array((l2,curl_l2,np.hypot(l2,curl_l2)))


def _by_edge(basis,array):
    lookup={tuple(edge):i for i,edge in enumerate(basis.dof_map.topology.edges.T)}
    order=np.array([lookup[edge] for edge in sorted(lookup)])
    if array.ndim==1:
        return array[order]
    return array[np.ix_(order,order)]


def test_geometry_diagnostics_report_distortion_and_mixed_orientation():
    regular=skfemntv.AffineTriN1Basis(_distorted_mesh(4))
    mixed=skfemntv.AffineTriN1Basis(
        _distorted_mesh(4,mixed_orientation=True)
    )
    diagnostics=regular.geometry_diagnostics
    assert diagnostics.minimum_area>0.
    assert diagnostics.minimum_absolute_determinant>0.
    assert diagnostics.maximum_aspect_ratio>1.
    assert diagnostics.inverted_cell_count==0
    assert mixed.geometry_diagnostics.inverted_cell_count>0
    assert mixed.geometry_diagnostics.minimum_signed_determinant<0.


def test_optional_aspect_limit_rejects_excessive_distortion():
    mesh=_distorted_mesh(4)
    measured=skfemntv.AffineTriN1Basis(mesh).geometry_diagnostics.maximum_aspect_ratio
    with np.testing.assert_raises_regex(ValueError,"aspect ratio exceeds"):
        skfemntv.AffineTriN1Basis(mesh,max_aspect_ratio=.99*measured)
    with np.testing.assert_raises_regex(ValueError,"positive finite"):
        skfemntv.AffineTriN1Basis(mesh,max_aspect_ratio=np.inf)


def test_distorted_mesh_manufactured_solution_converges():
    history=np.array([_solve(_distorted_mesh(n))[-1] for n in (4,8,16)])
    assert np.all(history[1:]<history[:-1])
    rates=np.log2(history[:-1]/history[1:])
    assert np.all(rates>.75),rates


def test_mixed_cell_orientation_preserves_operator_load_solution_and_error():
    forward=_solve(_distorted_mesh(8))
    mixed=_solve(_distorted_mesh(8,mixed_orientation=True))
    forward_basis,forward_matrix,forward_rhs,forward_solution,forward_error=forward
    mixed_basis,mixed_matrix,mixed_rhs,mixed_solution,mixed_error=mixed
    np.testing.assert_allclose(
        _by_edge(forward_basis,forward_matrix.toarray()),
        _by_edge(mixed_basis,mixed_matrix.toarray()),atol=8e-14,
    )
    np.testing.assert_allclose(
        _by_edge(forward_basis,forward_rhs),
        _by_edge(mixed_basis,mixed_rhs),atol=3e-14,
    )
    np.testing.assert_allclose(
        _by_edge(forward_basis,forward_solution),
        _by_edge(mixed_basis,mixed_solution),atol=2e-13,
    )
    np.testing.assert_allclose(forward_error,mixed_error,atol=2e-13)
