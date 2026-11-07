"""Physical-point evaluation for affine simplex H1 bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PointEvaluation:
    """Field values, physical gradients, and point-location diagnostics."""

    value: np.ndarray
    grad: np.ndarray
    cell_ids: np.ndarray
    reference_coordinates: np.ndarray
    on_boundary: np.ndarray
    ambiguous: np.ndarray
    outside: np.ndarray
    curl: np.ndarray | None = None
    local_basis_values: np.ndarray | None = None


def _shape_and_gradient(basis, reference):
    """Evaluate one local H1 shape at a reference simplex point."""
    base = basis._base_element
    dim = basis.mesh.dim()
    if basis._constant:
        return np.ones(1), np.zeros((1, dim))

    if dim == 2:
        xi, eta = reference
        bary = np.array((1.0 - xi - eta, xi, eta))
        bary_gradient = np.array(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
        if not basis._quadratic_tri:
            return bary, bary_gradient
        pairs = ((0, 1), (1, 2), (0, 2))
        shape = [value * (2.0 * value - 1.0) for value in bary]
        gradient = [
            (4.0 * value - 1.0) * bary_gradient[i]
            for i, value in enumerate(bary)
        ]
        for i, j in pairs:
            shape.append(4.0 * bary[i] * bary[j])
            gradient.append(
                4.0 * (bary[j] * bary_gradient[i] + bary[i] * bary_gradient[j])
            )
        return np.asarray(shape), np.asarray(gradient)

    xi, eta, zeta = reference
    bary = np.array((1.0 - xi - eta - zeta, xi, eta, zeta))
    bary_gradient = np.array(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0),
         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    if not basis._quadratic_tet:
        return bary, bary_gradient
    pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    shape = [value * (2.0 * value - 1.0) for value in bary]
    gradient = [
        (4.0 * value - 1.0) * bary_gradient[i]
        for i, value in enumerate(bary)
    ]
    for i, j in pairs:
        shape.append(4.0 * bary[i] * bary[j])
        gradient.append(
            4.0 * (bary[j] * bary_gradient[i] + bary[i] * bary_gradient[j])
        )
    return np.asarray(shape), np.asarray(gradient)


def evaluate_basis_at(basis, points, coefficients, *, outside="error"):
    """Evaluate an affine triangle/tetrahedron H1 field at physical points."""
    if outside not in ("error", "nan"):
        raise ValueError("outside must be 'error' or 'nan'")
    if getattr(basis,"_hcurl_basis",None) is not None:
        return _evaluate_hcurl_basis_at(
            basis,points,coefficients,outside=outside
        )
    if hasattr(basis, "subbases"):
        raise NotImplementedError("point evaluation of composite bases is not implemented")
    if not (basis._tri or basis._tet):
        raise NotImplementedError(
            "point evaluation currently supports triangle and tetrahedron bases"
        )
    dim = basis.mesh.dim()
    if not (
        basis._constant
        or (basis._tri and not basis._quadratic_tri)
        or basis._quadratic_tri
        or (basis._tet and not basis._quadratic_tet)
        or basis._quadratic_tet
    ):
        raise NotImplementedError("point evaluation supports simplex P0, P1, and P2")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] != dim:
        raise ValueError(f"points must have shape ({dim}, npoints)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (basis.N,):
        raise ValueError(f"coefficients must have shape ({basis.N},)")

    cell_ids = np.full(points.shape[1], -1, dtype=np.int64)
    references = np.full((dim, points.shape[1]), np.nan)
    boundary = np.zeros(points.shape[1], dtype=bool)
    ambiguous = np.zeros(points.shape[1], dtype=bool)
    outside_mask = np.zeros(points.shape[1], dtype=bool)
    components = basis.elem._dim
    local_nodes = basis.element_dofs.shape[0] // components
    value = np.full((components, points.shape[1]), np.nan)
    gradient = np.full((components, dim, points.shape[1]), np.nan)

    mesh_cells = basis._mesh.t[: dim + 1, basis.tind]
    vertices = basis._mesh.p[:, mesh_cells]
    geometry_cells = basis._mesh.t[:, basis.tind]
    centers = vertices.mean(axis=1)
    radii = np.linalg.norm(vertices - centers[:, None, :], axis=0).max(axis=0)
    scale = float(np.max(np.ptp(vertices.reshape(dim, -1), axis=1)))
    coordinate_scale = max(scale, float(np.max(np.abs(vertices))))
    physical_tolerance = 64.0 * np.finfo(float).eps * coordinate_scale
    affine_geometry = geometry_cells.shape[0] == dim + 1
    if dim == 2 and geometry_cells.shape[0] == 6:
        pairs = ((0, 1), (1, 2), (0, 2))
        expected = np.stack([
            0.5 * (vertices[:, i] + vertices[:, j]) for i, j in pairs
        ], axis=1)
        affine_geometry = np.all(
            np.abs(basis._mesh.p[:, geometry_cells[3:6]] - expected)
            <= physical_tolerance
        )
    elif dim == 3 and geometry_cells.shape[0] == 10:
        pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        expected = np.stack([
            0.5 * (vertices[:, i] + vertices[:, j]) for i, j in pairs
        ], axis=1)
        affine_geometry = np.all(
            np.abs(basis._mesh.p[:, geometry_cells[4:10]] - expected)
            <= physical_tolerance
        )
    if not affine_geometry:
        raise NotImplementedError(
            "point evaluation currently requires affine simplex geometry"
        )
    tree = cKDTree(centers.T)
    reference_tolerance = 1.0e-10
    max_radius = float(radii.max(initial=0.0)) + physical_tolerance

    for point_id, point in enumerate(points.T):
        candidates = tree.query_ball_point(point, max_radius)
        hits = []
        for local_cell in candidates:
            cell_vertices = vertices[:, :, local_cell]
            if np.any(point < cell_vertices.min(axis=1) - physical_tolerance):
                continue
            if np.any(point > cell_vertices.max(axis=1) + physical_tolerance):
                continue
            jacobian = cell_vertices[:, 1:] - cell_vertices[:, :1]
            reference = np.linalg.solve(jacobian, point - cell_vertices[:, 0])
            bary = np.concatenate(([1.0 - reference.sum()], reference))
            if np.min(bary) < -reference_tolerance or np.max(bary) > 1.0 + reference_tolerance:
                continue
            hits.append((int(basis.tind[local_cell]), local_cell, reference, bary))

        if not hits:
            outside_mask[point_id] = True
            continue
        hits.sort(key=lambda hit: hit[0])
        global_cell, local_cell, reference, bary = hits[0]
        cell_ids[point_id] = global_cell
        references[:, point_id] = reference
        boundary[point_id] = bool(np.any(np.abs(bary) <= reference_tolerance))
        ambiguous[point_id] = len(hits) > 1

        shape, reference_gradient = _shape_and_gradient(basis, reference)
        if len(shape) != local_nodes:
            raise RuntimeError("element shape and DOF counts do not match")
        jacobian = vertices[:, 1:, local_cell] - vertices[:, :1, local_cell]
        physical_gradient = reference_gradient @ np.linalg.inv(jacobian)
        local_coefficients = coefficients[
            basis.element_dofs[:, local_cell]
        ].reshape(local_nodes, components)
        value[:, point_id] = shape @ local_coefficients
        gradient[:, :, point_id] = np.einsum(
            "nd,nc->cd", physical_gradient, local_coefficients
        )

    if np.any(outside_mask) and outside == "error":
        point_id = int(np.flatnonzero(outside_mask)[0])
        raise ValueError(
            f"point {point_id} {points[:, point_id].tolist()} lies outside the basis mesh"
        )
    if components == 1:
        value = value[0]
        gradient = gradient[0]
    return PointEvaluation(
        value=value,
        grad=gradient,
        cell_ids=cell_ids,
        reference_coordinates=references,
        on_boundary=boundary,
        ambiguous=ambiguous,
        outside=outside_mask,
    )


def _evaluate_hcurl_basis_at(basis,points,coefficients,*,outside):
    from ._nedelec_reference import (
        TRI_DIRECTED_EDGES,TET_DIRECTED_EDGES,tri_n1_basis,tri_n1_curl,
        tet_n1_basis,tet_n1_curl,tri_n2_basis,tri_n2_curl,
        tri_n2_gradients,tri_n3_basis,tri_n3_curl,tri_n3_gradients,
        tet_n2_basis,tet_n2_curl,tet_n2_gradients,
        tet_n3_basis,tet_n3_curl,tet_n3_gradients,
    )
    if outside not in ("error","nan"):
        raise ValueError("outside must be 'error' or 'nan'")
    hcurl=basis._hcurl_basis
    global_sign=getattr(basis,"_hcurl_global_sign",1.)
    mesh=basis._mesh
    dim=mesh.dim()
    points=np.asarray(points,dtype=np.float64)
    if points.ndim==1:
        points=points[:,None]
    if points.ndim!=2 or points.shape[0]!=dim:
        raise ValueError(f"points must have shape ({dim}, npoints)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")
    coefficients=np.asarray(coefficients,dtype=np.float64)
    if coefficients.shape!=(basis.N,):
        raise ValueError(f"coefficients must have shape ({basis.N},)")
    cells=mesh.t[:dim+1,basis.tind]
    vertices=mesh.p[:,cells]
    centers=vertices.mean(axis=1)
    radii=np.linalg.norm(vertices-centers[:,None,:],axis=0).max(axis=0)
    scale=max(float(np.max(np.ptp(vertices.reshape(dim,-1),axis=1))),
              float(np.max(np.abs(vertices))))
    tolerance=64.*np.finfo(float).eps*scale
    tree=cKDTree(centers.T)
    cell_ids=np.full(points.shape[1],-1,dtype=np.int64)
    references=np.full((dim,points.shape[1]),np.nan)
    boundary=np.zeros(points.shape[1],dtype=bool)
    ambiguous=np.zeros(points.shape[1],dtype=bool)
    outside_mask=np.zeros(points.shape[1],dtype=bool)
    values=np.full((dim,points.shape[1]),np.nan)
    gradients=np.full((dim,dim,points.shape[1]),np.nan)
    curls=np.full((dim if dim==3 else 1,points.shape[1]),np.nan)
    local_count=basis.element_dofs.shape[0]
    local_basis_values=np.full((points.shape[1],dim,local_count),np.nan)
    local_edges=TRI_DIRECTED_EDGES if dim==2 else TET_DIRECTED_EDGES
    bary_grad=np.vstack((-np.ones(dim),np.eye(dim)))
    from .basis import ElementTetN2,ElementTetN3,ElementTriN2,ElementTriN3
    tet_n3=isinstance(basis.elem,ElementTetN3)
    tet_n2=isinstance(basis.elem,ElementTetN2)
    tri_n2=isinstance(basis.elem,ElementTriN2)
    tri_n3=isinstance(basis.elem,ElementTriN3)
    for point_id,point in enumerate(points.T):
        hits=[]
        candidates=tree.query_ball_point(point,float(radii.max(initial=0.))+tolerance)
        for local_cell in candidates:
            xyz=vertices[:,:,local_cell]
            if np.any(point<xyz.min(axis=1)-tolerance) or np.any(point>xyz.max(axis=1)+tolerance):
                continue
            jacobian=xyz[:,1:]-xyz[:,:1]
            reference=np.linalg.solve(jacobian,point-xyz[:,0])
            bary=np.r_[1.-reference.sum(),reference]
            if np.min(bary)<-1.e-10 or np.max(bary)>1.+1.e-10:
                continue
            hits.append((int(basis.tind[local_cell]),local_cell,reference,bary,jacobian))
        if not hits:
            outside_mask[point_id]=True
            continue
        hits.sort(key=lambda item:item[0])
        global_cell,local_cell,reference,bary,jacobian=hits[0]
        cell_ids[point_id]=global_cell
        references[:,point_id]=reference
        boundary[point_id]=bool(np.any(np.abs(bary)<=1.e-10))
        ambiguous[point_id]=len(hits)>1
        inverse=np.linalg.inv(jacobian)
        ref_values=(
            tet_n3_basis(reference[:,None])[:,:,0] if tet_n3 else
            tet_n2_basis(reference[:,None])[:,:,0] if tet_n2 else
            tri_n3_basis(reference[:,None])[:,:,0]
            if tri_n3 else
            tri_n2_basis(reference[:,None])[:,:,0] if tri_n2 else
            (tri_n1_basis(reference[:,None])[:,:,0] if dim==2 else
             tet_n1_basis(reference[:,None])[:,:,0])
        )
        ref_curls=(
            tet_n3_curl(reference[:,None])[:,:,0] if tet_n3 else
            tet_n2_curl(reference[:,None])[:,:,0] if tet_n2 else
            tri_n3_curl(reference[:,None])[:,0]
            if tri_n3 else
            tri_n2_curl(reference[:,None])[:,0] if tri_n2 else
            (tri_n1_curl(reference[:,None])[:,0] if dim==2 else
             tet_n1_curl(reference[:,None])[:,:,0])
        )
        signs=hcurl.basis_signs[:,local_cell]
        if tet_n2 or tet_n3:
            transform=hcurl.transformations[local_cell]
            ref_values=transform@ref_values
            ref_curls=transform@ref_curls
        elif tri_n3:
            permutation=hcurl.basis_permutation[:,local_cell]
            ref_values=ref_values[permutation]
            ref_curls=ref_curls[permutation]
        mapped=np.linalg.solve(jacobian.T,ref_values.T).T*signs[:,None]*global_sign
        local_basis_values[point_id]=mapped.T
        local=coefficients[basis.element_dofs[:,local_cell]]
        values[:,point_id]=local@mapped
        field_gradient=np.zeros((dim,dim))
        if tet_n3:
            reference_gradients=np.einsum(
                "ab,bcij->acij",hcurl.transformations[local_cell],
                tet_n3_gradients(reference[:,None])[:,:,:,0:1],
            )[:,:,:,0]
        elif tet_n2:
            reference_gradients=np.einsum(
                "ab,bcij->acij",hcurl.transformations[local_cell],
                tet_n2_gradients(reference[:,None])[:,:,:,0:1],
            )[:,:,:,0]
        elif tri_n3:
            reference_gradients=tri_n3_gradients(reference[:,None])[:,:,:,0]
            reference_gradients=reference_gradients[
                hcurl.basis_permutation[:,local_cell]
            ]
        elif tri_n2:
            reference_gradients=tri_n2_gradients(reference[:,None])[:,:,:,0]
        else:
            reference_gradients=np.stack([
                np.outer(bary_grad[second],bary_grad[first])
                -np.outer(bary_grad[first],bary_grad[second])
                for first,second in local_edges
            ])
        for local_edge,ref_gradient in enumerate(reference_gradients):
            physical_gradient=inverse.T@ref_gradient@inverse
            field_gradient+=local[local_edge]*signs[local_edge]*global_sign*physical_gradient
        gradients[:,:,point_id]=field_gradient
        determinant=np.linalg.det(jacobian)
        if dim==2:
            curls[0,point_id]=local@(
                ref_curls*signs*global_sign/determinant
            )
        else:
            physical_curls=np.einsum("ij,bj->bi",jacobian,ref_curls)/determinant
            curls[:,point_id]=local@(
                physical_curls*signs[:,None]*global_sign
            )
    if np.any(outside_mask) and outside=="error":
        point_id=int(np.flatnonzero(outside_mask)[0])
        raise ValueError(
            f"point {point_id} {points[:,point_id].tolist()} lies outside the basis mesh"
        )
    return PointEvaluation(
        values,gradients,cell_ids,references,boundary,ambiguous,outside_mask,
        curl=curls[0] if dim==2 else curls,
        local_basis_values=local_basis_values,
    )


__all__ = ["PointEvaluation", "evaluate_basis_at"]
