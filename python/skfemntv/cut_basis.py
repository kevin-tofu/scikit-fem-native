from __future__ import annotations

import numpy as np

from .basis import Basis,DiscreteField,ElementTetP1,ElementTriP1,ElementVector
from .levelset import CutCellQuadrature,ImplicitInterfaceQuadrature


class CutCellBasis:
    """P1 H1 basis tabulated on CSR-like cut-volume quadrature points.

    Quadrature points are flattened rather than padded by cell.  The
    ``quadrature_dofs`` array maps every point to global element DOFs, while
    ``cell_offsets`` retains cell-local ranges.
    """

    def __init__(self,basis: Basis,quadrature: CutCellQuadrature):
        if not isinstance(basis,Basis):
            raise TypeError("CutCellBasis requires a Basis")
        if not isinstance(quadrature,CutCellQuadrature):
            raise TypeError("CutCellBasis requires CutCellQuadrature")
        if len(quadrature.cell_offsets)!=basis.mesh.nelements+1:
            raise ValueError("basis and cut quadrature have different cell counts")
        if not isinstance(basis.elem,ElementVector) or not isinstance(
            basis.elem.elem,(ElementTriP1,ElementTetP1)
        ):
            raise NotImplementedError(
                "CutCellBasis currently supports vector-wrapped TriP1 and TetP1"
            )
        expected=basis.mesh.dim()+1
        node_count=expected
        components=basis.elem._dim
        if basis.mesh.t.shape[0]!=expected:
            raise NotImplementedError(
                "CutCellBasis currently supports affine Tri3 and Tet4 meshes"
            )
        positions={int(cell):local for local,cell in enumerate(basis.tind)}
        try:
            local_cells=np.asarray(
                [positions[int(cell)] for cell in quadrature.cells],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"cut quadrature cell {int(error.args[0])} is absent from Basis"
            ) from error

        self.mesh=basis.mesh
        self.elem=basis.elem
        self.N=basis.N
        self.parent_basis=basis
        self.quadrature=quadrature
        self.cell_offsets=quadrature.cell_offsets
        self.cells=quadrature.cells
        self.points=quadrature.points
        self.reference_coordinates=quadrature.reference_points
        self.weights=quadrature.weights
        self.normal_vectors=quadrature.normals
        self.tind=np.flatnonzero(np.diff(self.cell_offsets)>0)
        self.shape=np.column_stack((
            1.-self.reference_coordinates.sum(axis=1),
            self.reference_coordinates,
        ))
        self.gradients=np.empty((len(self.weights),expected,basis.mesh.dim()))
        reference_gradients=np.vstack((
            -np.ones(basis.mesh.dim()),np.eye(basis.mesh.dim())
        ))
        for cell in self.tind:
            selection=quadrature.cell_slice(int(cell))
            cell_nodes=basis.mesh.t[:,cell]
            jacobian=(
                basis.mesh.p[:,cell_nodes[1:]]-basis.mesh.p[:,[cell_nodes[0]]]
            )
            physical=reference_gradients@np.linalg.inv(jacobian)
            self.gradients[selection]=physical
        self.quadrature_dofs=(
            basis.element_dofs[:,local_cells].T.copy()
            if len(local_cells) else
            np.empty((0,basis.element_dofs.shape[0]),dtype=np.int64)
        )
        self.cell_dofs=np.zeros((
            basis.mesh.nelements,node_count,components
        ),dtype=np.int64)
        self.cell_dofs[np.asarray(basis.tind,dtype=np.int64)]=(
            basis.element_dofs.T.reshape(
                len(basis.tind),node_count,components
            )
        )
        # Native form assemblers can consume each flattened point as an entity
        # with one quadrature point.  This is a zero-padding adapter: storage
        # remains proportional to actual cut points and assembly stays in C++.
        self.dx=self.weights[:,None]
        self.tabulated_shape=self.shape[:,None,:]
        self.tabulated_gradients=self.gradients[:,None,:,:]
        self.global_coordinates=self.points[:,None,:]
        self.normals=self.normal_vectors[:,None,:]
        self.element_dofs=self.quadrature_dofs.T
        self.basis=()
        for array in (
            self.shape,self.gradients,self.quadrature_dofs,self.tind,
            self.dx,self.tabulated_shape,self.tabulated_gradients,
            self.global_coordinates,self.element_dofs,
            self.cell_dofs,self.normals,
        ):
            array.flags.writeable=False

    @property
    def npoints(self):
        return len(self.weights)

    @property
    def nelems(self):
        return len(self.tind)

    def cell_slice(self,cell):
        return self.quadrature.cell_slice(cell)

    def interpolate(self,coefficients):
        coefficients=np.asarray(coefficients,dtype=np.float64)
        if coefficients.shape!=(self.N,):
            raise ValueError(f"coefficients must have shape ({self.N},)")
        components=self.elem._dim
        nodes=self.shape.shape[1]
        local=coefficients[self.quadrature_dofs].reshape(
            self.npoints,nodes,components
        )
        value=np.einsum("qn,qnc->cq",self.shape,local)
        gradient=np.einsum("qnd,qnc->cdq",self.gradients,local)
        if components==1:
            value=value[0]
            gradient=gradient[0]
        return DiscreteField(value,gradient)

    def integrate(self,values):
        values=np.asarray(values,dtype=np.float64)
        if values.shape!=(self.npoints,):
            raise ValueError(
                f"cut integrand must have shape ({self.npoints},)"
            )
        return float(self.weights@values)


class ImplicitFacetBasis(CutCellBasis):
    """Trace of an affine P1 background basis on a reconstructed interface."""

    def __init__(self,basis: Basis,quadrature: ImplicitInterfaceQuadrature):
        if not isinstance(quadrature,ImplicitInterfaceQuadrature):
            raise TypeError(
                "ImplicitFacetBasis requires ImplicitInterfaceQuadrature"
            )
        super().__init__(basis,quadrature)
