from __future__ import annotations

import numpy as np

from .basis import (
    Basis,DiscreteField,ElementTetP1,ElementTriP1,ElementTriP2,ElementVector,
)
from .levelset import CutCellQuadrature,ImplicitInterfaceQuadrature


class CutCellBasis:
    """H1 basis tabulated on CSR-like cut-volume quadrature points.

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
        scalar=basis.elem.elem if isinstance(basis.elem,ElementVector) else None
        node_count=int(basis.mesh.t.shape[0])
        supported=(
            isinstance(scalar,ElementTriP1) and node_count==3
            or isinstance(scalar,ElementTriP2) and node_count==6
            or isinstance(scalar,ElementTetP1) and node_count==4
        )
        if not supported:
            raise NotImplementedError(
                "CutCellBasis currently supports vector-wrapped TriP1, "
                "straight-sided TriP2, and TetP1"
            )
        components=basis.elem._dim
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
        self.shape,reference_gradients=_tabulate_reference(
            scalar,self.reference_coordinates
        )
        self.gradients=np.empty((
            len(self.weights),node_count,basis.mesh.dim()
        ))
        for cell in self.tind:
            selection=quadrature.cell_slice(int(cell))
            cell_nodes=basis.mesh.t[:,cell]
            jacobian=(
                basis.mesh.p[:,cell_nodes[1:basis.mesh.dim()+1]]
                -basis.mesh.p[:,[cell_nodes[0]]]
            )
            self.gradients[selection]=(
                reference_gradients[selection]@np.linalg.inv(jacobian)
            )
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
        counts=np.diff(self.cell_offsets)[self.tind]
        self.active_cell_offsets=np.concatenate((
            np.array([0],dtype=np.int64),np.cumsum(counts,dtype=np.int64)
        ))
        self.active_cell_dofs=np.ascontiguousarray(self.cell_dofs[self.tind])
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
            self.active_cell_offsets,self.active_cell_dofs,
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
    """Trace of a supported background basis on a reconstructed interface."""

    def __init__(
        self,basis: Basis,quadrature: ImplicitInterfaceQuadrature,
        *,side: str="negative",
    ):
        if not isinstance(quadrature,ImplicitInterfaceQuadrature):
            raise TypeError(
                "ImplicitFacetBasis requires ImplicitInterfaceQuadrature"
            )
        if side not in {"negative","positive"}:
            raise ValueError("implicit facet side must be negative or positive")
        super().__init__(basis,quadrature)
        self.side=side
        if side=="positive":
            self.normal_vectors=-self.normal_vectors
            self.normal_vectors.flags.writeable=False
            self.normals=self.normal_vectors[:,None,:]
            self.normals.flags.writeable=False


def _tabulate_reference(element,points):
    dimension=points.shape[1]
    if isinstance(element,(ElementTriP1,ElementTetP1)):
        shape=np.column_stack((1.-points.sum(axis=1),points))
        constant=np.vstack((-np.ones(dimension),np.eye(dimension)))
        gradients=np.broadcast_to(
            constant,(len(points),len(constant),dimension)
        ).copy()
        return shape,gradients
    if isinstance(element,ElementTriP2):
        x=points[:,0];y=points[:,1]
        bary=np.column_stack((1.-x-y,x,y))
        shape=np.column_stack((
            bary[:,0]*(2.*bary[:,0]-1.),
            bary[:,1]*(2.*bary[:,1]-1.),
            bary[:,2]*(2.*bary[:,2]-1.),
            4.*bary[:,0]*bary[:,1],
            4.*bary[:,1]*bary[:,2],
            4.*bary[:,0]*bary[:,2],
        ))
        dbary=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
        gradients=np.empty((len(points),6,2),dtype=np.float64)
        for index in range(3):
            gradients[:,index]=(4.*bary[:,index]-1.)[:,None]*dbary[index]
        gradients[:,3]=4.*(
            bary[:,0,None]*dbary[1]+bary[:,1,None]*dbary[0]
        )
        gradients[:,4]=4.*(
            bary[:,1,None]*dbary[2]+bary[:,2,None]*dbary[1]
        )
        gradients[:,5]=4.*(
            bary[:,0,None]*dbary[2]+bary[:,2,None]*dbary[0]
        )
        return shape,gradients
    raise NotImplementedError("unsupported cut-cell reference element")
