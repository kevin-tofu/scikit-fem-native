from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


class MeshTet:
    """Minimal tetrahedral mesh container compatible with skfn.Basis."""

    def __init__(self, p: np.ndarray | None = None, t: np.ndarray | None = None):
        self.p = np.asarray(
            p if p is not None else
            [[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
            dtype=np.float64,
        )
        self.t = np.asarray(
            t if t is not None else [[0], [1], [2], [3]], dtype=np.int64
        )
        if self.p.ndim != 2 or self.p.shape[0] != 3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim != 2 or self.t.shape[0] != 4:
            raise ValueError("t must have shape (4, elements)")

    @classmethod
    def init_tensor(cls, x, y, z):
        x, y, z = map(np.asarray, (x, y, z))
        points = np.array(
            [[a, b, c] for c in z for b in y for a in x], dtype=float
        ).T
        nx, ny = len(x), len(y)
        node = lambda i,j,k: i + nx*(j + ny*k)
        pattern=((0,1,3,7),(0,3,2,7),(0,2,6,7),
                 (0,6,4,7),(0,4,5,7),(0,5,1,7))
        cells=[]
        for k in range(len(z)-1):
            for j in range(len(y)-1):
                for i in range(len(x)-1):
                    cube=(node(i,j,k),node(i+1,j,k),node(i,j+1,k),node(i+1,j+1,k),
                          node(i,j,k+1),node(i+1,j,k+1),node(i,j+1,k+1),node(i+1,j+1,k+1))
                    cells.extend(tuple(cube[q] for q in tet) for tet in pattern)
        return cls(points,np.asarray(cells,dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    def boundary_facets(self):
        faces={}
        for element in range(self.nelements):
            for local in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
                face=tuple(int(self.t[i,element]) for i in local);key=tuple(sorted(face))
                faces[key]=None if key in faces else (element,local,face)
        return np.array([v[2] for v in faces.values() if v is not None],dtype=np.int64).T


class MeshTet2(MeshTet):
    """Quadratic tetrahedral mesh with six shared edge nodes per element."""

    @classmethod
    def from_mesh(cls, mesh: MeshTet):
        points = [mesh.p[:, i].copy() for i in range(mesh.p.shape[1])]
        edge_nodes = {}
        rows = [list(map(int, mesh.t[i])) for i in range(4)]
        edge_order = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
        edge_rows = [[] for _ in edge_order]
        for element, vertices in enumerate(mesh.t.T):
            for row, (a,b) in zip(edge_rows, edge_order):
                edge=tuple(sorted((int(vertices[a]),int(vertices[b]))))
                if edge not in edge_nodes:
                    edge_nodes[edge]=len(points)
                    points.append(0.5*(mesh.p[:,edge[0]]+mesh.p[:,edge[1]]))
                row.append(edge_nodes[edge])
        return cls(np.asarray(points).T,np.asarray(rows+edge_rows,dtype=np.int64))

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshTet())
            self.p,self.t=generated.p,generated.t
            return
        self.p=np.asarray(p,dtype=np.float64);self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=3 or self.t.ndim!=2 or self.t.shape[0]!=10:
            raise ValueError("quadratic tetra mesh requires p (3,n) and t (10,e)")

    def boundary_facets(self):
        edge={(0,1):4,(1,2):5,(0,2):6,(0,3):7,(1,3):8,(2,3):9}
        faces={};vertices=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
        for e,nodes in enumerate(self.t.T):
            for face in vertices:
                full=tuple(face)+tuple(
                    edge[tuple(sorted((face[i],face[(i+1)%3])))] for i in range(3)
                )
                key=tuple(sorted(int(nodes[i]) for i in face))
                faces[key]=None if key in faces else tuple(int(nodes[i]) for i in full)
        return np.array([v for v in faces.values() if v is not None],dtype=np.int64).T


class MeshHex:
    def __init__(self, p: np.ndarray | None = None, t: np.ndarray | None = None):
        self.p = np.asarray(
            p if p is not None else
            [[0.,1.,0.,0.,1.,1.,0.,1.],
             [0.,0.,1.,0.,1.,0.,1.,1.],
             [0.,0.,0.,1.,0.,1.,1.,1.]], dtype=np.float64
        )
        self.t = np.asarray(
            t if t is not None else np.arange(8)[:, None], dtype=np.int64
        )
        if self.p.ndim != 2 or self.p.shape[0] != 3:
            raise ValueError("p must have shape (3, nodes)")
        if self.t.ndim != 2 or self.t.shape[0] != 8:
            raise ValueError("t must have shape (8, elements)")

    @classmethod
    def init_tensor(cls, x, y, z):
        x, y, z = map(np.asarray, (x, y, z))
        points = np.array(
            [[a,b,c] for c in z for b in y for a in x], dtype=float
        ).T
        nx, ny = len(x), len(y)
        node = lambda i,j,k: i + nx*(j + ny*k)
        cells = []
        for k in range(len(z)-1):
            for j in range(len(y)-1):
                for i in range(len(x)-1):
                    cells.append((
                        node(i,j,k), node(i+1,j,k), node(i,j+1,k), node(i,j,k+1),
                        node(i+1,j+1,k), node(i+1,j,k+1),
                        node(i,j+1,k+1), node(i+1,j+1,k+1),
                    ))
        return cls(points, np.asarray(cells, dtype=np.int64).T)

    @property
    def nelements(self):
        return self.t.shape[1]

    def dim(self):
        return 3

    def boundary_facets(self):
        if self.t.shape[0] == 8:
            local_faces=((0,1,4,2),(3,6,7,5),(0,3,5,1),
                         (2,4,7,6),(0,2,6,3),(1,5,7,4))
            corners=local_faces
        else:
            index=lambda i,j,k:i+3*j+9*k
            local_faces=(
                tuple(index(i,j,0) for j in range(3) for i in range(3)),
                tuple(index(i,j,2) for j in range(3) for i in range(3)),
                tuple(index(i,0,k) for k in range(3) for i in range(3)),
                tuple(index(i,2,k) for k in range(3) for i in range(3)),
                tuple(index(0,j,k) for k in range(3) for j in range(3)),
                tuple(index(2,j,k) for k in range(3) for j in range(3)),
            )
            corners=tuple((face[0],face[2],face[8],face[6]) for face in local_faces)
        found={}
        for e,nodes in enumerate(self.t.T):
            for face,corner in zip(local_faces,corners):
                key=tuple(sorted(int(nodes[i]) for i in corner))
                found[key]=None if key in found else tuple(int(nodes[i]) for i in face)
        return np.array([v for v in found.values() if v is not None],dtype=np.int64).T


class MeshHex2(MeshHex):
    @classmethod
    def from_mesh(cls,mesh:MeshHex):
        points=[];lookup={};cells=[]
        bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        grid=(0.,.5,1.)
        for vertices in mesh.t.T:
            xyz=mesh.p[:,vertices];cell=[]
            for zeta in grid:
                for eta in grid:
                    for xi in grid:
                        point=np.zeros(3)
                        for n,bit in enumerate(bits):
                            factors=np.where(bit,(xi,eta,zeta),1.-np.array((xi,eta,zeta)))
                            point+=np.prod(factors)*xyz[:,n]
                        key=tuple(np.round(point,14))
                        if key not in lookup:
                            lookup[key]=len(points);points.append(point)
                        cell.append(lookup[key])
            cells.append(cell)
        return cls(np.asarray(points).T,np.asarray(cells,dtype=np.int64).T)

    def __init__(self,p=None,t=None):
        if p is None or t is None:
            generated=type(self).from_mesh(MeshHex())
            self.p,self.t=generated.p,generated.t;return
        self.p=np.asarray(p,dtype=np.float64);self.t=np.asarray(t,dtype=np.int64)
        if self.p.ndim!=2 or self.p.shape[0]!=3 or self.t.ndim!=2 or self.t.shape[0]!=27:
            raise ValueError("quadratic hex mesh requires p (3,n) and t (27,e)")


class _ComposableElement:
    def __mul__(self,other):
        if isinstance(other,ElementComposite):
            return ElementComposite(self,*other.elems)
        return ElementComposite(self,other)


class ElementTetP1(_ComposableElement):
    def __init__(self):
        self.doflocs = np.array(
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        )


class ElementTetP2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array(
            [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
             [.5,0.,0.],[.5,.5,0.],[0.,.5,0.],[0.,0.,.5],
             [.5,0.,.5],[0.,.5,.5]]
        )


class ElementHex1(_ComposableElement):
    def __init__(self):
        self.doflocs = np.array(
            [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
             [1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]]
        )


class ElementHex2(_ComposableElement):
    def __init__(self):
        self.doflocs=np.array(
            [[x,y,z] for z in (0.,.5,1.) for y in (0.,.5,1.)
             for x in (0.,.5,1.)]
        )


class ElementVector(_ComposableElement):
    def __init__(self, element, dim: int = 3):
        self.elem = element
        self._dim = dim


class ElementComposite(_ComposableElement):
    """Ordered collection of nodal H1 subfields."""

    def __init__(self,*elements):
        if len(elements)<2:
            raise ValueError("ElementComposite requires at least two fields")
        flattened=[]
        for element in elements:
            if isinstance(element,ElementComposite):
                flattened.extend(element.elems)
            else:
                flattened.append(element)
        self.elems=tuple(flattened)


class _Field:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class Basis:
    def __init__(self, mesh: MeshTet, element: ElementVector, intorder=2):
        if isinstance(element,ElementComposite):
            self._init_composite(mesh,element,intorder)
            return
        if not isinstance(element, ElementVector) or not isinstance(
            element.elem, (ElementTetP1, ElementTetP2, ElementHex1, ElementHex2)
        ):
            raise NotImplementedError(
                "independent Basis supports vector TetP1/P2 and Hex1/2"
            )
        self.mesh, self.elem = mesh, element
        self._tet = isinstance(element.elem, (ElementTetP1,ElementTetP2))
        self._quadratic_tet=isinstance(element.elem,ElementTetP2)
        self._quadratic_hex=isinstance(element.elem,ElementHex2)
        if self._quadratic_tet:
            self.X=np.array(
                [[.25,.7857142857142857,.0714285714285714,.0714285714285714,.0714285714285714,
                  .1005964238332008,.3994035761667992,.3994035761667992,.3994035761667992,
                  .1005964238332008,.1005964238332008],
                 [.25,.0714285714285714,.0714285714285714,.0714285714285714,.7857142857142857,
                  .3994035761667992,.1005964238332008,.3994035761667992,.1005964238332008,
                  .3994035761667992,.1005964238332008],
                 [.25,.0714285714285714,.0714285714285714,.7857142857142857,.0714285714285714,
                  .3994035761667992,.3994035761667992,.1005964238332008,.1005964238332008,
                  .1005964238332008,.3994035761667992]])
            self.W=np.array([-.01315555555555555]+[.00762222222222222]*4+[.02488888888888888]*6)
        elif self._tet:
            self.X = np.array(
                [[.5854101966249685,.1381966011250105,.1381966011250105,.1381966011250105],
                 [.1381966011250105,.5854101966249685,.1381966011250105,.1381966011250105],
                 [.1381966011250105,.1381966011250105,.5854101966249685,.1381966011250105]]
            )
            self.W = np.full(4,1./24.)
        elif self._quadratic_hex:
            points=np.array([.5-np.sqrt(3./5.)/2.,.5,.5+np.sqrt(3./5.)/2.])
            one_weights=np.array([5./18.,4./9.,5./18.])
            entries=[(x,y,z,wx*wy*wz) for z,wz in zip(points,one_weights)
                     for y,wy in zip(points,one_weights) for x,wx in zip(points,one_weights)]
            self.X=np.array([entry[:3] for entry in entries]).T
            self.W=np.array([entry[3] for entry in entries])
        else:
            gauss=(1.+np.array([-1.,1.])/np.sqrt(3.))/2.
            self.X=np.array([[a,b,c] for c in gauss for b in gauss for a in gauss]).T
            self.W=np.full(8,1./8.)
        nodes = mesh.t.shape[0]
        components=element._dim
        self.N = mesh.p.shape[1] * components
        self.nodal_dofs = np.arange(self.N).reshape(-1, components).T
        self.element_dofs = self.nodal_dofs[:, mesh.t].transpose(2, 1, 0).reshape(
            mesh.nelements, nodes*components
        ).T
        self.doflocs = np.repeat(mesh.p, components, axis=1)
        self.tabulated_shape,self.tabulated_gradients,self.dx=self._geometry()
        element_coordinates=np.stack([
            mesh.p[:,mesh.t[:,element]].T
            for element in range(mesh.nelements)
        ])
        self.global_coordinates=np.einsum(
            "eqn,end->eqd",self.tabulated_shape,element_coordinates
        )
        self.normals=None
        self.basis = self._vector_fields()

    def _init_composite(self,mesh,element,intorder):
        vector_elements=tuple(
            field if isinstance(field,ElementVector)
            else ElementVector(field,dim=1)
            for field in element.elems
        )
        scalar_types=tuple(type(field.elem) for field in vector_elements)
        if len(set(scalar_types))!=1:
            raise NotImplementedError(
                "composite fields currently require the same nodal order"
            )
        subbases=[Basis(mesh,field,intorder=intorder) for field in vector_elements]
        components=tuple(field._dim for field in vector_elements)
        total_components=sum(components)
        node_count=mesh.p.shape[1]
        offsets=np.cumsum((0,)+components[:-1])
        for subbasis,offset,count in zip(subbases,offsets,components):
            nodal=np.stack([
                total_components*np.arange(node_count)+offset+c
                for c in range(count)
            ])
            subbasis.nodal_dofs=nodal
            subbasis.N=node_count*total_components
            local_nodes=mesh.t.shape[0]
            subbasis.element_dofs=nodal[:,mesh.t].transpose(
                2,1,0
            ).reshape(mesh.nelements,local_nodes*count).T
            subbasis.doflocs=np.repeat(mesh.p,count,axis=1)
        first=subbases[0]
        self.mesh,self.elem=mesh,element
        self.subbases=tuple(subbases)
        self.field_components=components
        self.N=node_count*total_components
        self.nodal_dofs=np.arange(self.N).reshape(
            node_count,total_components
        ).T
        self.element_dofs=np.concatenate(
            [subbasis.element_dofs for subbasis in subbases],axis=0
        )
        self.doflocs=np.repeat(mesh.p,total_components,axis=1)
        self.X,self.W=first.X,first.W
        self.dx=first.dx
        self.global_coordinates=first.global_coordinates
        self.normals=None
        self.basis=tuple()

    def _geometry(self):
        if self._quadratic_tet:
            bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            pairs=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
            nq=len(self.W);shape=np.empty((nq,10));refgrad=np.empty((nq,10,3))
            dl=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            for q,L in enumerate(bary):
                for i in range(4):
                    shape[q,i]=L[i]*(2.*L[i]-1.)
                    refgrad[q,i]=(4.*L[i]-1.)*dl[i]
                for k,(i,j) in enumerate(pairs,start=4):
                    shape[q,k]=4.*L[i]*L[j]
                    refgrad[q,k]=4.*(L[j]*dl[i]+L[i]*dl[j])
        elif self._tet:
            reference=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
            refgrad=np.broadcast_to(reference,(self.X.shape[1],4,3))
        elif self._quadratic_hex:
            grid=(0.,.5,1.)
            def values(x):
                return np.array([2.*(x-.5)*(x-1.),4.*x*(1.-x),2.*x*(x-.5)])
            def derivatives(x):
                return np.array([4.*x-3.,4.-8.*x,4.*x-1.])
            nq=self.X.shape[1];shape=np.empty((nq,27));refgrad=np.empty((nq,27,3))
            for q,(x,y,z) in enumerate(self.X.T):
                v=(values(x),values(y),values(z));d=(derivatives(x),derivatives(y),derivatives(z))
                n=0
                for k in range(3):
                    for j in range(3):
                        for i in range(3):
                            shape[q,n]=v[0][i]*v[1][j]*v[2][k]
                            refgrad[q,n]=(
                                d[0][i]*v[1][j]*v[2][k],
                                v[0][i]*d[1][j]*v[2][k],
                                v[0][i]*v[1][j]*d[2][k],
                            );n+=1
        else:
            bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
            shape=np.empty((self.X.shape[1],8));refgrad=np.empty((self.X.shape[1],8,3))
            for q,xi in enumerate(self.X.T):
                for n,bit in enumerate(bits):
                    factors=np.where(bit,xi,1.-xi);shape[q,n]=np.prod(factors)
                    for j in range(3):
                        refgrad[q,n,j]=(1. if bit[j] else -1.)*np.prod(np.delete(factors,j))
        nq,nodes=shape.shape
        gradients=np.empty((self.mesh.nelements,nq,nodes,3))
        dx=np.empty((self.mesh.nelements,nq))
        for e, nodes in enumerate(self.mesh.t.T):
            x = self.mesh.p[:, nodes]
            for q in range(nq):
                jacobian=x@refgrad[q]
                determinant=np.linalg.det(jacobian)
                gradients[e,q]=refgrad[q]@np.linalg.inv(jacobian)
                dx[e,q]=abs(determinant)*self.W[q]
        return np.broadcast_to(shape,(self.mesh.nelements,nq,shape.shape[1])).copy(),gradients,dx

    def _evaluate_reference(self, points):
        old_points = self.X
        self.X = np.asarray(points)
        try:
            if self._quadratic_tet:
                bary=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                pairs=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
                shape=np.empty((len(bary),10));grad=np.empty((len(bary),10,3))
                dl=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
                for q,L in enumerate(bary):
                    for i in range(4):
                        shape[q,i]=L[i]*(2*L[i]-1);grad[q,i]=(4*L[i]-1)*dl[i]
                    for k,(i,j) in enumerate(pairs,start=4):
                        shape[q,k]=4*L[i]*L[j];grad[q,k]=4*(L[j]*dl[i]+L[i]*dl[j])
                return shape,grad
            if self._tet:
                shape=np.vstack((1.-self.X.sum(axis=0),self.X)).T
                grad=np.broadcast_to(np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
                                     (shape.shape[0],4,3))
                return shape,grad
            if self._quadratic_hex:
                values=lambda x:np.array([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)])
                deriv=lambda x:np.array([4*x-3,4-8*x,4*x-1])
                shape=np.empty((self.X.shape[1],27));grad=np.empty((self.X.shape[1],27,3))
                for q,(x,y,z) in enumerate(self.X.T):
                    v=(values(x),values(y),values(z));d=(deriv(x),deriv(y),deriv(z));n=0
                    for k in range(3):
                        for j in range(3):
                            for i in range(3):
                                shape[q,n]=v[0][i]*v[1][j]*v[2][k]
                                grad[q,n]=(d[0][i]*v[1][j]*v[2][k],
                                           v[0][i]*d[1][j]*v[2][k],
                                           v[0][i]*v[1][j]*d[2][k]);n+=1
                return shape,grad
            bits=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
            shape=np.empty((self.X.shape[1],8));grad=np.empty((self.X.shape[1],8,3))
            for q,xi in enumerate(self.X.T):
                for n,bit in enumerate(bits):
                    factors=np.where(bit,xi,1-xi);shape[q,n]=np.prod(factors)
                    for j in range(3):
                        grad[q,n,j]=(1 if bit[j] else -1)*np.prod(np.delete(factors,j))
            return shape,grad
        finally:
            self.X = old_points

    def _vector_fields(self):
        fields = []
        nodes=self.mesh.t.shape[0];nq=self.X.shape[1]
        components=self.elem._dim
        for node in range(nodes):
            for component in range(components):
                value=np.zeros((components,self.mesh.nelements,nq))
                grad=np.zeros((components,3,self.mesh.nelements,nq))
                value[component] = self.tabulated_shape[:, :, node]
                grad[component] = self.tabulated_gradients[:, :, node].transpose(2, 0, 1)
                fields.append((_Field(value, grad),))
        return fields


class FacetBasis:
    def __init__(self, mesh, element, facets=None, intorder=2):
        volume = Basis(mesh, element, intorder=intorder)
        facets = mesh.boundary_facets() if facets is None else np.asarray(facets)
        expected_face_nodes = 3 if isinstance(element.elem,ElementTetP1) else (
            6 if isinstance(element.elem,ElementTetP2) else (
                4 if isinstance(element.elem,ElementHex1) else 9
            )
        )
        if facets.shape[0] != expected_face_nodes:
            facets = facets.T
        is_tet=isinstance(element.elem,(ElementTetP1,ElementTetP2))
        if is_tet:
            local_faces=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
            if isinstance(element.elem,ElementTetP2):
                edge_map={(0,1):4,(1,2):5,(0,2):6,(0,3):7,(1,3):8,(2,3):9}
                local_faces=tuple(tuple(face)+tuple(
                    edge_map[tuple(sorted((face[i],face[(i+1)%3])))] for i in range(3)
                ) for face in local_faces)
            corners=tuple(face[:3] for face in local_faces)
            if intorder>=4:
                a=.445948490915965;b=.091576213509771
                bary=np.array([[a,a,1-2*a],[a,1-2*a,a],[1-2*a,a,a],
                               [b,b,1-2*b],[b,1-2*b,b],[1-2*b,b,b]])
                face_weights=np.array([.1116907948390055]*3+[.054975871827661]*3)
            else:
                bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
                face_weights=np.full(3,1/6)
            face_points=bary[:,1:]
        else:
            if isinstance(element.elem,ElementHex1):
                local_faces=((0,1,4,2),(3,6,7,5),(0,3,5,1),
                             (2,4,7,6),(0,2,6,3),(1,5,7,4))
                corners=local_faces
            else:
                index=lambda i,j,k:i+3*j+9*k
                local_faces=(
                    tuple(index(i,j,0) for j in range(3) for i in range(3)),
                    tuple(index(i,j,2) for j in range(3) for i in range(3)),
                    tuple(index(i,0,k) for k in range(3) for i in range(3)),
                    tuple(index(i,2,k) for k in range(3) for i in range(3)),
                    tuple(index(0,j,k) for k in range(3) for j in range(3)),
                    tuple(index(2,j,k) for k in range(3) for j in range(3)))
                corners=tuple((face[0],face[2],face[8],face[6]) for face in local_faces)
            order=3 if intorder>=4 else 2
            points,weights=np.polynomial.legendre.leggauss(order)
            points=(points+1.)/2.;weights=weights/2.
            face_points=np.array([[r,s] for s in points for r in points])
            face_weights=np.array([wr*ws for ws in weights for wr in weights])
        lookup={}
        for e,nodes in enumerate(mesh.t.T):
            for local,corner in zip(local_faces,corners):
                lookup[tuple(sorted(int(nodes[i]) for i in corner))]=(e,local,corner)
        nq=len(face_weights);nodes_per_element=mesh.t.shape[0]
        self.mesh, self.elem = mesh, element
        self.X=face_points.T;self.W=face_weights
        self.N = volume.N
        self.doflocs = volume.doflocs
        components=element._dim
        self.element_dofs=np.empty((nodes_per_element*components,facets.shape[1]),dtype=np.int64)
        self.tabulated_shape=np.zeros((facets.shape[1],nq,nodes_per_element))
        self.tabulated_gradients=np.empty((facets.shape[1],nq,nodes_per_element,3))
        self.dx = np.empty((facets.shape[1], nq))
        self.global_coordinates=np.empty((facets.shape[1],nq,3))
        self.normals=np.empty((facets.shape[1],nq,3))
        self.parent_elements=np.empty(facets.shape[1],dtype=np.int64)
        self.local_faces=[]
        for f, face in enumerate(facets.T):
            face_corners=face[:3] if is_tet else (face[0],face[2] if len(face)==9 else face[1],
                face[8] if len(face)==9 else face[2],face[6] if len(face)==9 else face[3])
            e,local,corner=lookup[tuple(sorted(map(int,face_corners)))]
            self.parent_elements[f]=e
            self.local_faces.append(tuple(local))
            self.element_dofs[:, f] = volume.element_dofs[:, e]
            reference_corners=element.elem.doflocs[np.asarray(corner)]
            for q,(r,s) in enumerate(face_points):
                if is_tet:
                    reference=(1-r-s)*reference_corners[0]+r*reference_corners[1]+s*reference_corners[2]
                    derivatives=np.stack((reference_corners[1]-reference_corners[0],
                                          reference_corners[2]-reference_corners[0]),axis=1)
                else:
                    weights4=np.array([(1-r)*(1-s),r*(1-s),r*s,(1-r)*s])
                    reference=weights4@reference_corners
                    derivatives=np.stack((
                        (1-s)*(reference_corners[1]-reference_corners[0])+s*(reference_corners[2]-reference_corners[3]),
                        (1-r)*(reference_corners[3]-reference_corners[0])+r*(reference_corners[2]-reference_corners[1])),axis=1)
                shape,refgrad=volume._evaluate_reference(reference[None,:].T)
                x=mesh.p[:,mesh.t[:,e]];jacobian=x@refgrad[0]
                physical=refgrad[0]@np.linalg.inv(jacobian)
                tangents=jacobian@derivatives
                point=shape[0]@x.T
                normal=np.cross(tangents[:,0],tangents[:,1])
                normal/=np.linalg.norm(normal)
                if np.dot(normal,point-x.mean(axis=1))<0.:
                    normal=-normal
                self.tabulated_shape[f,q]=shape[0]
                self.tabulated_gradients[f,q]=physical
                self.global_coordinates[f,q]=point
                self.normals[f,q]=normal
                self.dx[f,q]=np.linalg.norm(np.cross(tangents[:,0],tangents[:,1]))*face_weights[q]
        self.basis = self._vector_fields()
        self.volume_basis=volume

    def _vector_fields(self):
        fields = []
        components=self.elem._dim
        for node in range(self.mesh.t.shape[0]):
            for component in range(components):
                value = np.zeros((components, self.dx.shape[0], self.dx.shape[1]))
                grad = np.zeros((components, 3, self.dx.shape[0], self.dx.shape[1]))
                value[component] = self.tabulated_shape[:, :, node]
                grad[component] = self.tabulated_gradients[:, :, node].transpose(2, 0, 1)
                fields.append((_Field(value, grad),))
        return fields
