Assembly model
==============

Backend selection
-----------------

Applications should select the assembly module explicitly at their backend
boundary.  Forms can retain the same general call shape while the application
chooses ``skfem.asm`` or ``skfemntv.asm``.

``skfemntv`` does not claim complete scikit-fem feature coverage.  Explicit
selection makes unsupported combinations visible and allows backend-equivalence
tests to remain precise.

Nonlinear coefficients
----------------------

Nonlinear material updates and history variables remain in application code.
The application computes coefficient or tangent tensors, then supplies them to
``NativeLinearForm``, ``NativeBilinearForm``, or
``NativeCrossBilinearForm`` for native integration and sparse assembly.

Interfaces
----------

``TriangleSupermesh`` and ``InterfaceSupermesh`` provide common interface
geometry and quadrature.  The application supplies its test and trial-space
tabulation through ``CrossTabulation`` and retains ownership of Mortar, dual,
Nitsche, or contact policy.

Physical-point evaluation
-------------------------

``Basis.evaluate_at(points, coefficients)`` evaluates values and physical
gradients at points with component-first coordinates ``(dim, npoints)``. The
initial implementation covers P0/P1/P2 fields on affine triangle and
tetrahedron meshes. It returns global cell IDs and flags points on cell
boundaries, points shared by multiple cells, and points outside the selected
basis. Use ``outside="nan"`` to retain outside points in the result; the
default is to raise an error. Curved simplex geometry, tensor-product cells,
composite fields, and general projection remain outside this first slice.

Mesh refinement
---------------

``MeshTri.refined(n)`` performs ``n`` uniform refinement steps.
``MeshTri.refined(element_ids)`` performs one conforming adaptive step for the
marked triangle IDs. Named boundaries and subdomains are carried to the
refined mesh. Adaptive refinement produces a conforming triangulation, though
its element layout may differ from scikit-fem. ``MeshTet.refined(n)`` and
``MeshTet.refined(element_ids)`` support uniform and conforming adaptive
tetrahedron refinement and preserve named boundaries and subdomains. The
tetrahedron split ordering matches scikit-fem for the same input arrays.
``MeshQuad.refined(n)`` also supports uniform refinement of linear
quadrilateral meshes while carrying named boundaries and subdomains forward.

``MeshTri``, ``MeshQuad``, and ``MeshTet`` provide ``restrict(elements,
return_mapping=True)`` to extract selected cells, compact unused vertices,
and return the old vertex IDs as the mapping. Their ``remove_elements``
methods apply the same compaction after removing selected cells. Callable and
named-subdomain selectors are supported; named boundaries and subdomains are
remapped to the resulting mesh.

Meshes can be imported and exported through the optional ``meshio`` package
with ``MeshTri.load(path)`` and ``mesh.save(path, point_data=...,
cell_data=...)``. Install the I/O extra with ``pip install
scikit-fem-native[io]``. Linear and selected quadratic triangle, quadrilateral,
tetrahedron, hexahedron, and wedge cell types are supported.
