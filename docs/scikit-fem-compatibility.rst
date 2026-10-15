scikit-fem compatibility and supported features
================================================

.. meta::
   :description: Learn which scikit-fem workflows, DOF ordering conventions, meshes, elements, and assembly operations are supported by scikit-fem-native and skfemntv.

``scikit-fem-native`` follows selected public conventions from
`scikit-fem <https://scikit-fem.readthedocs.io/>`_ so applications can retain
readable Python weak forms while choosing native assembly where the supported
subsets overlap.  It is an independent project and is not an official
scikit-fem distribution or backend.

Backend selection
-----------------

The installed distribution is ``scikit-fem-native`` and its import package is
``skfemntv``.  Applications should keep backend selection explicit:

.. code-block:: python

   import skfemntv as skfem

For a supported mesh, element, basis, and form, this preserves public DOF
ordering and numerical results.  It is not a promise that every scikit-fem
program can replace ``import skfem`` unchanged.  Keep equivalence tests at the
backend boundary and consult the runtime capability registry:

.. code-block:: python

   import skfemntv

   skfemntv.supports("space.h1")
   skfemntv.capabilities()
   skfemntv.get_compatibility("skfem.Basis.probes")

Compatibility boundary
----------------------

The supported general assembly path includes functional, linear, bilinear,
and cross-bilinear forms; selected meshes, elements, bases, quadrature, and
tabulation; caller-supplied coefficient fields; and native sparse scatter.
Higher-order DOFs follow scikit-fem's topological entity ordering: vertices
first, shared edges or facets next, and element interiors last.

Support is deliberately feature-specific.  Experimental H(curl) simplex
slices use separate basis and assembler classes and are not accepted by the
general ``Basis`` or ``asm`` interface.  Query ``capabilities()`` in the
installed version for the machine-readable, current boundary.
``compatibility_items()`` and ``get_compatibility()`` expose a finer inventory
of common upstream API names, their native mapping, and the status of gaps.

Parity target
-------------

The compatibility target is the public scikit-fem workflow, not only matching
form syntax.  For every supported slice, parity includes familiar API
semantics, DOF ordering, and numerical results.  The current reference
installation used for the compatibility inventory is scikit-fem 12.0.2; the
upstream API is documented in its `versioned API reference
<https://scikit-fem.readthedocs.io/en/stable/api.html>`_.

``ElementTriP2`` and ``ElementTetP2`` can be used on linear triangle and
tetrahedron meshes.  The basis constructs straight-sided quadratic geometry
nodes internally; this preserves the physical affine map while providing the
edge nodes required by the P2 field.  Named boundaries, including their
orientation metadata, and named cell subdomains are carried over during this
conversion.  ``Basis.mesh`` remains the original input mesh, matching
scikit-fem; the promoted topology is held privately for basis calculations.
P2 ``FacetBasis`` construction and boundary linear/bilinear
assembly are also covered on these linear meshes.  Mixed P2/P1 composite
bases, including Taylor--Hood velocity/pressure fields, promote the shared
mesh once and preserve the named regions.  For the same input node and cell
arrays, P2 ``doflocs`` and ``element_dofs`` use the scikit-fem ordering.

The current native implementation does not yet cover the full workflow around
assembly.  ``Basis.evaluate_at`` experimentally supports values and physical
gradients for affine triangle/tetrahedron P0, P1, and P2 bases; curved geometry,
tensor-product cells, and the upstream ``probes``/``interpolator`` return
contracts remain open.  ``Basis.probes`` and ``Basis.interpolator`` are
experimental for affine simplex P0/P1/P2 fields; composite fields, curved
geometry, and tensor-product cells remain unsupported.  ``MeshTri.refined``
supports uniform and conforming adaptive refinement for linear triangle meshes.
``MeshTet.refined`` supports uniform and conforming adaptive refinement for
linear tetrahedron meshes and carries named boundaries and subdomains forward.
``MeshQuad.refined`` supports uniform refinement for linear quadrilateral
meshes and preserves named boundaries and subdomains.
``skfemntv.helpers.curl`` experimentally matches the upstream helper for
interpolated scalar 2D and vector 2D/3D fields, and supports curl-curl
bilinear forms on vector H1 spaces in 2D and 3D.  H(curl) element DOFs remain
unsupported.  The ``det``, ``inv``, ``eye``, and ``cross`` helpers also
experimentally match upstream for numeric arrays with component axes first;
symbolic nonlinear form expressions using these helpers are rejected because
the native form language currently describes linear/bilinear operations.
``Mesh.load`` and
``Mesh.save`` use optional ``meshio`` support for selected common cell types.
Other mesh families' editing methods, several common helper operations, and
element families remain gaps.  ``MeshTri``, ``MeshQuad``, and ``MeshTet`` support
``restrict`` and ``remove_elements`` with compacted vertices and remapped
named regions.  The public
``condense``, ``enforce``, ``penalize``, and ``solve`` functions follow the
scikit-fem call signatures for sparse linear systems and generalized
eigenvalue problems.  The internal
``notes/skfem-parity-audit-20261015.md`` records the detailed inventory and
recommended order.

Operations kept outside the backend
-----------------------------------

Solver conveniences and application policy—including ``solve``, ``enforce``,
``penalize``, nonlinear material updates, contact policy, and time stepping—do
not belong to the native assembly backend.  Applications may continue to use
scikit-fem, SciPy, or their own solver layer for these operations.

Validation
----------

The :doc:`gallery` runs selected scikit-fem examples through equivalent
problem definitions and compares assembled operators and final results.  The
:doc:`scikit-fem-performance` page documents performance measurements
separately from numerical compatibility.
