scikit-fem compatibility and supported features
================================================

.. meta::
   :description: Learn which scikit-fem workflows, DOF ordering conventions, meshes, elements, and assembly operations are supported by skfem-native and skfemntv.

``skfem-native`` follows selected public conventions from
`scikit-fem <https://scikit-fem.readthedocs.io/>`_ so applications can retain
readable Python weak forms while choosing native assembly where the supported
subsets overlap.  It is an independent project and is not an official
scikit-fem distribution or backend.

Backend selection
-----------------

The installed distribution is ``skfem-native`` and its import package is
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
