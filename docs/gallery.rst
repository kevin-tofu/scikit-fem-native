Gallery compatibility
=====================

The initial compatibility gallery reproduces selected examples from the
`scikit-fem Gallery <https://scikit-fem.readthedocs.io/en/latest/listofexamples.html>`_.
Both backends receive the same mesh, finite-element space, boundary conditions,
and solver inputs.  The scripts compare assembled operators and final results.

Current coverage
----------------

- Example 1: two-dimensional Poisson equation
- Example 9: three-dimensional Poisson equation
- Example 19: transient heat equation

Run the suite from a source checkout:

.. code-block:: bash

   python examples/scikit-fem-gallery/run_all.py

The suite intentionally omits visualization and optional preconditioners.  Its
purpose is numerical backend equivalence, not screenshot reproduction.
