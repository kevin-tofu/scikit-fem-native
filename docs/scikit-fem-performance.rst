scikit-fem assembly performance comparison
===========================================

.. meta::
   :description: Reproducible scikit-fem and scikit-fem-native assembly performance comparison, including benchmark scope, environment, timing method, and recorded results.

``scikit-fem-native`` moves reusable finite-element integration and sparse-scatter
kernels into native code.  Whether this improves an application depends on the
mesh size, form, quadrature, hardware, thread configuration, and the amount of
work performed outside assembly.  The project therefore publishes a
reproducible comparison rather than a universal speed claim.

Reference benchmark
-------------------

The reference benchmark compares warm-cache Poisson P1 matrix and
right-hand-side assembly using equivalent meshes and spaces in scikit-fem and
scikit-fem-native.  Setup and assembly phases are reported explicitly so that
one-time construction cost is not hidden inside repeated assembly timing.

The
`benchmark methodology and driver <https://github.com/kevin-tofu/scikit-fem-native/tree/main/benchmarks/compare-with-skfem>`_,
`recorded environment and values <https://github.com/kevin-tofu/scikit-fem-native/blob/main/benchmarks/compare-with-skfem/results/poisson-linux-x86_64.md>`_,
and
`scaling plot <https://github.com/kevin-tofu/scikit-fem-native/blob/main/benchmarks/compare-with-skfem/results/poisson-linux-x86_64.png>`_
are versioned with the implementation.

Interpreting results
--------------------

Treat the recorded numbers as one reference environment, not a performance
guarantee.  For a meaningful local decision, use the same package versions,
record CPU and thread settings, test representative meshes and coefficient
updates, and include end-to-end solver time when assembly is not the dominant
cost.  Numerical agreement should be checked independently through the
:doc:`scikit-fem-compatibility` boundary.
