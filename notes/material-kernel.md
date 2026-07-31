# Native material kernel contract

Stateful constitutive laws are compiled C++ kernels selected once when a
`MaterialAssembler` is constructed.  The quadrature loop does not invoke a
Python callback or a virtual function.

A small-strain kernel declares its material-point state and exposes an update
equivalent to:

```cpp
struct MaterialKernel {
    static constexpr int state_size = /* fixed for this material */;
    using Result = /* stress */;

    Result update(
        const double* strain,
        const double* committed_state,
        double* trial_state,
        double* tangent_or_null,
        double evaluation_time_step
    ) const;
};
```

Passing a null tangent pointer is the residual-only path and must avoid
computing the algorithmic tangent.  The output state buffer always receives a
trial state.  Assembly never mutates the committed state; the caller commits
the trial only after accepting the nonlinear iteration or load step.

The global assembler stores committed and trial states as contiguous
`(integration_points, state_size)` arrays and does not know any J2 field
names.  Python state classes expose named zero-copy views where useful.

`J2MaterialKernel` is the first stateful implementation.  Its seven state scalars are
the six tensor-shear plastic-strain components and accumulated equivalent
plastic strain.  Its update is statically dispatched by the templated global
assembler, and its tangent is written directly into the caller's buffer to
avoid an extra per-integration-point copy.

The state-free linear-elastic kernel uses the same assembler template with
`state_size == 0`; its result is checked against the established native
elastic assembler.  This verifies that the global path is not tied to the J2
state layout.

Python exposes the common construction shape:

```python
assembler = skfn.MaterialAssembler(basis, material)
state = assembler.initial_state()
evaluation = assembler.assemble(u, state, num_threads=4)
state = evaluation.trial_state  # explicit commit
```

`J2Assembler` remains an alias for compatibility.  Adding another material
requires a compiled kernel, state adapter, Python material value object, and a
construction-time dispatch branch.  It does not require changing the public
assembly workflow.  Runtime Python material callbacks are intentionally not
part of this high-performance interface; user-defined Python forms remain the
flexible path when native fused performance is not required.

Implementation ownership is separated as follows:

- `material_assembler.hpp`: generic strain, state, element, coloring, and CSR
  assembly logic;
- `material_kernel.hpp`: compiled constitutive kernel contracts;
- `cpp/src/material/j2.cpp`: J2 material-point and Python construction binding;
- `cpp/src/material/linear_elastic.cpp`: state-free reference binding.
- `cpp/src/material/standard_linear_solid.cpp`: six-state viscoelastic
  material and binding.

The currently instantiated state sizes are zero (linear elasticity), six
(Standard Linear Solid viscous strain), and seven (J2 plasticity).

The generic assembler is exercised with 12, 30, 18, 15, 24, and 81 local vector
degrees of freedom corresponding to Tet4, Tet10, Wedge6, Pyramid5, Hex8, and
Hex27.  High-order and non-tensor-product tests include curved geometry and
integration-order sweeps, so material
kernels do not need topology- or order-specific branches.

Evaluation context such as a time-step override is passed through the generic
assembler into every kernel update.  Kernels that do not need it ignore it.
The Standard Linear Solid uses a positive override when supplied and otherwise
uses its construction-time default.  Changing the time step therefore does
not rebuild sparsity, scatter maps, coloring, or geometry.
