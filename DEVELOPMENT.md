# Development philosophy

## Purpose

scikit-fem makes finite-element formulations easy to express, inspect, and
test in Python.  `skfem-native` improves the numerical assembly part of that
workflow without moving application mathematics into a compiled framework.

This distinction matters increasingly in AI-assisted development.  Nonlinear
constitutive laws, coupled weak forms, and application-specific contact
algorithms can be written and revised rapidly when they remain ordinary Python
components.  The compiled layer should accelerate stable, reusable numerical
operations rather than become the owner of every material or formulation.

The project therefore aims to preserve two properties:

- application code remains readable Python close to the mathematics
- performance-critical, reusable assembly machinery runs in native code

## Ownership boundary

`skfemntv` owns:

- meshes, elements, bases, tabulation, and quadrature
- generic functional, linear, bilinear, and cross-bilinear assembly
- assembly from caller-supplied tensor coefficients
- sparse scatter and threaded native kernels
- cut-cell and implicit-interface quadrature
- interface supermeshes and contact-facet search
- geometry and integration diagnostics

The calling application owns:

- weak forms and coefficient fields
- constitutive updates and state histories
- contact and interface formulations
- multiplier spaces, stabilization, and algebraic reduction
- nonlinear stepping, linear solvers, and convergence policy

For example, an application can define one nonlinear material update and pass
the resulting coefficient and tangent tensors through either `skfem.asm` or
`skfemntv.asm`.  Likewise, `skfemntv` may construct shared interface geometry,
but the application decides whether that geometry is used for Mortar, dual,
Nitsche, or another contact formulation.

## Design principles

1. Keep the public assembly call shape close to scikit-fem.
2. Keep formulation-specific policy out of the backend.
3. Prefer generic coefficient and tensor assembly over material-specific APIs.
4. Make backend selection explicit and test numerical equivalence.
5. Keep diagnostics available at Python boundaries.
6. Remove misleading abstractions rather than preserving accidental APIs.

## Public degree-of-freedom ordering

Supported backends must expose the same global degree-of-freedom ordering as
scikit-fem.  The canonical order groups DOFs by topological entity: vertices,
then shared edges or facets, then element interiors.  Local native kernels may
use explicit high-order connectivity, but that representation must not leak as
a different public vector or matrix ordering.

Entity-based numbering is preferred over assigning new nodes while traversing
elements because it is independent of first encounter, keeps boundary and
interior blocks identifiable, supports direct backend interchange, and follows
the decomposition used by scikit-fem's `Dofs` implementation.

Gallery compatibility tests must compare public matrices and vectors directly.
A coordinate permutation is useful for diagnosing an ordering defect, but is
not an acceptable permanent compatibility layer for a supported element.

## Compatibility

`skfemntv` is a selectable backend, not a promise of complete scikit-fem API
coverage.  Shared call shapes should produce equivalent numerical results, but
applications should not rely on implicit backend substitution.

Version 0.2 intentionally removed formulation-specific 0.1 APIs such as native
material models, Mortar reduction policy, KKT types, and specialized Nitsche
assembly.  Those responsibilities belong to the application layer.
