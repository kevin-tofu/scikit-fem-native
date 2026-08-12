# H(curl) phase 1: reference triangle Nédélec basis

## Scope

This phase implements only the lowest-order first-family Nédélec basis on the
unit reference triangle.  It has no physical mapping, global edge DOFs, or
assembly.  Therefore `space.hcurl` and `dof.edge` remain planned, while
`element.tri_n1_reference` is experimental.

## Reference geometry and orientation

The reference vertices are

```text
v0 = (0, 0), v1 = (1, 0), v2 = (0, 1)
```

and the directed local edges follow the topology contract:

```text
e0 = (0 -> 1), e1 = (1 -> 2), e2 = (2 -> 0).
```

With barycentric coordinates

```text
lambda0 = 1 - x - y, lambda1 = x, lambda2 = y,
```

the basis on edge `(i -> j)` is the Whitney form

```text
Nij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i).
```

This gives

```text
N01 = (1 - y, x)
N12 = (-y, x)
N20 = (-y, x - 1)
```

and every scalar reference curl is `+2` under the convention
`curl(u) = partial_x(u_y) - partial_y(u_x)`.

## Edge degrees of freedom

The reference functional is the directed tangential line integral

```text
L_e(u) = integral_e u dot dx.
```

No unit-tangent/edge-length split is required: parameterizing the directed
edge directly supplies `dx`.  The basis is dual to these moments, so

```text
L_ei(Nj) = delta_ij.
```

Reversing an edge reverses the corresponding functional row and changes no
other row.  This is the reference-level counterpart of multiplying a local
edge basis/DOF by `element_edge_signs` on a physical mesh.

## Relationship to scikit-fem

scikit-fem `ElementTriN1` uses the same polynomial space but its local basis
orientation differs from the directed-edge convention above.  Pointwise
values and curls agree after applying

```text
(-1, -1, +1)
```

to scikit-fem basis indices `(0, 1, 2)`.  The comparison test records this
explicitly; neither implementation is silently relabeled.

## Discrete local/global orientation layer

`OrientedEdgeDofMap` now connects the reference convention to
`OrientedEdgeTopology` without adding a physical mapping.  For one scalar
global moment `g_E` on edge `E`, its coefficient in a cell-local directed
basis is

```text
c_(e,K) = element_edge_signs[e,K] * g_E.
```

Converting that local coefficient back to a global-oriented moment applies the
same sign.  Since every sign is `+1` or `-1`, the two signs cancel.  Tests now
establish:

- one global DOF per global edge;
- local-to-global DOF maps equal `element_edges`;
- local basis signs equal `element_edge_signs`;
- both cells sharing an edge recover the same global-oriented moment;
- cell vertex reordering leaves the represented global edge data invariant.

This is an algebraic orientation map, not yet `dof.edge`: no finite-element
space owns these DOFs and no physical tangential trace is evaluated.  The
narrow `dof.edge_orientation_map` capability is therefore experimental while
`dof.edge` remains planned.

## Affine covariant Piola mapping

The reference basis can now be mapped to an affine physical triangle with

```text
u(x) = J^(-T) u_hat(X),
curl(u)(x) = curl(u_hat)(X) / det(J).
```

Jacobians use component-first shape `(physical_dim, reference_dim, ...)`.
The determinant is signed: reversing the physical triangle orientation changes
the mapped scalar curl sign.  Tests verify tangential line-integral invariance
directly and compare mapped values/curls with scikit-fem `ElementTriN1`.

This primitive is recorded as `mapping.covariant_piola_tri_affine`.  The broad
`mapping.covariant_piola` capability remains planned because the mapping is not
yet integrated into `Basis`, global interpolation, or assembly.

## Minimal affine-triangle basis

`AffineTriN1Basis` now combines triangle quadrature, reference tabulation,
affine Piola mapping, `OrientedEdgeDofMap`, and physical integration weights.
Its stored layouts are explicit:

```text
public values: (local_basis, component, cell, quadrature)
public curls:  (local_basis, cell, quadrature)
dx:     (cell, quadrature)
```

Orientation signs are applied after Piola mapping, and `dx` uses
`abs(det(J))` while the mapped curl retains signed `1 / det(J)`.  Local mass
and curl-curl matrices are integrated from this data and, after matching edge
IDs by their global vertex pairs, agree with scikit-fem assembly.  Reordering
cell vertices preserves both assembled operators.

The narrow `space.hcurl_tri_n1_basis` capability is experimental.  The broad
`space.hcurl` capability remains planned because this basis is not accepted by
the public `Basis`, `asm`, interpolation, boundary-DOF, or solver workflows.

## Dedicated sparse assembler

`TriN1Assembler` now consumes the verified basis data and exposes
explicit `assemble_mass`, `assemble_curl_curl`, and `assemble_maxwell` methods.
Scalar constants and `(cell, quadrature)` coefficient fields are supported.
Its CSR pattern, matrix object, and local-entry scatter map are constructed
once and reused across repeated assembly calls.

The dedicated memory estimate includes retained basis arrays, an upper bound
for CSR storage, the local-entry scatter map, and temporary COO pattern arrays.
The standard memory-budget guard is applied before the sparse pattern is
constructed.  Constant and coordinate-dependent operators agree with
scikit-fem after matching edge IDs by vertex pair.

This is `assembly.hcurl_tri_n1`, marked experimental.  It remains intentionally
separate from public `asm` and from the typed H1 form dispatcher.

## Boundary edge DOFs and constrained solve

`AffineTriN1Basis.boundary_dofs` now selects all boundary edges, named mesh
boundaries, boundary-center predicates, explicit facet IDs, or a union of
named boundaries.  Results are unique global edge DOFs, so overlap is removed
when boundary regions meet.  Interior facet IDs are rejected rather than
silently converted to constraints.

The selection agrees with scikit-fem `ElementTriN1.get_dofs`.  A constrained
Maxwell-like problem using the dedicated mass plus curl-curl assembler and
SciPy `spsolve` agrees with a solve using the independently assembled
scikit-fem matrix.  Boundary coefficients remain exactly zero and the free
residual is checked.

This narrow feature is `dof.edge_boundary_tri`, marked experimental.  Solver
policy remains external and the general `dof.edge` capability remains planned
until the H(curl) basis enters the public space API.

## Experimental public API

The reviewed vertical slice exposes exactly three package-level names:

```text
AffineTriN1Basis
TriN1Assembler
estimate_tri_n1_assembly_memory
```

Reference formulas, Piola primitives, topology construction, and expression
internals remain implementation details.  The public README and
`examples/hcurl_tri_n1_maxwell.py` state the affine-triangle, lowest-order-only
boundary explicitly and keep SciPy solver policy outside the package.

## Next phase

Pause implementation and collect review feedback on naming, layouts, and the
dedicated API.  Before promoting broad `space.hcurl`, the project still needs
field interpolation, evaluation, a linear load path, convergence validation,
and a decision on integration with general form syntax.  Tetrahedra and curved
geometry should remain separate later milestones.
