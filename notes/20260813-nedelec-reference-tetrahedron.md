# H(curl) tetrahedron phase 1: reference TetN1 element

## Scope

This phase adds only the lowest-order first-family Nedelec basis on the unit
reference tetrahedron.  Physical mapping, a tetrahedral H(curl) basis object,
boundary selection, and assembly remain follow-up work.  The capability is
therefore `element.tet_n1_reference`, not a claim of TetN1 workflow support.

## Directed edges and basis

The unit tetrahedron uses vertices `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, and
`(0,0,1)`.  Its directed edges follow the existing topology contract:

```text
(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
```

For edge `(i,j)`, the basis is the Whitney form

```text
Nij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i)
curl(Nij) = 2 grad(lambda_i) cross grad(lambda_j).
```

The tangential line moments form the 6-by-6 identity.  Reversing one directed
edge negates exactly its functional row.  scikit-fem uses `(0,2)` where this
project deliberately uses `(2,0)`, so comparison applies the explicit local
sign vector `(1,1,-1,1,1,1)`.

## Next gate

The affine three-dimensional covariant Piola mapping is now available.  Values
use `J^-T u_hat` and vector curls use `J curl(u_hat) / det(J)`.  Tests verify
directed tangential-moment invariance on all six edges and the vector-curl
formula on a nonsymmetric physical tetrahedron.  Singular Jacobians fail
explicitly.

The next gate is `AffineTetN1Basis`: tetrahedral quadrature, batched mapping,
orientation signs, geometry diagnostics, and element mass/curl-curl comparison
must be introduced together before adding sparse assembly.

## Minimal affine tetrahedron basis

`AffineTetN1Basis` now combines Duffy-product tetrahedral quadrature, native
global-edge topology, batched Piola mapping, orientation signs, and physical
integration weights.  Its private value and vector-curl arrays use
`(cell, local_basis, component, quadrature)` for contiguous element assembly;
the public views use `(local_basis, component, cell, quadrature)`.

Geometry diagnostics report signed and absolute determinants, minimum volume,
maximum edge-cubed/determinant aspect indicator, and inverted-cell count.  An
optional application-owned aspect threshold rejects excessive distortion.
Assembled element mass and vector curl-curl matrices agree with scikit-fem on
a multi-cell tetrahedral mesh after matching global edges by vertex pair.

Sparse assembly, loads, boundary-edge selection, interpolation, and convergence
remain outside this checkpoint.  The next implementation step is a reusable
TetN1 CSR assembler using the already-native generic edge pattern builder.

## Dedicated sparse assembler

`TetN1Assembler` now provides reusable mass, vector curl-curl, and combined
Maxwell assembly.  It shares only topology-neutral CSR ownership and scalar
coefficient handling with the triangle assembler; scalar-curl and vector-curl
contractions remain separate named implementation paths so their array meaning
is visible.  The native edge CSR builder accepts six local edge DOFs without a
tetrahedron-specific pattern implementation.

Constant operators agree with scikit-fem on a structured multi-cell mesh.
Quadrature-dependent scalar coefficients, stable CSR object reuse, type
diagnostics, and preflight budget rejection are tested.  Loads, boundary edge
selection, interpolation, and a constrained Maxwell solve remain follow-up
work.

## Boundary constraints, load, and solve

`AffineTetN1Basis.boundary_dofs` maps each selected triangular boundary facet
to its three globally owned edges.  All, named, predicate, explicit-facet, and
named-union selection agree with scikit-fem; interior facets are rejected.

`TetN1LinearAssembler` accepts callable or quadrature-array three-component
loads and reuses one result vector.  A constrained mass-plus-curl-curl solve
uses the dedicated matrix and load assemblers with external SciPy solver
policy.  Boundary coefficients remain zero, the free residual vanishes, and
the matrix, load, and solution agree with independently assembled scikit-fem
data after matching global edges by vertex pair.

This completes the minimal affine TetN1 solve path.  Interpolation, field
evaluation, convergence, orientation robustness, and performance measurement
remain the next validation layers before broadening the H(curl) capability.

## Interpolation and H(curl) convergence

`AffineTetN1Basis.interpolate_edge_moments` integrates callable vector fields
along every ascending global edge.  `evaluate` returns values and
`evaluate_curl` returns vector curls, both in
`(component, cell, quadrature)` layout.  A constant vector is reproduced to
roundoff with zero discrete curl.

For the quadratic field `u=(0,xz,xy)` with exact curl `(0,-y,z)`, uniform
tetrahedral refinement decreases the L2, curl-L2, and combined H(curl) errors
with measured successive rates `(1.08, 1.00, 1.04)` and
`(1.02, 1.00, 1.01)` in one local run.  This establishes the expected
first-order interpolation behavior independently of the constrained solve.
