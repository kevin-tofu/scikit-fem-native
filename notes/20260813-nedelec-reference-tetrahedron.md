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
