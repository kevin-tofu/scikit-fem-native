# Experimental H(curl) public API review

## Decision

Expose three names only:

```text
AffineTriN1Basis
TriN1Assembler
estimate_tri_n1_assembly_memory
```

The feature remains experimental and separate from general `Basis` and `asm`.

## Changes made during review

### Assembler name

The first candidate name was `NativeTriN1Assembler`.  The implementation
currently performs element integration with NumPy and sparse pattern/scatter
work with SciPy/Python, not a native C++ kernel.  The `Native` prefix would
therefore make a performance and implementation claim that is not true.  The
public name is `TriN1Assembler`.  A future native implementation can preserve
that implementation-neutral name.

### Public and internal array layouts

Public mapped tabulation follows the established component-first convention:

```text
basis.values: (local_basis, component, cell, quadrature)
basis.curls:  (local_basis, cell, quadrature)
basis.dx:     (cell, quadrature)
```

The assembler keeps private cell-first contiguous arrays because each element
matrix consumes all local basis values for one cell.  The public properties are
views with axes moved into the scikit-fem-oriented convention.  Public layout
and kernel/storage layout are distinct intentionally, as with anisotropic H1
coefficients.

### CSR result ownership

All assembly methods overwrite and return the same CSR matrix object.  This is
consistent with reusable assemblers but can surprise a caller retaining two
results.  The class docstring and README require `.copy()` when a matrix must
survive another call.  A contract test fixes reuse of the matrix, data,
indices, and indptr objects.

### Coefficients

Mass and curl coefficients accept constants or arrays broadcasting to
`(cell, quadrature)`.  They do not currently accept callables.  This is kept
explicit rather than inventing a second parameter-evaluation API beside typed
forms.  The example uses constant coefficients; quadrature-dependent arrays
are independently compared with coordinate functions evaluated by scikit-fem.

## Deferred decisions

- integration into general `Basis` and symbolic `asm`;
- a native C++ scatter/integration kernel;
- linear forms and field interpolation;
- public topology/reference-element objects;
- tetrahedral and curved H(curl) elements.

Promotion of broad `space.hcurl` requires these concerns to be designed rather
than hidden behind the experimental triangle-specific API.
