# skfem-native

`skfem-native` (`import skfn`) is a compact C++ finite-element assembly engine
with a Python API.  Its shared H1 assembly core supports Tet4 and Hex8,
including multiple quadrature points and non-affine Hex8 geometry, and exposes
a reusable SciPy CSR matrix.

```python
from skfn import LinearElasticity, NativeAssembler

assembler = NativeAssembler(
    coordinates, connectivity, element_dofs,
    LinearElasticity(young_modulus=210e9, poisson_ratio=0.3),
)
out = assembler.evaluate(u)
print(out.residual, out.tangent)
```

With the independent scikit-fem-style API:

```python
basis = Basis(mesh, ElementVector(ElementTetP1()))
assembler = NativeAssembler.from_basis(
    basis, LinearElasticity(young_modulus=210e9, poisson_ratio=0.3)
)
```

Compressible Neo-Hookean assembly uses Lamé parameters and returns an
analytically consistent tangent:

```python
from skfn import NeoHookean

kernel = NeoHookean.from_young_poisson(young_modulus=100.0, poisson_ratio=0.3)
assembler = NativeAssembler.from_basis(basis, kernel)
out = assembler.assemble(u, state=None, loads=external_force)
```

See `examples/neo_hookean_tet4.py` for a complete Newton solve.

Quadratic nodal H1 elements remain available through the tabulated-basis
assembly core while their independent mesh and mapping API is being completed.

The public form API is intentionally source-compatible with scikit-fem:

```python
import numpy as np
import skfn as skfem
from skfn.helpers import dot

basis = skfem.Basis(
    skfem.MeshTet(),
    skfem.ElementVector(skfem.ElementTetP1()),
)

@skfem.LinearForm
def body_force(v, w):
    return dot(w.force, v)

force = np.array([0.0, 0.0, -1.0])[:, None, None]
rhs = skfem.asm(body_force, basis, force=force)
```

Physical quadrature coordinates are available as `w.x`.  `FacetBasis` also
provides its outward unit normal as `w.n`; both may participate in NumPy
expressions evaluated before native assembly:

```python
def load(x):
    return np.stack((1.0 + x[0], x[1] ** 2, -0.5 * x[2]))

@skfem.LinearForm
def varying_load(v, w):
    return dot(load(w.x), v)

@skfem.LinearForm
def pressure_normal(v, w):
    return dot(w.n, v)

@skfem.BilinearForm
def weighted_mass(u, v, w):
    return (1.0 + w.x[0] ** 2) * dot(u, v)
```

Supported value and gradient contractions are dispatched to native assembly.
Unsupported forms raise `skfn.UnsupportedNativeForm`; `skfn.asm` never silently
falls back to Python assembly.  Use the upstream `skfem.asm` explicitly when a
reference fallback is desired.  The same native API works with `FacetBasis`
for surface tractions.

Nonmatching coplanar P1 triangle surfaces can be coupled through a reusable
supermesh:

```python
supermesh = skfem.TriangleSupermesh(
    master_points,
    master_triangles,
    slave_points,
    slave_triangles,
    components=3,
)
coupling = supermesh.assemble()
```

Triangle intersections and overlap quadrature are generated once.  Repeated
coupling assembly reuses a rectangular CSR pattern and runs the cross-basis
quadrature and scatter in C++.

Master and slave spaces may have different component counts.  For example, a
scalar multiplier can be coupled to a three-component displacement with a
component tensor:

```python
supermesh = skfem.TriangleSupermesh.from_facets(
    multiplier_facet_basis,
    displacement_facet_basis,
)

# Shape: (..., multiplier components, displacement components).
# Leading entity/quadrature axes are broadcast automatically.
coupling = supermesh.assemble_tensor(np.array([[[1.0, 2.0, 3.0]]]))
```

Tensor coefficients need not be symmetric.  Reversing the master and slave
spaces and transposing the component axes produces the transpose coupling
matrix.

Cross-basis value and full physical-gradient contractions share the same
native fixed-CSR assembler:

```python
# coefficient axes:
# (row component, column component, column spatial direction)
value_gradient = supermesh.assemble_cross(
    coefficient,
    row_kind="value",
    column_kind="gradient",
)

gradient_gradient = supermesh.assemble_cross(
    diffusivity,
    row_kind="gradient",
    column_kind="gradient",
)
```

All four value/value, value/gradient, gradient/value, and gradient/gradient
combinations are supported.  Mixed value-gradient contractions require a
tensor coefficient because the spatial direction must be explicit.  A scalar
gradient-gradient coefficient contracts matching component and spatial axes.
The same contractions are available through interface forms:

```python
from skfn.helpers import avg, ddot, dot, grad, jump

@skfem.BilinearForm
def gradient_jump(u, v, w):
    return w.kappa * ddot(jump(grad(u)), jump(grad(v)))

@skfem.BilinearForm
def directional_flux(u, v, w):
    return dot(jump(v), dot(w.beta, avg(grad(u))))
```

The dependency-free contraction loop is isolated behind C++ basis and
coefficient views in `cross_contraction.hpp`.  This keeps the public and
assembly APIs stable if a specialized tensor/SIMD backend is selected later.

For curved Tet10/Hex27 facets, search geometry is refined adaptively:

```python
supermesh = skfem.TriangleSupermesh.from_facets(
    master_basis,
    slave_basis,
    geometry_tolerance=1e-4,
    max_subdivision_level=6,
    projection_tolerance=1e-6,
)
print(supermesh.diagnostics)
```

The chord-error tolerance controls only reusable intersection geometry; parent
high-order shape values and gradients are still evaluated isoparametrically at
every generated overlap quadrature point.

`TriangleSupermesh.from_facets(master, slave)` accepts independent Tet4,
Tet10, Hex8, and Hex27 `FacetBasis` objects.  Search geometry is triangulated,
while shape values at overlap quadrature points are evaluated from the full
parent isoparametric element; high-order coupling is therefore scattered
directly to the original element DOFs.

Interface forms remain formulation-neutral:

```python
from skfn.helpers import avg, dot, jump, normal_grad

@skfem.BilinearForm
def interface_form(u, v, w):
    return (
        w.a * dot(jump(u), jump(v))
        + w.b * dot(avg(normal_grad(u)), jump(v))
    )

matrix = skfem.asm(
    interface_form,
    master_basis,
    slave_basis,
    integration=supermesh,
    a=penalty_like_coefficient,
    b=flux_like_coefficient,
)
```

Interface `LinearForm` uses the same reusable supermesh quadrature and returns
one vector containing the master DOFs followed by the slave DOFs:

```python
@skfem.LinearForm
def interface_load(v, w):
    return dot(w.traction, jump(v))

rhs = skfem.asm(
    interface_load,
    master_basis,
    slave_basis,
    integration=supermesh,
    traction=traction,
)
```

`avg(v)`, `jump(grad(v))`, `avg(grad(v))`, and normal-gradient traces are
assembled natively as well.  The weights supplied by `jump` produce equal and
opposite master/slave resultants for a constant interface traction.

Supermesh forms receive geometry evaluated at every overlap quadrature point:

```python
@skfem.LinearForm
def varying_interface_load(v, w):
    return dot(load(w.x), jump(v))

@skfem.LinearForm
def master_pressure(v, w):
    return dot(w.n_master, jump(v))

@skfem.BilinearForm
def gap_weighted_penalty(u, v, w):
    return (1.0 + w.gap ** 2) * dot(jump(u), jump(v))
```

`w.n_master` and `w.n_slave` are the independent outward unit normals;
`w.gap` is the signed search-geometry separation measured along the master
normal.  Curved Tet10 and Hex27 normals are evaluated from the original
isoparametric facets rather than copied from the tessellated search triangles.

User parameters, callables, and geometry values share one numerical
quadrature-expression context:

```python
def pressure_law(x, gap, scale):
    return scale * (1.0 + x[0] + gap ** 2)

@skfem.LinearForm
def nonlinear_pressure(v, w):
    pressure = w.pressure_law(w.x, w.gap, w.scale)
    return dot(pressure * w.n_master, jump(v))

rhs = skfem.asm(
    nonlinear_pressure,
    master_basis,
    slave_basis,
    integration=supermesh,
    pressure_law=pressure_law,
    scale=1.7,
)
```

NumPy ufuncs are supported as coefficient expressions, so scalar, vector, and
tensor parameters can be combined with coordinates, normals, and gap before
the resulting contiguous coefficient is passed to native integration.

Volume and facet bilinear forms may contain multiple value and gradient terms:

```python
@skfem.BilinearForm
def reaction_diffusion(u, v, w):
    return (
        w.reaction * (1.0 + w.x[0] ** 2) * dot(u, v)
        + w.diffusivity * np.exp(-w.x[1])
          * ddot(grad(u), grad(v))
    )
```

Terms of the same kind are added as quadrature coefficients.  The combined
value and gradient coefficients are then assembled in one native traversal,
with one CSR zeroing and scatter pass.

Composite nodal H1 fields use the same expanded signature as scikit-fem:

```python
element = skfem.ElementTetP1() * skfem.ElementTetP1()
basis = skfem.Basis(mesh, element)

@skfem.BilinearForm
def coupled(u1, u2, v1, v2, w):
    return (
        (1.0 + w.x[0]) * u1 * v1
        + 2.0 * u2 * v2
        + 0.3 * u2 * v1
        - 0.4 * u1 * v2
        + w.diffusion * dot(grad(u1), grad(v1))
    )
```

`ElementComposite` interleaves its nodal subfield DOFs and caches each native
rectangular block assembler.  It supports component-compatible value and
gradient contractions as well as mixed-order vector-scalar divergence blocks
using the usual mixed-form notation:

```python
element = (
    skfem.ElementVector(skfem.ElementTetP2())
    * skfem.ElementTetP1()
)
basis = skfem.Basis(mesh, element, intorder=4)

@skfem.BilinearForm
def mixed(u, p, v, q, w):
    return (
        w.mu * ddot(grad(u), grad(v))
        - p * div(v)
        - q * div(u)
    )
```

Both `p * div(v)` and `q * div(u)` are assembled as native rectangular
gradient-value blocks; no Python element loop or runtime scikit-fem fallback
is used.  Mixed-order nodal fields share one geometry and quadrature context;
for example, the Taylor–Hood pair above places velocity DOFs on Tet10 vertices
and edges while pressure DOFs remain on its four vertices.

`skfn` only traces and assembles the requested jump, weighted average, value,
and outward-normal-gradient contractions; it does not select a Nitsche,
mortar, or contact formulation.

Build and test with:

```sh
python -m pip install -e '.[test]'
pytest
```
