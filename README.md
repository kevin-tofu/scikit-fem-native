# skfem-native 0.2

`skfemntv` is a native numerical assembly backend for finite-element
applications.  Version 0.2 deliberately does not own application formulations,
material models, contact laws, or nonlinear solution policies.

## Ownership boundary

The calling application owns:

- weak forms and coefficient fields
- constitutive updates and state histories
- contact and interface formulations
- multiplier spaces and algebraic reduction
- stabilization, solver, and load-step policies

`skfemntv` owns:

- meshes, elements, bases, and tabulation
- quadrature and geometry diagnostics
- generic functional, linear, bilinear, and cross-bilinear assembly
- caller-supplied tensor coefficient assembly
- cut-cell and implicit-interface quadrature
- interface supermesh construction and contact-facet search
- sparse scatter and threaded native kernels

## Assembly API

Forms use the same call shape as scikit-fem:

```python
import skfemntv
from skfemntv.helpers import dot, grad

mesh = skfemntv.MeshTet()
basis = skfemntv.Basis(mesh, skfemntv.ElementTetP1())

@skfemntv.BilinearForm
def diffusion(u, v, w):
    return dot(grad(u), grad(v))

matrix = skfemntv.asm(diffusion, basis)
```

Applications with pretabulated coefficient tensors can use
`NativeLinearForm`, `NativeBilinearForm`, `NativeCrossBilinearForm`, and
`NativeCrossBilinearForm.assemble_tensor()`.

Nonmatching interfaces use `TriangleSupermesh` or `InterfaceSupermesh` for
geometry and shared quadrature.  The application supplies its own test-space
shape values through `CrossTabulation`; skfemntv does not select Mortar, dual,
Nitsche, or contact formulations.

## Removed in 0.2

The following 0.1 APIs were removed rather than deprecated:

- `NativeAssembler`
- `MaterialAssembler`
- `J2Plasticity` and J2 state/history APIs
- `StandardLinearSolid`
- `LinearElasticity` and element-specific aliases
- `NeoHookean` and element-specific aliases
- `TriangleSupermesh.assemble_mortar()`
- Mortar result, metadata, reduction, and KKT types
- `assemble_symmetric_nitsche()`

These belong in the application layer.  kktkit, for example, defines one J2
return mapping and sends the resulting stress and tangent tensors to either the
scikit-fem or skfemntv assembly backend.

## Development

```bash
python -m pip install -e .
pytest -q
```

The project version is `0.2.0`.
