# Stabilization checkpoint before H(curl)

This checkpoint freezes the current H1-oriented mathematical boundary before
edge-owned degrees of freedom are designed.  It is intentionally a stability
milestone, not a claim that the finite-element feature set is complete.

## Supported at this checkpoint

- H1 nodal scalar and vector spaces on the declared mesh/element families;
- value, physical-gradient, symmetric-gradient, and divergence contractions;
- scalar anisotropic diffusion with component-first tensor coefficients;
- first-axis coefficient component selection and multiple named fields;
- linear, bilinear, cross-basis, composite, facet, and interface assembly;
- cut-cell and implicit-interface workflows covered by the capability registry;
- conservative assembly-memory preflight and optional budget enforcement.

The machine-readable source of truth remains `capabilities.py`.  This note is a
review guide; new capability claims must be added to the registry and tested.

## Explicitly not supported

- H(curl), Nédélec elements, oriented edge DOFs, and covariant Piola mapping;
- H(div), oriented facet DOFs, and contravariant Piola mapping;
- general symbolic `outer`, `transpose`, symmetric, or skew tensor algebra;
- arbitrary physical-point location/evaluation;
- an implicit fallback from native forms to upstream scikit-fem assembly.

Unsupported mathematics must fail explicitly.  It must not be approximated by
an H1 operation with a superficially similar array shape.

## Architecture boundary

The typed-form implementation now has an acyclic dependency direction:

```text
errors / coefficients
        ↓
form terms / H1 fields / interface terms
        ↓
composite fields / form parameters
        ↓
form compiler
        ↓
forms.py dispatcher
        ↓
native linear and bilinear assemblers
```

`test_form_architecture.py` checks that the internal typed-form modules never
import the `forms.py` dispatcher, that their internal import graph is acyclic,
and that private expression nodes do not leak through `skfemntv.__all__`.

## Representative regression set

| Mathematical area | Primary regression file |
|---|---|
| scalar/vector H1 and elasticity | `test_compatible_forms.py` |
| mixed and Taylor-Hood forms | `test_composite_forms.py` |
| anisotropic coefficients/layouts | `test_anisotropic_forms.py` |
| coefficient components/diagnostics | `test_coefficient_components.py` |
| cross-basis assembly | `test_cross_basis_assembly.py` |
| jump, average, Nitsche and interface data | `test_interface_forms.py` |
| memory estimates and enforcement | `test_memory_preflight.py` |
| capability contract | `test_capabilities.py` |
| internal dependency/public API boundary | `test_form_architecture.py` |

Run the representative set during form work and the complete test directory
before declaring a checkpoint complete.

## Entry conditions for H(curl) implementation

Before implementation, a design note must specify global edge numbering,
element-local edge orientation signs, Nédélec basis functionals, covariant
Piola transformation, physical curl transformation, boundary-edge selection,
and independent mass/curl-curl comparisons with scikit-fem.  H(curl) should
enter through new field and mapping types rather than conditionals added to the
current H1 symbolic nodes.

The topology-only part of this condition is now specified in
`20260812-hcurl-edge-topology-design.md` and guarded by
`test_edge_topology.py`.  This does not change the planned status of H(curl) or
edge-owned DOFs.
