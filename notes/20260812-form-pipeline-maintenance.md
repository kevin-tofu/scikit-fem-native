# Typed form pipeline and maintenance boundary

This note is the map to the native form implementation.  New mathematical
operations should not be added until their place in this pipeline is clear.

## Pipeline

| Stage | Responsibility | Location |
|---|---|---|
| User expression | `dot`, `ddot`, `grad`, `div`, `mul` syntax | `helpers.py`, bottom of `forms.py` |
| Symbolic H1 fields | Trial/test values, gradients, symmetric gradients and divergence | `_h1_fields.py` |
| Composite H1 fields | Composite subfields, weighting and divergence coupling | `_composite_fields.py` |
| Lowered terms | Assembly-independent linear/bilinear term and sum records | `_form_terms.py` |
| Interface expressions | Trace transformations and interface-specific term records | `_interface_terms.py` |
| Parameter handling | Named coefficients, first-axis components, lookup and diagnostics | `_coefficients.py` |
| Trace parameters | Quadrature-value arithmetic and the form parameter namespace | `_form_parameters.py` |
| Shared failure contract | Unsupported typed-form exception | `_errors.py` |
| Form compilation | Call user functions with typed fields and normalize term results | `_form_compiler.py`, with form-specific validation in `forms.py` |
| Layout normalization | Convert public component-first arrays only at a kernel boundary | `_anisotropic_tensor_coefficient` and per-kind shape handling |
| Native assembly | Allocate/reuse sparse structures and invoke C++ kernels | `linear_form.py`, `bilinear_form.py` |

The public tensor convention and the native kernel convention are deliberately
different.  Public arrays retain scikit-fem's component-first axes.  Conversion
to entity/quadrature-first storage belongs at the final kernel boundary, where
quadrature-local tensors need contiguous trailing axes.

## Coefficient invariant

All assembly paths resolve coefficients through `_coefficients.py`.

- `Coefficient("material")` represents an omitted or not-yet-resolved field.
- `CoefficientComponent("material", 0)` selects the first public axis.
- legacy string descriptors are resolved by the same function while older
  expression nodes are retired gradually;
- missing, noninteger, and out-of-range diagnostics are defined once;
- assembly functions receive either `None` or an already numeric coefficient.

Do not add another `isinstance(coefficient, str)` branch to an assembler.

## Supported lowering

| Expression | Typed kind | Native path |
|---|---|---|
| `dot(u, v)` | `value` | component-wise value kernel |
| `ddot(grad(u), grad(v))` | `gradient` | component-wise gradient kernel |
| `ddot(sym_grad(u), sym_grad(v))` | `symmetric_gradient` | symmetric-gradient kernel |
| `div(u) * div(v)` | `divergence` | divergence kernel |
| `dot(mul(A, grad(u)), grad(v))` | `gradient_tensor` | explicit cross-contraction tensor kernel |
| coefficient with a test value/gradient | linear `value`/`gradient` | native linear kernel |
| composite field contractions | field-pair term | cached composite block assembler |
| jump/average trace contractions | interface term | interface integration assembler |

## Rules for the next refactor

1. Preserve typed expression nodes; do not fall back to arbitrary NumPy
   tracing for symbolic fields.
2. A new helper must have a precise mathematical lowering and an independent
   scikit-fem comparison test before becoming supported.
3. Keep coefficient lookup independent of expression-node classes.
4. Keep public-to-native axis movement visible and documented at the kernel
   boundary.
5. Symbolic H1 fields, composite fields, lowered terms, and interface traces
   now have separate modules.  Keep those boundaries before adding H(curl),
   edge orientation, or Piola mappings.  H(curl) introduces a new field and
   mapping model and should not be inserted into the current H1 dispatcher.

## Deferred work

`outer`, general `transpose`, symmetric/skew tensor constructors, and H(curl)
remain deferred.  Existing `sym_grad` is a dedicated typed operation, not a
general symbolic tensor algebra.  Symbolic H1, composite, interface, parameter
wrapping, and the common tracing boundary are now separate.  Form-specific
lowering remains beside assembly dispatch in `forms.py`; it should only move
when two or more paths share an identical validation rule.  Only after that
should the deferred helpers be designed.
