Assembly model
==============

Backend selection
-----------------

Applications should select the assembly module explicitly at their backend
boundary.  Forms can retain the same general call shape while the application
chooses ``skfem.asm`` or ``skfemntv.asm``.

``skfemntv`` does not claim complete scikit-fem feature coverage.  Explicit
selection makes unsupported combinations visible and allows backend-equivalence
tests to remain precise.

Nonlinear coefficients
----------------------

Nonlinear material updates and history variables remain in application code.
The application computes coefficient or tangent tensors, then supplies them to
``NativeLinearForm``, ``NativeBilinearForm``, or
``NativeCrossBilinearForm`` for native integration and sparse assembly.

Interfaces
----------

``TriangleSupermesh`` and ``InterfaceSupermesh`` provide common interface
geometry and quadrature.  The application supplies its test and trial-space
tabulation through ``CrossTabulation`` and retains ownership of Mortar, dual,
Nitsche, or contact policy.
