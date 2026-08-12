from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from weakref import WeakKeyDictionary

import numpy as np

from .linear_form import NativeCompositeLinearForm,NativeLinearForm
from .bilinear_form import (
    NativeBilinearForm,NativeCompositeBilinearForm,
    NativeCrossBilinearForm,
)
from ._skfn import integrate_functional
from ._coefficients import (
    Coefficient as _Coefficient,
    CoefficientComponent as _CoefficientComponent,
    evaluate_coefficient as _evaluate_coefficient,
    is_symbolic_coefficient as _is_symbolic_coefficient,
    resolve_coefficient as _resolve_coefficient,
)
from ._errors import UnsupportedNativeForm
from ._form_terms import (
    BilinearSum as _BilinearSum,
    BilinearTerm as _BilinearTerm,
    CompositeBilinearSum as _CompositeBilinearSum,
    CompositeBilinearTerm as _CompositeBilinearTerm,
    LinearSum as _Sum,
    LinearTerm as _Term,
)
from ._interface_terms import (
    InterfaceBilinearTerm as _InterfaceBilinearTerm,
    InterfaceCoefficientTrace as _InterfaceCoefficientTrace,
    InterfaceLinearSum as _InterfaceLinearSum,
    InterfaceLinearTerm as _InterfaceLinearTerm,
    InterfaceSum as _InterfaceSum,
    InterfaceTrace as _InterfaceTrace,
)
from ._h1_fields import (
    Divergence as _Divergence,
    SymmetricGradient as _SymmetricGradient,
    TensorGradient as _TensorGradient,
    TestGradient as _TestGradient,
    TestValue as _TestValue,
    TrialGradient as _TrialGradient,
    TrialValue as _TrialValue,
)
from ._composite_fields import (
    CompositeDivergence as _CompositeDivergence,
    CompositeField as _CompositeField,
    CompositeLinearSum as _CompositeLinearSum,
    CompositeLinearTerm as _CompositeLinearTerm,
    CompositeWeightedField as _CompositeWeightedField,
    composite_contraction as _composite_contraction,
    composite_divergence_contraction as _composite_divergence_contraction,
)
from ._form_parameters import (
    Parameters as _Parameters,
    QuadratureValue as _QuadratureValue,
    parameter_values as _parameter_values,
)
from ._form_compiler import (
    extract_terms as _extract_terms,
    trace_expression as _trace_expression,
)


class _LinearForm:
    def __init__(self, function: Callable):
        self.function = function
        self._native_cache = WeakKeyDictionary()

    def assemble(self, *bases, **kwargs):
        return asm(self, *bases, **kwargs)


class _Functional:
    def __init__(self,function: Callable):
        self.function=function

    def assemble(self,*bases,**kwargs):
        return asm(self,*bases,**kwargs)


class _BilinearForm:
    def __init__(self, function: Callable):
        self.function = function
        self._native_cache = WeakKeyDictionary()
        self._cross_cache = WeakKeyDictionary()

    def assemble(self, *bases, **kwargs):
        return asm(self, *bases, **kwargs)


def LinearForm(function=None, **kwargs):
    if kwargs:
        # Accept decorator syntax from the compatible subset.  Assembly
        # remains native; these options never trigger a fallback.
        return lambda fn: _LinearForm(fn)
    return _LinearForm(function) if function is not None else _LinearForm


def Functional(function=None,**kwargs):
    if kwargs:
        return lambda fn:_Functional(fn)
    return _Functional(function) if function is not None else _Functional


def BilinearForm(function=None, **kwargs):
    if kwargs:
        return lambda fn: _BilinearForm(fn)
    return _BilinearForm(function) if function is not None else _BilinearForm


def _native_functional_assemble(form,basis,kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    try:
        expression=form.function(
            _Parameters(_parameter_values(geometry,kwargs))
        )
        values=np.asarray(expression,dtype=np.float64)
    except Exception as error:
        raise UnsupportedNativeForm(
            f"Functional contains an unsupported operation: {error}"
        ) from error
    try:
        values=np.broadcast_to(values,basis.dx.shape)
    except ValueError as error:
        raise UnsupportedNativeForm(
            "Functional must evaluate to one scalar per quadrature point"
        ) from error
    return integrate_functional(
        np.ascontiguousarray(values),
        np.ascontiguousarray(basis.dx,dtype=np.float64),
    )


def _compile_linear(form: _LinearForm,basis,kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    expression=_trace_expression(
        form.function,(_TestValue(),),geometry,kwargs,
        context="LinearForm",
    )
    return _extract_terms(
        expression,_Term,_Sum,
        message=(
            "native LinearForm must reduce to dot(coefficient, v), "
            "ddot(coefficient, grad(v)), or a sum of those terms"
        ),
    )


def _native_linear_assemble(form,basis,kwargs,num_threads=0):
    terms = _compile_linear(form,basis,kwargs)
    native = form._native_cache.get(basis)
    if native is None:
        native = NativeLinearForm(basis)
        form._native_cache[basis] = native
    value = None
    gradient = None
    for term in terms:
        coefficient = _evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor
        )
        if term.kind == "value":
            if (
                native._shape[-1]==1
                and coefficient.shape==native._shape[:2]
            ):
                coefficient=coefficient[...,None]
            elif (
                coefficient.ndim
                and coefficient.shape[0] == native._shape[-1]
            ):
                coefficient = np.moveaxis(coefficient, 0, -1)
            value = coefficient if value is None else value + coefficient
        elif term.kind == "gradient":
            if (
                coefficient.ndim >= 2
                and coefficient.shape[:2] == native._gradient_shape[-2:]
            ):
                coefficient = np.moveaxis(coefficient, (0, 1), (-2, -1))
            gradient = (
                coefficient if gradient is None else gradient + coefficient
            )
        else:
            raise UnsupportedNativeForm
    result,_=native.assemble(
        value=value,gradient=gradient,num_threads=num_threads
    )
    if result.shape[0]==basis.N:
        return result
    padded=np.zeros(basis.N,dtype=result.dtype)
    padded[:result.shape[0]]=result
    return padded


def _native_composite_linear_assemble(form,basis,kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    test=tuple(
        _CompositeField("test",field)
        for field in range(len(basis.subbases))
    )
    expression=_trace_expression(
        form.function,test,geometry,kwargs,context="composite LinearForm"
    )
    if isinstance(expression,_CompositeWeightedField):
        expression=expression._linear_term()
    terms=(
        expression.terms
        if isinstance(expression,_CompositeLinearSum)
        else (expression,)
        if isinstance(expression,_CompositeLinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "composite LinearForm must reduce to subfield value or "
            "gradient contractions"
        )
    native=form._native_cache.get(basis)
    if native is None:
        native=NativeCompositeLinearForm(basis)
        form._native_cache[basis]=native
    grouped={}
    for term in terms:
        coefficient=_evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor
        )
        field_native=native.assembler(term.field)
        if term.kind=="value":
            if (
                field_native._shape[-1]==1
                and coefficient.shape==field_native._shape[:2]
            ):
                coefficient=coefficient[...,None]
            elif (
                coefficient.ndim
                and coefficient.shape[0]==field_native._shape[-1]
            ):
                coefficient=np.moveaxis(coefficient,0,-1)
        elif term.kind=="gradient":
            if (
                coefficient.ndim>=2
                and coefficient.shape[:2]
                ==field_native._gradient_shape[-2:]
            ):
                coefficient=np.moveaxis(coefficient,(0,1),(-2,-1))
        else:
            raise UnsupportedNativeForm(
                f"unsupported composite linear kind {term.kind!r}"
            )
        values=grouped.setdefault(
            term.field,{"value":None,"gradient":None}
        )
        previous=values[term.kind]
        values[term.kind]=(
            coefficient if previous is None else previous+coefficient
        )
    result=np.zeros(basis.N,dtype=np.float64)
    for field,values in grouped.items():
        result+=native.assemble(field,**values)
    return result


def _native_bilinear_assemble(
    form,basis,kwargs,num_threads=0,*,memory_limit_bytes=None,
    memory_safety_factor=1.25,
):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    expression=_trace_expression(
        form.function,(_TrialValue(),_TestValue()),geometry,kwargs,
        context="BilinearForm",
    )
    terms=_extract_terms(
        expression,_BilinearTerm,_BilinearSum,
        message=(
            "native BilinearForm must reduce to dot(u, v), "
            "ddot(grad(u), grad(v)), or a sum of those terms"
        ),
    )
    value=None
    gradient=None
    symmetric_gradient=None
    divergence=None
    tensor_terms=[]
    for term in terms:
        coefficient=_evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor,squeeze=True
        )
        if term.kind=="value":
            value=coefficient if value is None else value+coefficient
        elif term.kind=="gradient":
            gradient=(
                coefficient if gradient is None else gradient+coefficient
            )
        elif term.kind=="symmetric_gradient":
            symmetric_gradient=(
                coefficient if symmetric_gradient is None
                else symmetric_gradient+coefficient
            )
        elif term.kind=="divergence":
            divergence=(
                coefficient if divergence is None
                else divergence+coefficient
            )
        elif term.kind=="gradient_tensor":
            tensor_terms.append((coefficient,1.))
        else:
            raise UnsupportedNativeForm(
                f"unsupported bilinear term kind {term.kind!r}"
            )
    result=None
    if any(item is not None for item in (
        value,gradient,symmetric_gradient,divergence
    )):
        native = form._native_cache.get(basis)
        if native is None:
            native = NativeBilinearForm(
                basis,
                memory_limit_bytes=memory_limit_bytes,
                memory_safety_factor=memory_safety_factor,
            )
            form._native_cache[basis] = native
        elif memory_limit_bytes is not None:
            from .preflight import enforce_memory_budget
            enforce_memory_budget(
                native.memory_estimate,memory_limit_bytes,
                safety_factor=memory_safety_factor,
            )
        result=native.assemble(
            value=value,gradient=gradient,
            symmetric_gradient=symmetric_gradient,divergence=divergence,
            num_threads=num_threads
        )
    if tensor_terms:
        if basis.elem._dim!=1:
            raise UnsupportedNativeForm(
                "anisotropic gradient tensor currently requires a scalar field"
            )
        by_trial=form._cross_cache.setdefault(basis,WeakKeyDictionary())
        tensor_native=by_trial.get(basis)
        if tensor_native is None:
            tensor_native=NativeCrossBilinearForm(
                basis,basis,memory_limit_bytes=memory_limit_bytes,
                memory_safety_factor=memory_safety_factor,
            )
            by_trial[basis]=tensor_native
        dimension=basis.mesh.dim()
        for raw,factor in tensor_terms:
            tensor=_anisotropic_tensor_coefficient(
                raw,basis.dx.shape,dimension
            )*factor
            matrix=tensor_native.assemble_tensor(
                "gradient","gradient",tensor,num_threads=num_threads
            )
            result=matrix if result is None else result+matrix
    if result is None:
        raise UnsupportedNativeForm("bilinear form contains no assembled terms")
    return result


def _anisotropic_tensor_coefficient(raw,coefficient_shape,dimension):
    """Normalize scalar-field diffusion tensors to native trailing axes."""
    tensor=np.asarray(raw,dtype=np.float64)
    if tensor.shape==(dimension,dimension):
        tensor=np.broadcast_to(
            tensor,coefficient_shape+(dimension,dimension)
        )
    elif tensor.shape==(dimension,dimension)+coefficient_shape:
        # Public form coefficients follow scikit-fem's component-first layout:
        # (dim, dim, entity, quadrature).  The C++ kernel consumes one compact
        # tensor per quadrature point, so its internal layout is entity-first:
        # (entity, quadrature, dim, dim).  Keep this conversion explicit; the
        # two layouts have the same mathematical indices but different memory
        # traversal conventions.
        tensor=np.moveaxis(tensor,(0,1),(-2,-1))
    elif tensor.shape==coefficient_shape+(dimension,dimension):
        # Accepted as a low-level/native compatibility layout.  User-facing
        # documentation intentionally recommends the component-first form.
        pass
    else:
        raise ValueError(
            "anisotropic tensor must use the recommended scikit-fem shape "
            f"{(dimension, dimension) + coefficient_shape}, constant shape "
            f"({dimension}, {dimension}), "
            "or the low-level native compatibility shape "
            f"{coefficient_shape + (dimension, dimension)}; got {tensor.shape}"
        )
    return np.ascontiguousarray(tensor[...,None,:,None,:])


def _native_cross_bilinear_assemble(
    form,trial_basis,test_basis,idx,kwargs,*,memory_limit_bytes=None,
    memory_safety_factor=1.25,
):
    geometry={
        "x":_QuadratureValue(np.moveaxis(
            test_basis.global_coordinates,-1,0
        )),
    }
    if idx is not None:
        geometry["idx"]=idx
    if test_basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(
            test_basis.normals,-1,0
        ))
    try:
        expression=form.function(
            _TrialValue(),_TestValue(),
            _Parameters(_parameter_values(geometry,kwargs)),
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"cross-basis BilinearForm contains an unsupported "
            f"operation: {error}"
        ) from error
    terms=(
        expression.terms if isinstance(expression,_BilinearSum)
        else (expression,) if isinstance(expression,_BilinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "cross-basis BilinearForm must reduce to value, gradient, "
            "or divergence contractions"
        )
    by_trial=form._cross_cache.setdefault(
        trial_basis,WeakKeyDictionary()
    )
    native=by_trial.get(test_basis)
    if native is None:
        native=NativeCrossBilinearForm(
            test_basis,trial_basis,
            memory_limit_bytes=memory_limit_bytes,
            memory_safety_factor=memory_safety_factor,
        )
        by_trial[test_basis]=native
    elif memory_limit_bytes is not None:
        from .preflight import enforce_memory_budget
        enforce_memory_budget(
            native.memory_estimate,memory_limit_bytes,
            safety_factor=memory_safety_factor,
        )
    result=None
    for term in terms:
        coefficient=_evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor,squeeze=True
        )
        matrix=native.assemble(term.kind,coefficient)
        result=matrix if result is None else result+matrix
    return result


def _native_interior_bilinear_assemble(
    form,trial_bases,test_bases,kwargs
):
    trial_bases=tuple(trial_bases)
    test_bases=tuple(test_bases)
    if not trial_bases or not test_bases:
        raise ValueError("interior facet basis lists cannot be empty")
    result=None
    for trial_index,trial_basis in enumerate(trial_bases):
        for test_index,test_basis in enumerate(test_bases):
            matrix=_native_cross_bilinear_assemble(
                form,trial_basis,test_basis,
                (trial_index,test_index),kwargs,
            )
            result=matrix if result is None else result+matrix
    return result


def _native_composite_bilinear_assemble(form,basis,kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    trial=tuple(
        _CompositeField("trial",field)
        for field in range(len(basis.subbases))
    )
    test=tuple(
        _CompositeField("test",field)
        for field in range(len(basis.subbases))
    )
    try:
        expression=form.function(
            *trial,*test,
            _Parameters(_parameter_values(geometry,kwargs)),
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"composite BilinearForm contains an unsupported "
            f"operation: {error}"
        ) from error
    terms=(
        expression.terms
        if isinstance(expression,_CompositeBilinearSum)
        else (expression,)
        if isinstance(expression,_CompositeBilinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "composite BilinearForm must reduce to subfield value or "
            "gradient contractions"
        )
    native=form._native_cache.get(basis)
    if native is None:
        native=NativeCompositeBilinearForm(basis)
        form._native_cache[basis]=native
    result=None
    for term in terms:
        coefficient=_evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor,squeeze=True
        )
        block=native.assemble(
            term.row_field,term.column_field,
            kind=term.kind,coefficient=coefficient,
        )
        result=block if result is None else result+block
    return result


def _interface_geometry(integration,kwargs):
    geometry={
        "x":_QuadratureValue(np.moveaxis(
            integration.global_coordinates,-1,0
        )),
        "n_master":_QuadratureValue(np.moveaxis(
            integration.master_normals,-1,0
        )),
        "n_slave":_QuadratureValue(np.moveaxis(
            integration.slave_normals,-1,0
        )),
        "gap":_QuadratureValue(integration.gap),
    }
    return _parameter_values(geometry,kwargs)


def _native_interface_functional_assemble(form,integration,kwargs):
    try:
        expression=form.function(
            _Parameters(_interface_geometry(integration,kwargs))
        )
        values=np.asarray(expression,dtype=np.float64)
    except Exception as error:
        raise UnsupportedNativeForm(
            f"interface Functional contains an unsupported "
            f"operation: {error}"
        ) from error
    try:
        values=np.broadcast_to(
            values,integration._coefficient_shape
        )
    except ValueError as error:
        raise UnsupportedNativeForm(
            "interface Functional must evaluate to one scalar per "
            "overlap quadrature point"
        ) from error
    return integrate_functional(
        np.ascontiguousarray(values),
        np.ascontiguousarray(integration._weights,dtype=np.float64),
    )


def _native_interface_assemble(form,integration,kwargs,num_threads=0):
    try:
        expression=form.function(
            _InterfaceTrace("trial"),_InterfaceTrace("test"),
            _Parameters(_interface_geometry(integration,kwargs))
        )
    except Exception as error:
        if isinstance(error,UnsupportedNativeForm):
            raise
        raise UnsupportedNativeForm(
            f"interface form contains an unsupported operation: {error}"
        ) from error
    terms=(
        expression.terms if isinstance(expression,_InterfaceSum)
        else (expression,) if isinstance(expression,_InterfaceBilinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "interface form must contract jump/avg traces with dot"
        )
    result=None
    for term in terms:
        if term.row.weights is None or term.column.weights is None:
            raise UnsupportedNativeForm(
                "both interface trial and test fields require jump() or avg()"
            )
        coefficient=_evaluate_coefficient(
            term.coefficient,kwargs,factor=term.factor
        )
        if np.ndim(coefficient):
            coefficient=np.asarray(coefficient).squeeze()
        matrix=integration.assemble_traces(
            term.row.weights,term.column.weights,
            row_kind=term.row.kind,column_kind=term.column.kind,
            coefficient=coefficient,num_threads=num_threads or None,
        )
        result=matrix if result is None else result+matrix
    return result


def _native_interface_linear_assemble(
    form,integration,kwargs,num_threads=0,
):
    try:
        expression=form.function(
            _InterfaceTrace("test"),
            _Parameters(_interface_geometry(integration,kwargs))
        )
    except Exception as error:
        if isinstance(error,UnsupportedNativeForm):
            raise
        raise UnsupportedNativeForm(
            f"interface LinearForm contains an unsupported operation: {error}"
        ) from error
    terms=(
        expression.terms if isinstance(expression,_InterfaceLinearSum)
        else (expression,) if isinstance(expression,_InterfaceLinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "interface LinearForm must contract a coefficient with "
            "jump/avg of its test field"
        )
    result=None
    for term in terms:
        if term.trace.weights is None:
            raise UnsupportedNativeForm(
                "interface LinearForm test field requires jump() or avg()"
            )
        coefficient=_resolve_coefficient(term.coefficient,kwargs)
        vector=integration.assemble_linear_trace(
            term.trace.weights,trace_kind=term.trace.kind,
            coefficient=term.factor*np.asarray(
                coefficient,dtype=np.float64
            ),
            num_threads=num_threads or None,
        )
        result=vector if result is None else result+vector
    return result


def asm(
    form,*bases,num_threads=None,memory_limit_bytes=None,
    memory_safety_factor=1.25,**kwargs
):
    """Assemble strictly with the native backend.

    Unsupported forms raise ``UnsupportedNativeForm``; this function never
    silently delegates assembly to scikit-fem.
    """
    requested_threads=0
    if num_threads is not None:
        from .runtime import available_num_threads
        if (
            isinstance(num_threads,bool)
            or not isinstance(num_threads,int)
            or num_threads<1
        ):
            raise ValueError("num_threads must be a positive integer")
        requested_threads=min(num_threads,available_num_threads())
    if isinstance(form,_Functional):
        integration=kwargs.pop("integration",None)
        if integration is not None:
            if len(bases) not in (0,2):
                raise UnsupportedNativeForm(
                    "interface Functional accepts no bases or its "
                    "master and slave bases"
                )
            return _native_interface_functional_assemble(
                form,integration,kwargs
            )
        if len(bases)!=1:
            raise UnsupportedNativeForm(
                "native Functional requires one Basis or FacetBasis"
            )
        return _native_functional_assemble(form,bases[0],kwargs)
    if isinstance(form,_LinearForm):
        integration=kwargs.pop("integration",None)
        if integration is not None:
            if len(bases)!=2:
                raise UnsupportedNativeForm(
                    "interface LinearForm requires master and slave bases"
                )
            return _native_interface_linear_assemble(
                form,integration,kwargs,requested_threads
            )
        if len(bases)==1:
            if hasattr(bases[0],"subbases"):
                if num_threads is not None:
                    raise UnsupportedNativeForm(
                        "per-call threads are not yet supported for "
                        "composite LinearForm"
                    )
                return _native_composite_linear_assemble(
                    form,bases[0],kwargs
                )
            return _native_linear_assemble(
                form,bases[0],kwargs,requested_threads
            )
        raise UnsupportedNativeForm(
            "native LinearForm requires one basis, or two bases with "
            "an interface integration"
        )
    if isinstance(form, _BilinearForm):
        integration=kwargs.pop("integration",None)
        if integration is not None:
            if len(bases)!=2:
                raise UnsupportedNativeForm(
                    "interface assembly requires master and slave bases"
                )
            return _native_interface_assemble(
                form,integration,kwargs,requested_threads
            )
        if (
            len(bases)==2
            and isinstance(bases[0],(list,tuple))
            and isinstance(bases[1],(list,tuple))
        ):
            if num_threads is not None:
                raise UnsupportedNativeForm(
                    "per-call threads are not yet supported for "
                    "interior BilinearForm"
                )
            return _native_interior_bilinear_assemble(
                form,bases[0],bases[1],kwargs
            )
        if len(bases)==2:
            if num_threads is not None:
                raise UnsupportedNativeForm(
                    "per-call threads are not yet supported for "
                    "cross-basis BilinearForm"
                )
            return _native_cross_bilinear_assemble(
                form,bases[0],bases[1],None,kwargs,
                memory_limit_bytes=memory_limit_bytes,
                memory_safety_factor=memory_safety_factor,
            )
        if len(bases) != 1:
            raise UnsupportedNativeForm(
                "native BilinearForm currently requires one shared basis"
            )
        if hasattr(bases[0],"subbases"):
            if num_threads is not None:
                raise UnsupportedNativeForm(
                    "per-call threads are not yet supported for "
                    "composite BilinearForm"
                )
            return _native_composite_bilinear_assemble(
                form,bases[0],kwargs
            )
        return _native_bilinear_assemble(
            form,bases[0],kwargs,requested_threads,
            memory_limit_bytes=memory_limit_bytes,
            memory_safety_factor=memory_safety_factor,
        )
    raise TypeError(
        "skfemntv.asm accepts forms created by skfemntv.Functional, "
        "skfemntv.LinearForm, or skfemntv.BilinearForm; use skfem.asm "
        "explicitly for scikit-fem forms"
    )


def dot(left, right):
    if isinstance(left,_TensorGradient) and isinstance(
        right,(_TrialGradient,_TestGradient)
    ):
        if left.gradient.__class__ is right.__class__:
            raise UnsupportedNativeForm(
                "anisotropic contraction requires trial and test gradients"
            )
        return _BilinearTerm(
            "gradient_tensor",left.coefficient,
            left.gradient.factor*right.factor,
        )
    if isinstance(right,_TensorGradient) and isinstance(
        left,(_TrialGradient,_TestGradient)
    ):
        if right.gradient.__class__ is left.__class__:
            raise UnsupportedNativeForm(
                "anisotropic contraction requires trial and test gradients"
            )
        return _BilinearTerm(
            "gradient_tensor",right.coefficient,
            right.gradient.factor*left.factor,
        )
    if _is_symbolic_coefficient(left) and isinstance(right,_CompositeField):
        if right.role=="test" and right.kind=="value":
            return _CompositeLinearTerm(right.field,"value",left)
    if _is_symbolic_coefficient(right) and isinstance(left,_CompositeField):
        if left.role=="test" and left.kind=="value":
            return _CompositeLinearTerm(left.field,"value",right)
    if (
        isinstance(left,(np.ndarray,_QuadratureValue))
        or (hasattr(left,"value") and hasattr(left,"grad"))
    ) and isinstance(
        right,_CompositeField
    ):
        if right.role=="test" and right.kind=="value":
            return _CompositeLinearTerm(
                right.field,"value",np.asarray(left)
            )
    if (
        isinstance(right,(np.ndarray,_QuadratureValue))
        or (hasattr(right,"value") and hasattr(right,"grad"))
    ) and isinstance(
        left,_CompositeField
    ):
        if left.role=="test" and left.kind=="value":
            return _CompositeLinearTerm(
                left.field,"value",np.asarray(right)
            )
    if _is_symbolic_coefficient(left) and isinstance(right, _TestValue):
        return _Term("value", left)
    if _is_symbolic_coefficient(right) and isinstance(left, _TestValue):
        return _Term("value", right)
    if isinstance(right,_TestValue) and (
        isinstance(left,(np.ndarray,_QuadratureValue))
        or (hasattr(left,"value") and hasattr(left,"grad"))
    ):
        return _Term("value",np.asarray(left))
    if isinstance(left,_TestValue) and (
        isinstance(right,(np.ndarray,_QuadratureValue))
        or (hasattr(right,"value") and hasattr(right,"grad"))
    ):
        return _Term("value",np.asarray(right))
    if isinstance(left,(np.ndarray,_QuadratureValue)) and isinstance(
        right,_InterfaceTrace
    ):
        if right.role=="test" and right.kind!="gradient":
            return _InterfaceLinearTerm(right,np.asarray(left))
    if isinstance(right,(np.ndarray,_QuadratureValue)) and isinstance(
        left,_InterfaceTrace
    ):
        if left.role=="test" and left.kind!="gradient":
            return _InterfaceLinearTerm(left,np.asarray(right))
    if isinstance(left,(np.ndarray,_QuadratureValue)) and isinstance(
        right,_InterfaceTrace
    ):
        if right.kind=="gradient":
            return _InterfaceCoefficientTrace(
                right,np.asarray(left)
            )
    if isinstance(right,(np.ndarray,_QuadratureValue)) and isinstance(
        left,_InterfaceTrace
    ):
        if left.kind=="gradient":
            return _InterfaceCoefficientTrace(
                left,np.asarray(right)
            )
    if (
        isinstance(left, _TrialValue)
        and isinstance(right, _TestValue)
    ) or (
        isinstance(right, _TrialValue)
        and isinstance(left, _TestValue)
    ):
        return _BilinearTerm("value",factor=left.factor*right.factor)
    if isinstance(left,_CompositeField) and isinstance(
        right,_CompositeField
    ):
        if left.kind=="gradient" and right.kind=="gradient":
            return _composite_contraction(left,right,"gradient")
        if left.kind!="value" or right.kind!="value":
            raise UnsupportedNativeForm(
                "composite dot requires two values or two gradients"
            )
        return _composite_contraction(left,right,"value")
    if isinstance(left,_Coefficient) and isinstance(right,_InterfaceTrace):
        if right.role=="test":
            if right.kind=="gradient":
                raise UnsupportedNativeForm(
                    "use ddot(coefficient, grad(test)) for a full gradient"
                )
            return _InterfaceLinearTerm(right,left)
        if right.kind!="gradient":
            raise UnsupportedNativeForm(
                "an interface coefficient contraction requires grad(field)"
            )
        return _InterfaceCoefficientTrace(right,left.name)
    if isinstance(right,_Coefficient) and isinstance(left,_InterfaceTrace):
        if left.role=="test":
            if left.kind=="gradient":
                raise UnsupportedNativeForm(
                    "use ddot(coefficient, grad(test)) for a full gradient"
                )
            return _InterfaceLinearTerm(left,right)
        if left.kind!="gradient":
            raise UnsupportedNativeForm(
                "an interface coefficient contraction requires grad(field)"
            )
        return _InterfaceCoefficientTrace(left,right.name)
    if isinstance(left,_InterfaceTrace) and isinstance(
        right,_InterfaceCoefficientTrace
    ):
        if left.role=="test" and right.trace.role=="trial":
            return _InterfaceBilinearTerm(
                left,right.trace,right.coefficient
            )
        if left.role=="trial" and right.trace.role=="test":
            if isinstance(right.coefficient,str):
                raise UnsupportedNativeForm(
                    "test-gradient coefficients must be evaluated in the form"
                )
            coefficient=np.moveaxis(np.asarray(right.coefficient),-3,-1)
            return _InterfaceBilinearTerm(
                right.trace,left,coefficient
            )
        raise UnsupportedNativeForm(
            "interface coefficient contraction requires trial and test fields"
        )
    if isinstance(right,_InterfaceTrace) and isinstance(
        left,_InterfaceCoefficientTrace
    ):
        if right.role=="test" and left.trace.role=="trial":
            return _InterfaceBilinearTerm(
                right,left.trace,left.coefficient
            )
        if right.role=="trial" and left.trace.role=="test":
            if isinstance(left.coefficient,str):
                raise UnsupportedNativeForm(
                    "test-gradient coefficients must be evaluated in the form"
                )
            coefficient=np.moveaxis(np.asarray(left.coefficient),-3,-1)
            return _InterfaceBilinearTerm(
                left.trace,right,coefficient
            )
        raise UnsupportedNativeForm(
            "interface coefficient contraction requires trial and test fields"
        )
    if isinstance(left,_InterfaceTrace) and isinstance(right,_InterfaceTrace):
        if left.role=="test":
            return _InterfaceBilinearTerm(left,right)
        if right.role=="test":
            return _InterfaceBilinearTerm(right,left)
        raise UnsupportedNativeForm("interface dot requires trial and test")
    return np.einsum("i...,i...->...", left, right)


def mul(left,right):
    """Matrix-vector product in the typed native form subset."""
    if isinstance(right,(_TrialGradient,_TestGradient)) and isinstance(
        left,(_Coefficient,_CoefficientComponent,np.ndarray,_QuadratureValue)
    ):
        coefficient=(
            left if _is_symbolic_coefficient(left) else np.asarray(left)
        )
        return _TensorGradient(right,coefficient)
    raise UnsupportedNativeForm(
        "native mul currently supports coefficient @ trial/test gradient"
    )


def ddot(left, right):
    if (
        isinstance(left,_SymmetricGradient)
        and isinstance(right,_SymmetricGradient)
        and left.role!=right.role
    ):
        return _BilinearTerm(
            "symmetric_gradient",factor=left.factor*right.factor
        )
    if _is_symbolic_coefficient(left) and isinstance(right,_CompositeField):
        if right.role=="test" and right.kind=="gradient":
            return _CompositeLinearTerm(right.field,"gradient",left)
    if _is_symbolic_coefficient(right) and isinstance(left,_CompositeField):
        if left.role=="test" and left.kind=="gradient":
            return _CompositeLinearTerm(left.field,"gradient",right)
    if isinstance(left,(np.ndarray,_QuadratureValue)) and isinstance(
        right,_CompositeField
    ):
        if right.role=="test" and right.kind=="gradient":
            return _CompositeLinearTerm(
                right.field,"gradient",np.asarray(left)
            )
    if isinstance(right,(np.ndarray,_QuadratureValue)) and isinstance(
        left,_CompositeField
    ):
        if left.role=="test" and left.kind=="gradient":
            return _CompositeLinearTerm(
                left.field,"gradient",np.asarray(right)
            )
    if _is_symbolic_coefficient(left) and isinstance(right, _TestGradient):
        return _Term("gradient", left)
    if _is_symbolic_coefficient(right) and isinstance(left, _TestGradient):
        return _Term("gradient", right)
    if isinstance(left,(np.ndarray,_QuadratureValue)) and isinstance(
        right,_TestGradient
    ):
        return _Term("gradient",np.asarray(left))
    if isinstance(right,(np.ndarray,_QuadratureValue)) and isinstance(
        left,_TestGradient
    ):
        return _Term("gradient",np.asarray(right))
    if (
        isinstance(left, _TrialGradient)
        and isinstance(right, _TestGradient)
    ) or (
        isinstance(right, _TrialGradient)
        and isinstance(left, _TestGradient)
    ):
        return _BilinearTerm("gradient",factor=left.factor*right.factor)
    if isinstance(left,_CompositeField) and isinstance(
        right,_CompositeField
    ):
        if left.kind!="gradient" or right.kind!="gradient":
            raise UnsupportedNativeForm(
                "composite ddot requires two gradients"
            )
        return _composite_contraction(left,right,"gradient")
    if isinstance(left,_Coefficient) and isinstance(right,_InterfaceTrace):
        if right.role=="test" and right.kind=="gradient":
            return _InterfaceLinearTerm(right,left)
    if isinstance(right,_Coefficient) and isinstance(left,_InterfaceTrace):
        if left.role=="test" and left.kind=="gradient":
            return _InterfaceLinearTerm(left,right)
    if isinstance(left,(np.ndarray,_QuadratureValue)) and isinstance(
        right,_InterfaceTrace
    ):
        if right.role=="test" and right.kind=="gradient":
            return _InterfaceLinearTerm(right,np.asarray(left))
    if isinstance(right,(np.ndarray,_QuadratureValue)) and isinstance(
        left,_InterfaceTrace
    ):
        if left.role=="test" and left.kind=="gradient":
            return _InterfaceLinearTerm(left,np.asarray(right))
    if isinstance(left,_InterfaceTrace) and isinstance(right,_InterfaceTrace):
        if left.kind!="gradient" or right.kind!="gradient":
            raise UnsupportedNativeForm(
                "interface ddot requires two full gradients"
            )
        if left.role=="test":
            return _InterfaceBilinearTerm(left,right)
        if right.role=="test":
            return _InterfaceBilinearTerm(right,left)
        raise UnsupportedNativeForm("interface ddot requires trial and test")
    return np.einsum("ij...,ij...->...", left, right)


def grad(value):
    if isinstance(value, _TestValue):
        return _TestGradient(value.factor)
    if isinstance(value, _TrialValue):
        return _TrialGradient(value.factor)
    if isinstance(value,_InterfaceTrace):
        return value._interface_transform("kind","gradient")
    if isinstance(value,_CompositeField):
        return _CompositeField(value.role,value.field,"gradient")
    try:
        return value.grad
    except AttributeError as error:
        raise UnsupportedNativeForm("grad() requires a form field") from error


def div(value):
    if isinstance(value,_TrialValue):
        return _Divergence("trial",value.factor)
    if isinstance(value,_TestValue):
        return _Divergence("test",value.factor)
    if isinstance(value,_CompositeField):
        if value.kind!="value":
            raise UnsupportedNativeForm("div() expects a composite value")
        return _CompositeDivergence(value)
    try:
        return np.einsum("ii...->...",value.grad)
    except AttributeError as error:
        raise UnsupportedNativeForm("div() requires a vector field") from error
