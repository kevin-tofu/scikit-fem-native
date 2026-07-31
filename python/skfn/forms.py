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


class UnsupportedNativeForm(Exception):
    """Raised when a form cannot be assembled by the native backend."""


class _QuadratureValue:
    """Numerical geometry value that preserves form multiplication."""

    __array_priority__=1000

    def __init__(self,value):
        self.value=np.asarray(value)

    def __array__(self,dtype=None):
        return np.asarray(self.value,dtype=dtype)

    def __getitem__(self,key):
        return _QuadratureValue(self.value[key])

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        values=[
            item.value if isinstance(item,_QuadratureValue) else item
            for item in inputs
        ]
        result=getattr(ufunc,method)(*values,**kwargs)
        return _QuadratureValue(result)

    def __mul__(self,other):
        if isinstance(other,(
            _BilinearTerm,_CompositeBilinearTerm,
            _InterfaceBilinearTerm,
        )):
            return other*np.asarray(self.value)
        if isinstance(other,_CompositeField):
            return _CompositeWeightedField(
                other,np.asarray(self.value)
            )
        return _QuadratureValue(self.value*np.asarray(other))

    def __rmul__(self,other):
        if isinstance(other,(
            _BilinearTerm,_CompositeBilinearTerm,
            _InterfaceBilinearTerm,
        )):
            return other*np.asarray(self.value)
        return _QuadratureValue(np.asarray(other)*self.value)

    def __add__(self,other):
        return _QuadratureValue(self.value+np.asarray(other))

    def __radd__(self,other):
        return _QuadratureValue(np.asarray(other)+self.value)

    def __sub__(self,other):
        return _QuadratureValue(self.value-np.asarray(other))

    def __rsub__(self,other):
        return _QuadratureValue(np.asarray(other)-self.value)

    def __truediv__(self,other):
        return _QuadratureValue(self.value/np.asarray(other))

    def __rtruediv__(self,other):
        return _QuadratureValue(np.asarray(other)/self.value)

    def __pow__(self,other):
        return _QuadratureValue(self.value**other)

    def __neg__(self):
        return _QuadratureValue(-self.value)


@dataclass(frozen=True)
class _TestValue:
    factor: float = 1.

    def __mul__(self,value):
        return _TestValue(self.factor*value) if np.isscalar(value) else NotImplemented

    __rmul__=__mul__


@dataclass(frozen=True)
class _TestGradient:
    factor: float = 1.


@dataclass(frozen=True)
class _TrialValue:
    factor: float = 1.

    def __mul__(self,value):
        return _TrialValue(self.factor*value) if np.isscalar(value) else NotImplemented

    __rmul__=__mul__


@dataclass(frozen=True)
class _TrialGradient:
    factor: float = 1.


@dataclass(frozen=True)
class _CompositeField:
    role: str
    field: int
    kind: str = "value"

    def __mul__(self,other):
        if isinstance(other,_CompositeDivergence):
            return _composite_divergence_contraction(self,other)
        if isinstance(other,_CompositeField):
            if self.kind!="value" or other.kind!="value":
                return NotImplemented
            return _composite_contraction(self,other,"value")
        if isinstance(other,_Coefficient):
            return _CompositeWeightedField(self,other.name)
        if np.isscalar(other) or isinstance(
            other,(np.ndarray,_QuadratureValue)
        ) or (
            hasattr(other,"value") and hasattr(other,"grad")
        ):
            return _CompositeWeightedField(self,np.asarray(other))
        return NotImplemented

    __rmul__=__mul__

    def __neg__(self):
        return _CompositeWeightedField(self,-1.)


@dataclass(frozen=True)
class _CompositeWeightedField:
    field: _CompositeField
    coefficient: Any

    def __mul__(self,other):
        if isinstance(other,_CompositeDivergence):
            term=_composite_divergence_contraction(self.field,other)
            return _CompositeBilinearTerm(
                term.row_field,term.column_field,term.kind,
                self.coefficient,term.factor,
            )
        if not isinstance(other,_CompositeField):
            return NotImplemented
        term=_composite_contraction(self.field,other,"value")
        return _CompositeBilinearTerm(
            term.row_field,term.column_field,term.kind,
            self.coefficient,term.factor,
        )

    __rmul__=__mul__

    def _linear_term(self):
        if self.field.role!="test":
            raise UnsupportedNativeForm(
                "composite LinearForm requires test subfields"
            )
        return _CompositeLinearTerm(
            self.field.field,self.field.kind,self.coefficient
        )

    def __add__(self,other):
        return self._linear_term()+other

    __radd__=__add__

    def __neg__(self):
        if isinstance(self.coefficient,str):
            return _CompositeLinearTerm(
                self.field.field,self.field.kind,self.coefficient,-1.
            )
        return _CompositeWeightedField(self.field,-np.asarray(self.coefficient))

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _CompositeDivergence:
    field: _CompositeField

    def __mul__(self,other):
        if isinstance(other,(_CompositeField,_CompositeWeightedField)):
            return other*self
        return NotImplemented

    __rmul__=__mul__


@dataclass(frozen=True)
class _CompositeLinearTerm:
    field: int
    kind: str
    coefficient: Any
    factor: float = 1.

    def __mul__(self,other):
        if np.isscalar(other):
            return _CompositeLinearTerm(
                self.field,self.kind,self.coefficient,self.factor*other
            )
        return NotImplemented

    __rmul__=__mul__

    def __neg__(self):
        return _CompositeLinearTerm(
            self.field,self.kind,self.coefficient,-self.factor
        )

    def __add__(self,other):
        if isinstance(other,_CompositeWeightedField):
            other=other._linear_term()
        if isinstance(other,_CompositeLinearTerm):
            return _CompositeLinearSum((self,other))
        if isinstance(other,_CompositeLinearSum):
            return _CompositeLinearSum((self,)+other.terms)
        return NotImplemented

    __radd__=__add__

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _CompositeLinearSum:
    terms: tuple[_CompositeLinearTerm,...]

    def __add__(self,other):
        if isinstance(other,_CompositeWeightedField):
            other=other._linear_term()
        if isinstance(other,_CompositeLinearTerm):
            return _CompositeLinearSum(self.terms+(other,))
        if isinstance(other,_CompositeLinearSum):
            return _CompositeLinearSum(self.terms+other.terms)
        return NotImplemented

    __radd__=__add__

    def __neg__(self):
        return _CompositeLinearSum(tuple(-term for term in self.terms))

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _Coefficient:
    name: str

    def __getitem__(self, _):
        raise UnsupportedNativeForm(
            "coefficient indexing is not supported by native forms"
        )

    def __mul__(self, other):
        if isinstance(other,_CompositeField):
            return other*self
        if isinstance(other, (
            _BilinearTerm,_CompositeBilinearTerm,
            _InterfaceBilinearTerm,
        )):
            if isinstance(other,_CompositeBilinearTerm):
                return _CompositeBilinearTerm(
                    other.row_field,other.column_field,
                    other.kind,self.name,other.factor
                )
            if isinstance(other,_InterfaceBilinearTerm):
                return _InterfaceBilinearTerm(
                    other.row,other.column,self.name,other.factor
                )
            return _BilinearTerm(other.kind, self.name, other.factor)
        return NotImplemented

    __rmul__ = __mul__


@dataclass(frozen=True)
class _Term:
    kind: str
    coefficient: Any
    factor: float = 1.0

    def __neg__(self):
        return _Term(self.kind, self.coefficient, -self.factor)

    def __mul__(self, value):
        if np.isscalar(value):
            return _Term(self.kind, self.coefficient, self.factor * value)
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        return _Sum((self,)) + other

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class _Sum:
    terms: tuple[_Term, ...]

    def __add__(self, other):
        if isinstance(other, _Term):
            return _Sum(self.terms + (other,))
        if isinstance(other, _Sum):
            return _Sum(self.terms + other.terms)
        raise UnsupportedNativeForm

    __radd__ = __add__

    def __neg__(self):
        return _Sum(tuple(-term for term in self.terms))

    def __sub__(self, other):
        return self + (-other)


@dataclass(frozen=True)
class _BilinearTerm:
    kind: str
    coefficient: Any = None
    factor: float = 1.0

    def __mul__(self, other):
        if np.isscalar(other):
            return _BilinearTerm(
                self.kind, self.coefficient, self.factor * other
            )
        if isinstance(other, _Coefficient):
            return _BilinearTerm(self.kind, other.name, self.factor)
        if isinstance(other,np.ndarray) or (
            hasattr(other,"value") and hasattr(other,"grad")
        ):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple bilinear coefficients are not supported"
                )
            return _BilinearTerm(
                self.kind,np.asarray(other),self.factor
            )
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return _BilinearTerm(
            self.kind,self.coefficient,-self.factor
        )

    def __add__(self,other):
        if isinstance(other,_BilinearTerm):
            return _BilinearSum((self,other))
        if isinstance(other,_BilinearSum):
            return _BilinearSum((self,)+other.terms)
        return NotImplemented

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _BilinearSum:
    terms: tuple[_BilinearTerm,...]

    def __add__(self,other):
        if isinstance(other,_BilinearTerm):
            return _BilinearSum(self.terms+(other,))
        if isinstance(other,_BilinearSum):
            return _BilinearSum(self.terms+other.terms)
        return NotImplemented

    __radd__=__add__

    def __neg__(self):
        return _BilinearSum(tuple(-term for term in self.terms))

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _CompositeBilinearTerm:
    row_field: int
    column_field: int
    kind: str
    coefficient: Any = None
    factor: float = 1.

    def __mul__(self,other):
        if np.isscalar(other):
            return _CompositeBilinearTerm(
                self.row_field,self.column_field,self.kind,
                self.coefficient,self.factor*other,
            )
        if isinstance(other,_Coefficient):
            return _CompositeBilinearTerm(
                self.row_field,self.column_field,self.kind,
                other.name,self.factor,
            )
        if isinstance(other,np.ndarray) or (
            hasattr(other,"value") and hasattr(other,"grad")
        ):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple composite coefficients are not supported"
                )
            return _CompositeBilinearTerm(
                self.row_field,self.column_field,self.kind,
                np.asarray(other),self.factor,
            )
        return NotImplemented

    __rmul__=__mul__

    def __neg__(self):
        return _CompositeBilinearTerm(
            self.row_field,self.column_field,self.kind,
            self.coefficient,-self.factor,
        )

    def __add__(self,other):
        if isinstance(other,_CompositeBilinearTerm):
            return _CompositeBilinearSum((self,other))
        if isinstance(other,_CompositeBilinearSum):
            return _CompositeBilinearSum((self,)+other.terms)
        return NotImplemented

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _CompositeBilinearSum:
    terms: tuple[_CompositeBilinearTerm,...]

    def __add__(self,other):
        if isinstance(other,_CompositeBilinearTerm):
            return _CompositeBilinearSum(self.terms+(other,))
        if isinstance(other,_CompositeBilinearSum):
            return _CompositeBilinearSum(self.terms+other.terms)
        return NotImplemented

    __radd__=__add__

    def __neg__(self):
        return _CompositeBilinearSum(tuple(-term for term in self.terms))

    def __sub__(self,other):
        return self+(-other)


@dataclass(frozen=True)
class _InterfaceTrace:
    role: str
    weights: tuple[float,float] | None = None
    kind: str = "value"

    def _interface_transform(self,operation,value):
        if operation=="weights":
            return _InterfaceTrace(self.role,tuple(value),self.kind)
        return _InterfaceTrace(self.role,self.weights,value)


@dataclass(frozen=True)
class _InterfaceCoefficientTrace:
    trace: _InterfaceTrace
    coefficient: Any


@dataclass(frozen=True)
class _InterfaceLinearTerm:
    trace: _InterfaceTrace
    coefficient: Any
    factor: float = 1.

    def __neg__(self):
        return _InterfaceLinearTerm(
            self.trace,self.coefficient,-self.factor
        )

    def __mul__(self,value):
        if np.isscalar(value):
            return _InterfaceLinearTerm(
                self.trace,self.coefficient,self.factor*value
            )
        return NotImplemented

    __rmul__=__mul__

    def __add__(self,other):
        if isinstance(other,_InterfaceLinearTerm):
            return _InterfaceLinearSum((self,other))
        if isinstance(other,_InterfaceLinearSum):
            return _InterfaceLinearSum((self,)+other.terms)
        return NotImplemented


@dataclass(frozen=True)
class _InterfaceLinearSum:
    terms: tuple[_InterfaceLinearTerm,...]

    def __add__(self,other):
        if isinstance(other,_InterfaceLinearTerm):
            return _InterfaceLinearSum(self.terms+(other,))
        if isinstance(other,_InterfaceLinearSum):
            return _InterfaceLinearSum(self.terms+other.terms)
        return NotImplemented


@dataclass(frozen=True)
class _InterfaceBilinearTerm:
    row: _InterfaceTrace
    column: _InterfaceTrace
    coefficient: Any = None
    factor: float = 1.

    def __mul__(self,other):
        if np.isscalar(other):
            return _InterfaceBilinearTerm(
                self.row,self.column,self.coefficient,self.factor*other
            )
        if isinstance(other,_Coefficient):
            return _InterfaceBilinearTerm(
                self.row,self.column,other.name,self.factor
            )
        if isinstance(other,np.ndarray):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple interface coefficients are not supported"
                )
            return _InterfaceBilinearTerm(
                self.row,self.column,other,self.factor
            )
        return NotImplemented

    __rmul__=__mul__

    def __add__(self,other):
        if isinstance(other,_InterfaceBilinearTerm):
            return _InterfaceSum((self,other))
        return NotImplemented


@dataclass(frozen=True)
class _InterfaceSum:
    terms: tuple[_InterfaceBilinearTerm,...]

    def __add__(self,other):
        if isinstance(other,_InterfaceBilinearTerm):
            return _InterfaceSum(self.terms+(other,))
        if isinstance(other,_InterfaceSum):
            return _InterfaceSum(self.terms+other.terms)
        return NotImplemented


class _Parameters:
    def __init__(self, geometry=None):
        self._geometry={} if geometry is None else geometry

    def __getattr__(self, name):
        if name in self._geometry:
            return self._geometry[name]
        return _Coefficient(name)

    def __getitem__(self, name):
        if name in self._geometry:
            return self._geometry[name]
        return _Coefficient(name)


def _parameter_values(geometry,kwargs):
    values=dict(geometry)
    for name,value in kwargs.items():
        values[name]=(
            value
            if callable(value) or (
                hasattr(value,"value") and hasattr(value,"grad")
            )
            else _QuadratureValue(value)
        )
    return values


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
    try:
        expression = form.function(
            _TestValue(),
            _Parameters(_parameter_values(geometry,kwargs)),
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"form contains an operation that cannot be traced: {error}"
        ) from error
    if isinstance(expression, _Term):
        return (expression,)
    if isinstance(expression, _Sum):
        return expression.terms
    raise UnsupportedNativeForm(
        "native LinearForm must reduce to dot(coefficient, v), "
        "ddot(coefficient, grad(v)), or a sum of those terms"
    )


def _native_linear_assemble(form, basis, kwargs):
    terms = _compile_linear(form,basis,kwargs)
    native = form._native_cache.get(basis)
    if native is None:
        native = NativeLinearForm(basis)
        form._native_cache[basis] = native
    value = None
    gradient = None
    for term in terms:
        if isinstance(term.coefficient,_Coefficient):
            if term.coefficient.name not in kwargs:
                raise ValueError(
                    f"missing form parameter {term.coefficient.name!r}"
                )
            raw_coefficient=kwargs[term.coefficient.name]
        else:
            raw_coefficient=term.coefficient
        coefficient = term.factor * np.asarray(
            raw_coefficient, dtype=np.float64
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
    result, _ = native.assemble(value=value, gradient=gradient)
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
    try:
        expression=form.function(
            *test,_Parameters(_parameter_values(geometry,kwargs))
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"composite LinearForm contains an unsupported "
            f"operation: {error}"
        ) from error
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
        if isinstance(term.coefficient,str):
            if term.coefficient not in kwargs:
                raise ValueError(
                    f"missing form parameter {term.coefficient!r}"
                )
            raw=kwargs[term.coefficient]
        else:
            raw=term.coefficient
        coefficient=term.factor*np.asarray(raw,dtype=np.float64)
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


def _native_bilinear_assemble(form, basis, kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    try:
        expression = form.function(
            _TrialValue(), _TestValue(),
            _Parameters(_parameter_values(geometry,kwargs))
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"BilinearForm contains an unsupported operation: {error}"
        ) from error
    terms=(
        expression.terms if isinstance(expression,_BilinearSum)
        else (expression,) if isinstance(expression,_BilinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "native BilinearForm must reduce to dot(u, v), "
            "ddot(grad(u), grad(v)), or a sum of those terms"
        )
    value=None
    gradient=None
    for term in terms:
        coefficient=term.factor
        if term.coefficient is not None:
            if isinstance(term.coefficient,str):
                if term.coefficient not in kwargs:
                    raise ValueError(
                        f"missing form parameter {term.coefficient!r}"
                    )
                raw_coefficient=kwargs[term.coefficient]
            else:
                raw_coefficient=term.coefficient
            coefficient=coefficient*np.asarray(
                raw_coefficient,dtype=np.float64
            )
            if coefficient.ndim>2:
                coefficient=np.squeeze(coefficient)
        if term.kind=="value":
            value=coefficient if value is None else value+coefficient
        elif term.kind=="gradient":
            gradient=(
                coefficient if gradient is None else gradient+coefficient
            )
        else:
            raise UnsupportedNativeForm(
                f"unsupported bilinear term kind {term.kind!r}"
            )
    native = form._native_cache.get(basis)
    if native is None:
        native = NativeBilinearForm(basis)
        form._native_cache[basis] = native
    return native.assemble(value=value,gradient=gradient)


def _native_cross_bilinear_assemble(
    form,trial_basis,test_basis,idx,kwargs
):
    geometry={
        "x":_QuadratureValue(np.moveaxis(
            test_basis.global_coordinates,-1,0
        )),
        "idx":idx,
    }
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
            f"interior facet BilinearForm contains an unsupported "
            f"operation: {error}"
        ) from error
    terms=(
        expression.terms if isinstance(expression,_BilinearSum)
        else (expression,) if isinstance(expression,_BilinearTerm)
        else None
    )
    if terms is None:
        raise UnsupportedNativeForm(
            "interior facet BilinearForm must reduce to value or "
            "gradient contractions"
        )
    by_trial=form._cross_cache.setdefault(
        trial_basis,WeakKeyDictionary()
    )
    native=by_trial.get(test_basis)
    if native is None:
        native=NativeCrossBilinearForm(test_basis,trial_basis)
        by_trial[test_basis]=native
    result=None
    for term in terms:
        coefficient=term.factor
        if term.coefficient is not None:
            raw=(
                kwargs[term.coefficient]
                if isinstance(term.coefficient,str)
                else term.coefficient
            )
            coefficient=coefficient*np.asarray(raw,dtype=np.float64)
            if np.ndim(coefficient)>2:
                coefficient=np.squeeze(coefficient)
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
        coefficient=term.factor
        if term.coefficient is not None:
            if isinstance(term.coefficient,str):
                if term.coefficient not in kwargs:
                    raise ValueError(
                        f"missing form parameter {term.coefficient!r}"
                    )
                raw=kwargs[term.coefficient]
            else:
                raw=term.coefficient
            coefficient=coefficient*np.asarray(raw,dtype=np.float64)
            if coefficient.ndim>2:
                coefficient=np.squeeze(coefficient)
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


def _native_interface_assemble(form,integration,kwargs):
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
        coefficient=term.factor
        if term.coefficient is not None:
            if isinstance(term.coefficient,str):
                if term.coefficient not in kwargs:
                    raise ValueError(
                        f"missing form parameter {term.coefficient!r}"
                    )
                raw_coefficient=kwargs[term.coefficient]
            else:
                raw_coefficient=term.coefficient
            coefficient=coefficient*np.asarray(
                raw_coefficient,dtype=np.float64
            ).squeeze()
        matrix=integration.assemble_traces(
            term.row.weights,term.column.weights,
            row_kind=term.row.kind,column_kind=term.column.kind,
            coefficient=coefficient,
        )
        result=matrix if result is None else result+matrix
    return result


def _native_interface_linear_assemble(form,integration,kwargs):
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
        if isinstance(term.coefficient,_Coefficient):
            name=term.coefficient.name
            if name not in kwargs:
                raise ValueError(f"missing form parameter {name!r}")
            coefficient=kwargs[name]
        else:
            coefficient=term.coefficient
        vector=integration.assemble_linear_trace(
            term.trace.weights,trace_kind=term.trace.kind,
            coefficient=term.factor*np.asarray(
                coefficient,dtype=np.float64
            ),
        )
        result=vector if result is None else result+vector
    return result


def asm(form, *bases, **kwargs):
    """Assemble strictly with the native backend.

    Unsupported forms raise ``UnsupportedNativeForm``; this function never
    silently delegates assembly to scikit-fem.
    """
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
                form,integration,kwargs
            )
        if len(bases)==1:
            if hasattr(bases[0],"subbases"):
                return _native_composite_linear_assemble(
                    form,bases[0],kwargs
                )
            return _native_linear_assemble(form,bases[0],kwargs)
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
            return _native_interface_assemble(form,integration,kwargs)
        if (
            len(bases)==2
            and isinstance(bases[0],(list,tuple))
            and isinstance(bases[1],(list,tuple))
        ):
            return _native_interior_bilinear_assemble(
                form,bases[0],bases[1],kwargs
            )
        if len(bases) != 1:
            raise UnsupportedNativeForm(
                "native BilinearForm currently requires one shared basis"
            )
        if hasattr(bases[0],"subbases"):
            return _native_composite_bilinear_assemble(
                form,bases[0],kwargs
            )
        return _native_bilinear_assemble(form, bases[0], kwargs)
    raise TypeError(
        "skfn.asm accepts forms created by skfn.Functional, "
        "skfn.LinearForm, or skfn.BilinearForm; use skfem.asm "
        "explicitly for scikit-fem forms"
    )


def _composite_contraction(left,right,kind):
    if not isinstance(left,_CompositeField) or not isinstance(
        right,_CompositeField
    ):
        raise UnsupportedNativeForm(
            "composite contraction requires two subfields"
        )
    if left.role==right.role:
        raise UnsupportedNativeForm(
            "composite contraction requires trial and test subfields"
        )
    row=left if left.role=="test" else right
    column=right if left.role=="test" else left
    return _CompositeBilinearTerm(
        row.field,column.field,kind
    )


def _composite_divergence_contraction(value,divergence):
    gradient=divergence.field
    if value.role==gradient.role:
        raise UnsupportedNativeForm(
            "divergence coupling requires trial and test subfields"
        )
    if gradient.role=="test":
        return _CompositeBilinearTerm(
            gradient.field,value.field,"row_divergence"
        )
    return _CompositeBilinearTerm(
        value.field,gradient.field,"column_divergence"
    )


def dot(left, right):
    if isinstance(left,_Coefficient) and isinstance(right,_CompositeField):
        if right.role=="test" and right.kind=="value":
            return _CompositeLinearTerm(right.field,"value",left.name)
    if isinstance(right,_Coefficient) and isinstance(left,_CompositeField):
        if left.role=="test" and left.kind=="value":
            return _CompositeLinearTerm(left.field,"value",right.name)
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
    if isinstance(left, _Coefficient) and isinstance(right, _TestValue):
        return _Term("value", left)
    if isinstance(right, _Coefficient) and isinstance(left, _TestValue):
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
        if left.role!="test":
            raise UnsupportedNativeForm("interface dot requires a test field")
        return _InterfaceBilinearTerm(
            left,right.trace,right.coefficient
        )
    if isinstance(right,_InterfaceTrace) and isinstance(
        left,_InterfaceCoefficientTrace
    ):
        if right.role!="test":
            raise UnsupportedNativeForm("interface dot requires a test field")
        return _InterfaceBilinearTerm(
            right,left.trace,left.coefficient
        )
    if isinstance(left,_InterfaceTrace) and isinstance(right,_InterfaceTrace):
        if left.role=="test":
            return _InterfaceBilinearTerm(left,right)
        if right.role=="test":
            return _InterfaceBilinearTerm(right,left)
        raise UnsupportedNativeForm("interface dot requires trial and test")
    return np.einsum("i...,i...->...", left, right)


def ddot(left, right):
    if isinstance(left,_Coefficient) and isinstance(right,_CompositeField):
        if right.role=="test" and right.kind=="gradient":
            return _CompositeLinearTerm(right.field,"gradient",left.name)
    if isinstance(right,_Coefficient) and isinstance(left,_CompositeField):
        if left.role=="test" and left.kind=="gradient":
            return _CompositeLinearTerm(left.field,"gradient",right.name)
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
    if isinstance(left, _Coefficient) and isinstance(right, _TestGradient):
        return _Term("gradient", left)
    if isinstance(right, _Coefficient) and isinstance(left, _TestGradient):
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
    if isinstance(value,_CompositeField):
        if value.kind!="value":
            raise UnsupportedNativeForm("div() expects a composite value")
        return _CompositeDivergence(value)
    try:
        return np.einsum("ii...->...",value.grad)
    except AttributeError as error:
        raise UnsupportedNativeForm("div() requires a vector field") from error
