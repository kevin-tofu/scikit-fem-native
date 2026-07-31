from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from weakref import WeakKeyDictionary

import numpy as np

from .linear_form import NativeLinearForm
from .bilinear_form import NativeBilinearForm


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
        if isinstance(other,_BilinearTerm):
            return other*np.asarray(self.value)
        return _QuadratureValue(self.value*other)

    def __rmul__(self,other):
        if isinstance(other,_BilinearTerm):
            return other*np.asarray(self.value)
        return _QuadratureValue(other*self.value)

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
    pass


@dataclass(frozen=True)
class _TestGradient:
    pass


@dataclass(frozen=True)
class _TrialValue:
    pass


@dataclass(frozen=True)
class _TrialGradient:
    pass


@dataclass(frozen=True)
class _Coefficient:
    name: str

    def __getitem__(self, _):
        raise UnsupportedNativeForm(
            "coefficient indexing is not supported by native forms"
        )

    def __mul__(self, other):
        if isinstance(other, (_BilinearTerm,_InterfaceBilinearTerm)):
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
        if isinstance(other,np.ndarray):
            if self.coefficient is not None:
                raise UnsupportedNativeForm(
                    "multiple bilinear coefficients are not supported"
                )
            return _BilinearTerm(self.kind,other,self.factor)
        return NotImplemented

    __rmul__ = __mul__


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
    coefficient: str


@dataclass(frozen=True)
class _InterfaceBilinearTerm:
    row: _InterfaceTrace
    column: _InterfaceTrace
    coefficient: str | None = None
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


class _LinearForm:
    def __init__(self, function: Callable):
        self.function = function
        self._native_cache = WeakKeyDictionary()

    def assemble(self, basis, **kwargs):
        return asm(self, basis, **kwargs)


class _BilinearForm:
    def __init__(self, function: Callable):
        self.function = function
        self._native_cache = WeakKeyDictionary()

    def assemble(self, *bases, **kwargs):
        return asm(self, *bases, **kwargs)


def LinearForm(function=None, **kwargs):
    if kwargs:
        # Preserve less common scikit-fem decorator options through fallback.
        return lambda fn: _LinearForm(fn)
    return _LinearForm(function) if function is not None else _LinearForm


def BilinearForm(function=None, **kwargs):
    if kwargs:
        return lambda fn: _BilinearForm(fn)
    return _BilinearForm(function) if function is not None else _BilinearForm


def _compile_linear(form: _LinearForm,basis):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    try:
        expression = form.function(_TestValue(), _Parameters(geometry))
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
    terms = _compile_linear(form,basis)
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
            if coefficient.ndim and coefficient.shape[0] == native._shape[-1]:
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
    return result


def _native_bilinear_assemble(form, basis, kwargs):
    geometry={"x":_QuadratureValue(
        np.moveaxis(basis.global_coordinates,-1,0)
    )}
    if basis.normals is not None:
        geometry["n"]=_QuadratureValue(np.moveaxis(basis.normals,-1,0))
    try:
        expression = form.function(
            _TrialValue(), _TestValue(), _Parameters(geometry)
        )
    except UnsupportedNativeForm:
        raise
    except Exception as error:
        raise UnsupportedNativeForm(
            f"BilinearForm contains an unsupported operation: {error}"
        ) from error
    if not isinstance(expression, _BilinearTerm):
        raise UnsupportedNativeForm(
            "native BilinearForm currently supports dot(u, v) and "
            "ddot(grad(u), grad(v))"
        )
    coefficient = expression.factor
    if expression.coefficient is not None:
        if isinstance(expression.coefficient,str):
            if expression.coefficient not in kwargs:
                raise ValueError(
                    f"missing form parameter {expression.coefficient!r}"
                )
            raw_coefficient=kwargs[expression.coefficient]
        else:
            raw_coefficient=expression.coefficient
        coefficient = coefficient * np.asarray(
            raw_coefficient, dtype=np.float64
        )
        if coefficient.ndim > 2:
            coefficient = np.squeeze(coefficient)
    native = form._native_cache.get(basis)
    if native is None:
        native = NativeBilinearForm(basis)
        form._native_cache[basis] = native
    if expression.kind == "value":
        return native.assemble(value=coefficient)
    return native.assemble(gradient=coefficient)


def _native_interface_assemble(form,integration,kwargs):
    try:
        expression=form.function(
            _InterfaceTrace("trial"),_InterfaceTrace("test"),_Parameters()
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
            if term.coefficient not in kwargs:
                raise ValueError(f"missing form parameter {term.coefficient!r}")
            coefficient=coefficient*np.asarray(
                kwargs[term.coefficient],dtype=np.float64
            ).squeeze()
        matrix=integration.assemble_traces(
            term.row.weights,term.column.weights,
            row_kind=term.row.kind,column_kind=term.column.kind,
            coefficient=coefficient,
        )
        result=matrix if result is None else result+matrix
    return result


def asm(form, *bases, **kwargs):
    """Assemble strictly with the native backend.

    Unsupported forms raise ``UnsupportedNativeForm``; this function never
    silently delegates assembly to scikit-fem.
    """
    if isinstance(form, _LinearForm) and len(bases) == 1:
        return _native_linear_assemble(form, bases[0], kwargs)
    if isinstance(form, _BilinearForm):
        integration=kwargs.pop("integration",None)
        if integration is not None:
            if len(bases)!=2:
                raise UnsupportedNativeForm(
                    "interface assembly requires master and slave bases"
                )
            return _native_interface_assemble(form,integration,kwargs)
        if len(bases) != 1:
            raise UnsupportedNativeForm(
                "native BilinearForm currently requires one shared basis"
            )
        return _native_bilinear_assemble(form, bases[0], kwargs)
    if isinstance(form, _LinearForm):
        raise UnsupportedNativeForm(
            "native LinearForm currently requires exactly one basis"
        )
    raise TypeError(
        "skfn.asm accepts forms created by skfn.LinearForm or "
        "skfn.BilinearForm; use skfem.asm explicitly for scikit-fem forms"
    )


def dot(left, right):
    if isinstance(left, _Coefficient) and isinstance(right, _TestValue):
        return _Term("value", left)
    if isinstance(right, _Coefficient) and isinstance(left, _TestValue):
        return _Term("value", right)
    if isinstance(right,_TestValue) and isinstance(
        left,(np.ndarray,_QuadratureValue)
    ):
        return _Term("value",np.asarray(left))
    if isinstance(left,_TestValue) and isinstance(
        right,(np.ndarray,_QuadratureValue)
    ):
        return _Term("value",np.asarray(right))
    if (
        isinstance(left, _TrialValue)
        and isinstance(right, _TestValue)
    ) or (
        isinstance(right, _TrialValue)
        and isinstance(left, _TestValue)
    ):
        return _BilinearTerm("value")
    if isinstance(left,_Coefficient) and isinstance(right,_InterfaceTrace):
        if right.kind!="gradient":
            raise UnsupportedNativeForm(
                "an interface coefficient contraction requires grad(field)"
            )
        return _InterfaceCoefficientTrace(right,left.name)
    if isinstance(right,_Coefficient) and isinstance(left,_InterfaceTrace):
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
    if isinstance(left, _Coefficient) and isinstance(right, _TestGradient):
        return _Term("gradient", left)
    if isinstance(right, _Coefficient) and isinstance(left, _TestGradient):
        return _Term("gradient", right)
    if (
        isinstance(left, _TrialGradient)
        and isinstance(right, _TestGradient)
    ) or (
        isinstance(right, _TrialGradient)
        and isinstance(left, _TestGradient)
    ):
        return _BilinearTerm("gradient")
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
        return _TestGradient()
    if isinstance(value, _TrialValue):
        return _TrialGradient()
    if isinstance(value,_InterfaceTrace):
        return value._interface_transform("kind","gradient")
    try:
        return value.grad
    except AttributeError as error:
        raise UnsupportedNativeForm("grad() requires a form field") from error
