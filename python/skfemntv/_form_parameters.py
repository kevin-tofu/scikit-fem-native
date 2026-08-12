"""Parameter namespace and numerical values used during form tracing."""

from __future__ import annotations

import numpy as np

from ._coefficients import Coefficient
from ._composite_fields import CompositeField,CompositeWeightedField
from ._form_terms import BilinearTerm,CompositeBilinearTerm
from ._h1_fields import Divergence
from ._interface_terms import InterfaceBilinearTerm


class QuadratureValue:
    """Numerical form value preserving typed multiplication semantics."""

    __array_priority__=1000

    def __init__(self,value,*,parameter_name=None):
        self.value=np.asarray(value)
        self.parameter_name=parameter_name

    def __array__(self,dtype=None):
        return np.asarray(self.value,dtype=dtype)

    def __getitem__(self,key):
        if self.parameter_name is not None:
            if isinstance(key,bool) or not isinstance(key,(int,np.integer)):
                raise ValueError(
                    f"form parameter {self.parameter_name!r} component index "
                    "must be an integer"
                )
            try:
                value=self.value[key]
            except IndexError as error:
                raise ValueError(
                    f"form parameter {self.parameter_name!r} has no "
                    f"component {key}"
                ) from error
            return QuadratureValue(value)
        return QuadratureValue(self.value[key])

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        values=[
            item.value if isinstance(item,QuadratureValue) else item
            for item in inputs
        ]
        return QuadratureValue(getattr(ufunc,method)(*values,**kwargs))

    def __mul__(self,other):
        if isinstance(other,(
            BilinearTerm,CompositeBilinearTerm,InterfaceBilinearTerm,
        )):
            return other*np.asarray(self.value)
        if isinstance(other,CompositeField):
            return CompositeWeightedField(other,np.asarray(self.value))
        if isinstance(other,Divergence):
            return Divergence(other.role,other.factor,np.asarray(self.value))
        return QuadratureValue(self.value*np.asarray(other))

    def __rmul__(self,other):
        if isinstance(other,(
            BilinearTerm,CompositeBilinearTerm,InterfaceBilinearTerm,
        )):
            return other*np.asarray(self.value)
        return QuadratureValue(np.asarray(other)*self.value)

    def __add__(self,other):
        return QuadratureValue(self.value+np.asarray(other))

    def __radd__(self,other):
        return QuadratureValue(np.asarray(other)+self.value)

    def __sub__(self,other):
        return QuadratureValue(self.value-np.asarray(other))

    def __rsub__(self,other):
        return QuadratureValue(np.asarray(other)-self.value)

    def __truediv__(self,other):
        return QuadratureValue(self.value/np.asarray(other))

    def __rtruediv__(self,other):
        return QuadratureValue(np.asarray(other)/self.value)

    def __pow__(self,other):
        return QuadratureValue(self.value**other)

    def __neg__(self):
        return QuadratureValue(-self.value)


class Parameters:
    """Attribute/item namespace returning symbolic descriptors when absent."""

    def __init__(self,geometry=None):
        self._geometry={} if geometry is None else geometry

    def __getattr__(self,name):
        if name in self._geometry:
            return self._geometry[name]
        return Coefficient(name)

    def __getitem__(self,name):
        if name in self._geometry:
            return self._geometry[name]
        return Coefficient(name)


def parameter_values(geometry,kwargs):
    """Wrap user parameters while retaining discrete fields and callables."""
    values=dict(geometry)
    for name,value in kwargs.items():
        values[name]=(
            value
            if callable(value) or (
                hasattr(value,"value") and hasattr(value,"grad")
            )
            else QuadratureValue(value,parameter_name=name)
        )
    return values


__all__=["Parameters","QuadratureValue","parameter_values"]
