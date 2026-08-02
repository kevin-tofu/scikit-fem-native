from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionDiagnostics:
    entity_count: int | None
    selected_count: int
    is_empty: bool
    minimum_id: int | None
    maximum_id: int | None


class EntityRegion(np.ndarray):
    """Immutable sorted global entity IDs with NumPy compatibility."""

    def __new__(cls,ids,entity_count=None):
        values=np.asarray(ids)
        if values.ndim!=1:
            raise ValueError("region IDs must be one-dimensional")
        if values.size==0:
            values=np.empty(0,dtype=np.int64)
        if values.dtype==bool or not np.issubdtype(values.dtype,np.integer):
            raise TypeError("region IDs must be integers, not a mask")
        values=np.unique(values.astype(np.int64,copy=False))
        count=None if entity_count is None else int(entity_count)
        if count is not None and count<0:
            raise ValueError("entity_count must be nonnegative")
        if np.any(values<0):
            raise ValueError("region IDs must be nonnegative")
        if count is not None and len(values) and values[-1]>=count:
            raise IndexError("region ID is outside entity_count")
        result=np.asarray(values,dtype=np.int64).view(cls)
        result.entity_count=count
        result.flags.writeable=False
        return result

    def __array_finalize__(self,parent):
        if parent is not None:
            self.entity_count=getattr(parent,"entity_count",None)

    def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):
        converted=tuple(
            np.asarray(value) if isinstance(value,EntityRegion) else value
            for value in inputs
        )
        if kwargs.get("out") is not None:
            kwargs["out"]=tuple(
                np.asarray(value) if isinstance(value,EntityRegion) else value
                for value in kwargs["out"]
            )
        return getattr(ufunc,method)(*converted,**kwargs)

    @property
    def ids(self):
        return self

    @property
    def diagnostics(self):
        return RegionDiagnostics(
            entity_count=self.entity_count,
            selected_count=len(self),
            is_empty=len(self)==0,
            minimum_id=int(self[0]) if len(self) else None,
            maximum_id=int(self[-1]) if len(self) else None,
        )

    def _compatible(self,other):
        if type(self) is not type(other):
            raise TypeError("region operations require the same entity kind")
        if (
            self.entity_count is not None
            and other.entity_count is not None
            and self.entity_count!=other.entity_count
        ):
            raise ValueError("region entity counts do not match")
        return (
            self.entity_count
            if self.entity_count is not None else other.entity_count
        )

    def union(self,other):
        count=self._compatible(other)
        return type(self)(np.union1d(self,other),count)

    def intersection(self,other):
        count=self._compatible(other)
        return type(self)(np.intersect1d(self,other),count)

    def difference(self,other):
        count=self._compatible(other)
        return type(self)(np.setdiff1d(self,other),count)

    def complement(self):
        if self.entity_count is None:
            raise ValueError("region complement requires entity_count")
        return type(self)(
            np.setdiff1d(np.arange(self.entity_count),self),
            self.entity_count,
        )

    def __or__(self,other):
        return self.union(other)

    def __and__(self,other):
        return self.intersection(other)

    def __sub__(self,other):
        return self.difference(other)

    def __invert__(self):
        return self.complement()


class CellRegion(EntityRegion):
    pass


class FacetRegion(EntityRegion):
    pass


class NodeRegion(EntityRegion):
    pass
