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
    def __new__(
        cls,ids,entity_count=None,*,sides=None,normal_signs=None
    ):
        original=np.asarray(ids)
        if original.ndim!=1:
            raise ValueError("region IDs must be one-dimensional")
        original_sides=(
            None if sides is None else np.asarray(sides,dtype=np.int8)
        )
        original_signs=(
            None if normal_signs is None
            else np.asarray(normal_signs,dtype=np.int8)
        )
        if original_sides is not None and original_sides.shape!=original.shape:
            raise ValueError("facet sides must contain one value per input facet")
        if original_signs is not None and original_signs.shape!=original.shape:
            raise ValueError(
                "facet normal_signs must contain one value per input facet"
            )
        result=super().__new__(cls,ids,entity_count)
        count=len(result)
        normalized_sides=(
            np.zeros(count,dtype=np.int8) if sides is None
            else np.empty(count,dtype=np.int8)
        )
        normalized_signs=(
            np.ones(count,dtype=np.int8) if normal_signs is None
            else np.empty(count,dtype=np.int8)
        )
        if count and (sides is not None or normal_signs is not None):
            metadata={}
            for index,entity in enumerate(original):
                value=(
                    int(original_sides[index])
                    if original_sides is not None else 0,
                    int(original_signs[index])
                    if original_signs is not None else 1,
                )
                key=int(entity)
                if key in metadata and metadata[key]!=value:
                    raise ValueError(
                        "duplicate facet has conflicting orientation metadata"
                    )
                metadata[key]=value
            for index,entity in enumerate(np.asarray(result)):
                normalized_sides[index],normalized_signs[index]=metadata[
                    int(entity)
                ]
        if np.any((normalized_sides<0)|(normalized_sides>1)):
            raise ValueError("facet sides must contain one 0 or 1 per facet")
        if np.any((normalized_signs!=-1)&(normalized_signs!=1)):
            raise ValueError(
                "facet normal_signs must contain one -1 or 1 per facet"
            )
        normalized_sides.flags.writeable=False
        normalized_signs.flags.writeable=False
        result.sides=normalized_sides
        result.normal_signs=normalized_signs
        return result

    def __array_finalize__(self,parent):
        super().__array_finalize__(parent)
        if parent is not None:
            self.sides=getattr(parent,"sides",None)
            self.normal_signs=getattr(parent,"normal_signs",None)

    def __getitem__(self,key):
        result=super().__getitem__(key)
        if isinstance(result,FacetRegion):
            result.sides=np.asarray(self.sides)[key].reshape(-1)
            result.normal_signs=np.asarray(self.normal_signs)[key].reshape(-1)
            result.sides.flags.writeable=False
            result.normal_signs.flags.writeable=False
        return result

    def _metadata(self,ids):
        positions=np.searchsorted(np.asarray(self),ids)
        return self.sides[positions],self.normal_signs[positions]

    @staticmethod
    def _require_matching(ids,left,right):
        if not len(ids): return
        left_sides,left_signs=left._metadata(ids)
        right_sides,right_signs=right._metadata(ids)
        if not (
            np.array_equal(left_sides,right_sides)
            and np.array_equal(left_signs,right_signs)
        ):
            raise ValueError("facet region orientation metadata conflicts")

    def union(self,other):
        count=self._compatible(other)
        common=np.intersect1d(self,other)
        self._require_matching(common,self,other)
        ids=np.union1d(self,other)
        sides=np.empty(len(ids),dtype=np.int8)
        signs=np.empty(len(ids),dtype=np.int8)
        for source in (self,other):
            positions=np.searchsorted(ids,np.asarray(source))
            sides[positions]=source.sides
            signs[positions]=source.normal_signs
        return type(self)(
            ids,count,sides=sides,normal_signs=signs
        )

    def intersection(self,other):
        count=self._compatible(other)
        ids=np.intersect1d(self,other)
        self._require_matching(ids,self,other)
        sides,signs=self._metadata(ids)
        return type(self)(ids,count,sides=sides,normal_signs=signs)

    def difference(self,other):
        count=self._compatible(other)
        ids=np.setdiff1d(self,other)
        sides,signs=self._metadata(ids)
        return type(self)(ids,count,sides=sides,normal_signs=signs)


class NodeRegion(EntityRegion):
    pass
