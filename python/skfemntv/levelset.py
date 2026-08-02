from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .regions import CellRegion


class CellClassification(IntEnum):
    """Sign classification of a cell sampled at all of its mesh nodes."""

    OUTSIDE = 0
    INSIDE = 1
    CUT = 2
    TOUCHING = 3


@dataclass(frozen=True)
class LevelSetDiagnostics:
    cell_count: int
    node_count: int
    inside_count: int
    outside_count: int
    cut_count: int
    touching_count: int
    tolerance: float
    minimum_value: float
    maximum_value: float


@dataclass(frozen=True)
class CellClassificationResult:
    """Global cell labels and first-class regions derived from a level set."""

    labels: np.ndarray
    inside: CellRegion
    outside: CellRegion
    cut: CellRegion
    touching: CellRegion
    diagnostics: LevelSetDiagnostics

    def region(self,classification: CellClassification | str) -> CellRegion:
        if isinstance(classification,str):
            try:
                classification=CellClassification[classification.upper()]
            except KeyError as error:
                raise ValueError(
                    f"unknown cell classification {classification!r}"
                ) from error
        try:
            value=CellClassification(classification)
        except (TypeError,ValueError) as error:
            raise ValueError(
                f"unknown cell classification {classification!r}"
            ) from error
        return {
            CellClassification.INSIDE:self.inside,
            CellClassification.OUTSIDE:self.outside,
            CellClassification.CUT:self.cut,
            CellClassification.TOUCHING:self.touching,
        }[value]

    @property
    def active(self) -> CellRegion:
        """Cells intersecting or lying on the non-positive side."""
        return self.inside|self.cut|self.touching


class LevelSet:
    """A scalar level set supplied as a callable or global nodal values.

    Negative values define the inside.  Classification samples every node in
    each mesh connectivity column, including high-order nodes.
    """

    def __init__(
        self,field,*,tolerance: float | None=None,
        relative_tolerance: float=64.*np.finfo(np.float64).eps,
    ):
        if not callable(field):
            values=np.asarray(field,dtype=np.float64)
            if values.ndim!=1:
                raise ValueError("level-set nodal values must be one-dimensional")
            field=np.array(values,dtype=np.float64,copy=True)
            field.flags.writeable=False
        if tolerance is not None and (
            not np.isfinite(tolerance) or tolerance<0.
        ):
            raise ValueError("level-set tolerance must be finite and nonnegative")
        if not np.isfinite(relative_tolerance) or relative_tolerance<0.:
            raise ValueError(
                "level-set relative_tolerance must be finite and nonnegative"
            )
        self._field=field
        self.tolerance=None if tolerance is None else float(tolerance)
        self.relative_tolerance=float(relative_tolerance)

    @property
    def is_nodal(self) -> bool:
        return not callable(self._field)

    def values(self,mesh) -> np.ndarray:
        if callable(self._field):
            values=np.asarray(self._field(mesh.p),dtype=np.float64)
            if values.shape==(1,mesh.p.shape[1]):
                values=values[0]
        else:
            values=np.asarray(self._field)
        if values.shape!=(mesh.p.shape[1],):
            raise ValueError(
                "level-set field must produce one scalar per mesh node; "
                f"expected {(mesh.p.shape[1],)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            bad=int(np.flatnonzero(~np.isfinite(values))[0])
            raise ValueError(f"level-set value at node {bad} is not finite")
        result=np.array(values,dtype=np.float64,copy=True)
        result.flags.writeable=False
        return result

    def classify(self,mesh) -> CellClassificationResult:
        values=self.values(mesh)
        value_scale=max(1.,float(np.max(np.abs(values),initial=0.)))
        tolerance=(
            self.tolerance if self.tolerance is not None
            else self.relative_tolerance*value_scale
        )
        cell_values=values[np.asarray(mesh.t,dtype=np.int64)]
        negative=cell_values < -tolerance
        positive=cell_values > tolerance
        has_negative=np.any(negative,axis=0)
        has_positive=np.any(positive,axis=0)
        has_zero=np.any(~(negative|positive),axis=0)

        labels=np.full(
            mesh.nelements,CellClassification.TOUCHING,dtype=np.int8
        )
        labels[np.all(negative,axis=0)]=CellClassification.INSIDE
        labels[np.all(positive,axis=0)]=CellClassification.OUTSIDE
        labels[has_negative&has_positive]=CellClassification.CUT
        # ``has_zero`` is intentionally explicit: negative/zero and
        # positive/zero cells touch the interface without a sampled crossing.
        touching=has_zero&~(has_negative&has_positive)
        labels[touching]=CellClassification.TOUCHING
        labels.flags.writeable=False

        count=int(mesh.nelements)
        make_region=lambda kind:CellRegion(
            np.flatnonzero(labels==kind),count
        )
        inside=make_region(CellClassification.INSIDE)
        outside=make_region(CellClassification.OUTSIDE)
        cut=make_region(CellClassification.CUT)
        touching_region=make_region(CellClassification.TOUCHING)
        diagnostics=LevelSetDiagnostics(
            cell_count=count,node_count=int(mesh.p.shape[1]),
            inside_count=len(inside),outside_count=len(outside),
            cut_count=len(cut),touching_count=len(touching_region),
            tolerance=float(tolerance),minimum_value=float(np.min(values)),
            maximum_value=float(np.max(values)),
        )
        return CellClassificationResult(
            labels,inside,outside,cut,touching_region,diagnostics
        )
