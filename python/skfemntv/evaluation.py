from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationDiagnostics:
    element_count: int
    quadrature_evaluations: int
    assembly_seconds: float
    kernel_seconds: float | None = None
    scatter_seconds: float | None = None
    invalid_element_count: int = 0
    material_failure_count: int = 0


@dataclass(frozen=True)
class NativeEvaluation:
    residual: np.ndarray
    tangent: Any | None
    trial_state: None
    diagnostics: EvaluationDiagnostics
