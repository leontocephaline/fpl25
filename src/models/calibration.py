"""
Calibration utilities for mapping raw model predictions to better-aligned values.

Two strategies are provided:
- Linear: Ordinary least squares y = a*x + b
- Isotonic: Monotonic, non-parametric calibration suitable when rank is decent

These utilities expose a minimal, sklearn-like API with `.fit(X, y)` and
`.transform(x)` to be easy to integrate in pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


@dataclass
class LinearCalibrator:
    """Linear calibration y = a*x + b.

    Fits a simple linear regression mapping from predictions to actuals.
    """
    model: Optional[LinearRegression] = None

    def fit(self, preds: np.ndarray, actuals: np.ndarray) -> "LinearCalibrator":
        x = np.asarray(preds).reshape(-1, 1)
        y = np.asarray(actuals)
        self.model = LinearRegression()
        self.model.fit(x, y)
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.asarray(preds)
        x = np.asarray(preds).reshape(-1, 1)
        return self.model.predict(x)


@dataclass
class IsotonicCalibrator:
    """Monotonic calibration using isotonic regression.

    Useful when monotonicity between raw predictions and actuals is expected,
    but the mapping is non-linear.
    """
    iso: Optional[IsotonicRegression] = None

    def fit(self, preds: np.ndarray, actuals: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(preds)
        y = np.asarray(actuals)
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(x, y)
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.asarray(preds)
        x = np.asarray(preds)
        return self.iso.transform(x)


def build_calibrator(kind: str) -> Optional[object]:
    """Factory for calibrators.

    Args:
        kind: One of {"none", "linear", "isotonic"} (case-insensitive)

    Returns:
        A calibrator instance or None when kind == "none".
    """
    k = (kind or "none").strip().lower()
    if k == "none":
        return None
    if k == "linear":
        return LinearCalibrator()
    if k == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibrator kind: {kind}")
