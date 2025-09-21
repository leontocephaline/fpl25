"""
Unit tests focusing on:
- Model path resolution preference order (system.model_save_path > model_save_path > default)
- Baseline fallback predictions when no regressors are available

These tests complement existing tests and specifically guard against
regressors-not-found resulting in all-zero outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import sys

# Ensure src/ is importable
ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.ml_predictor import MLPredictor  # type: ignore


@pytest.fixture
def base_cfg() -> Dict:
    # Minimal config dict structure expected by MLPredictor
    return {
        "ml": {
            "model_type": "ensemble",
            "inference_backend": "cpu",
        },
        "system": {
            "model_save_path": "models/"
        }
    }


def test_model_save_path_resolution_prefers_system(tmp_path: Path, base_cfg: Dict) -> None:
    # Prepare a custom path under tmp
    custom_dir = tmp_path / "my_models"
    cfg = dict(base_cfg)
    cfg["system"] = {"model_save_path": str(custom_dir)}
    cfg["ml"] = {"model_type": "xgboost"}
    predictor = MLPredictor(cfg)
    assert predictor.model_save_path == custom_dir
    assert custom_dir.exists() and custom_dir.is_dir()


def test_model_save_path_resolution_fallback(tmp_path: Path, base_cfg: Dict) -> None:
    # Only legacy top-level key
    legacy_dir = tmp_path / "legacy_models"
    cfg = dict(base_cfg)
    cfg["model_save_path"] = str(legacy_dir)
    cfg["ml"] = {"model_type": "lightgbm"}
    predictor = MLPredictor(cfg)
    assert predictor.model_save_path == legacy_dir
    assert legacy_dir.exists() and legacy_dir.is_dir()


def test_baseline_fallback_when_no_regressors(tmp_path: Path, base_cfg: Dict) -> None:
    # Point model path at an empty directory so no models are discovered
    empty_models_dir = tmp_path / "empty_models"
    cfg = dict(base_cfg)
    cfg["system"] = {"model_save_path": str(empty_models_dir)}
    cfg["ml"] = {"model_type": "ensemble", "inference_backend": "cpu"}
    cfg["current_gameweek"] = 3
    predictor = MLPredictor(cfg)

    # Minimal realistic inputs reflecting DataProcessor outputs
    # Use one position to keep shapes simple
    df = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "position": ["MID", "MID", "MID"],
            # Signals used by the baseline (original df, not X)
            "points_per_game": [4.0, 6.0, 2.0],
            "form_numeric": [5.0, 1.0, 3.0],
            # Fixture horizon difficulty as engineered in DataProcessor
            "fixture_difficulty_next_4": [2.0, 3.0, 4.0],
            # Extra numeric columns so prepare_features() has something to pass to classifiers (if any)
            "minutes": [900, 800, 700],
            "total_points": [40, 50, 15],
        }
    )

    preds, _ = predictor.predict_player_points(
        df.copy(), feature_lineage={}, cleaned_stats={}, horizon=3, retrain=False
    )

    # Should produce predictions even with no regressors discovered
    assert not preds.empty, "Baseline fallback should yield non-empty predictions"
    assert set(["id", "predicted_points"]).issubset(set(preds.columns))

    # All predictions must be bounded and non-negative (baseline clamps to a plausible range)
    assert (preds["predicted_points"] >= 0).all()
    assert (preds["predicted_points"] <= 20).all()

    # Baseline fallback sets a low prediction_confidence if available
    if "prediction_confidence" in preds.columns:
        assert float(preds["prediction_confidence"].median()) <= 0.25

    # Ensure not all zeros
    assert (preds["predicted_points"] > 0).any(), "Predictions should not be all zeros with baseline fallback"
