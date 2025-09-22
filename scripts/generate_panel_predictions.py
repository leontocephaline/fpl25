#!/usr/bin/env python3
"""
Generate per-(player, gameweek) predictions using a time-aware panel from SQLite.

This script:
- Builds a per-(player_id, GW) panel using only pre-GW information
- Trains per-position models (XGBoost/LightGBM) on prior GWs within season
- Predicts for the target GW
- Optionally filters outputs to only players who actually played (for benchmarking)
- Optionally applies a post-hoc calibration layer (linear or isotonic)
- Exports a minimal CSV compatible with scripts/validate_predictions.py

Usage:
  python scripts/generate_panel_predictions.py \
    --season 2024-25 \
    --gw 3 \
    --output-dir data/backtest_sanity_check \
    --model-type lightgbm \
    --calibration none \
    --only-played

Notes:
- For calibration, we currently skip fitting due to preserving row alignment; you can
  enable it to calibrate using training predictions in a subsequent iteration.
- This script is intended for historical backtesting and diagnostic regeneration.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import os

# Ensure project root is on sys.path when executed from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.historical_db import FPLHistoricalDB
from src.data.panel_builder import GWPanelBuilder, PanelConfig
from src.models.ml_predictor import MLPredictor
from src.models.calibration import build_calibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-(player, GW) predictions from SQLite panel")
    parser.add_argument("--season", required=True, help="Season label present in DB, e.g., 2024-25")
    parser.add_argument("--gw", type=int, help="Single target gameweek to predict (1-38)")
    parser.add_argument("--gw-start", type=int, help="Start GW for walk-forward generation (inclusive)")
    parser.add_argument("--gw-end", type=int, help="End GW for walk-forward generation (inclusive)")
    parser.add_argument("--db-path", default=None, help="Optional path to SQLite DB (defaults to data/fpl_history.db)")
    parser.add_argument("--output-dir", default="data/backtest_sanity_check", help="Directory to write predictions CSVs")
    parser.add_argument("--model-type", default="lightgbm", choices=["ensemble", "xgboost", "lightgbm"], help="Model family to use")
    parser.add_argument("--calibration", default="none", choices=["none", "linear", "isotonic"], help="Optional post-hoc calibration")
    parser.add_argument("--objective", default="regression", choices=["regression", "tweedie"], help="Model objective function")
    parser.add_argument("--only-played", action="store_true", help="Export only rows with minutes>0 (benchmarking)")
    parser.add_argument("--drop-features", default=None, help="Comma-separated regex patterns of feature columns to drop for ablations")

    # Hyperparameter overrides
    parser.add_argument("--lgbm-n-estimators", type=int, default=None)
    parser.add_argument("--lgbm-learning-rate", type=float, default=None)
    parser.add_argument("--lgbm-num-leaves", type=int, default=None)
    parser.add_argument("--xgb-n-estimators", type=int, default=None)
    parser.add_argument("--xgb-learning-rate", type=float, default=None)
    parser.add_argument("--xgb-max-depth", type=int, default=None)

    return parser.parse_args()


def _default_ml_config(model_type: str, current_gw: int, args: argparse.Namespace) -> Dict:
    # Conservative defaults for speed and stability
    xgb_params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    lgb_params = {
        "objective": "regression",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return {
        "ml": {
            "model_type": model_type,
            "xgboost": xgb_params,
            "lightgbm": lgb_params,
            "enable_shap": False,
            "enable_prediction_archival": False,  # we export explicitly below
            "save_feature_importance": False,
            "quiet_feature_logs": True,
            "use_global_feature_superset": False,
        },
        "model_save_path": "models/",
        "current_gameweek": int(current_gw),
        # keep reports/plots defaults unused here
    }


def _train_models(predictor: MLPredictor, train_df: pd.DataFrame, target_gw: int) -> None:
    """Train per-position models in-place using MLPredictor."""
    positions = sorted([p for p in train_df["position"].dropna().unique().tolist()])
    for pos in positions:
        pos_df = train_df.loc[train_df["position"] == pos].copy()
        if pos_df.empty:
            continue

        # 1. Train classifier on all players for the position
        print(f"[train] Training classifier for {pos} at GW {target_gw}: rows={len(pos_df)}")
        predictor.train_classifier_model(pos_df, pos, target_gw=int(target_gw))

        # 2. Train regressor only on players who scored points
        pos_train_scored = pos_df[pos_df['total_points'] > 0].copy()
        if len(pos_train_scored) < 5:
            print(f"[train] Skipping regressor for {pos} at GW {target_gw}: insufficient rows ({len(pos_train_scored)})")
            continue
        
        print(f"[train] Training regressor for {pos} at GW {target_gw}: rows={len(pos_train_scored)}")
        predictor.train_position_model(pos_train_scored, pos, feature_lineage={}, target_gw=int(target_gw))


def _predict_for_gw(predictor: MLPredictor, predict_df: pd.DataFrame, gw: int) -> pd.DataFrame:
    preds, _ = predictor.predict_player_points(
        predict_df.copy(), feature_lineage={}, cleaned_stats={}, horizon=int(gw), retrain=False
    )
    # Ensure minimal schema
    if "player_id" not in preds.columns and "id" in preds.columns:
        preds["player_id"] = preds["id"]
    preds["gameweek"] = int(gw)
    # Keep tiny set for validator, but retain position if present
    keep = [c for c in ["player_id", "gameweek", "predicted_points", "position"] if c in preds.columns]
    return preds[keep].copy()


def _fit_calibrator_on_training(predictor: MLPredictor, train_df: pd.DataFrame, target_gw: int, kind: str):
    """Fit an optional calibrator using training predictions vs. actuals.

    We predict on the training panel with the model trained for `target_gw` and
    align with the `total_points` target from `train_df`.

    Returns a fitted calibrator instance (or None).
    """
    calibrator = build_calibrator(kind)
    if calibrator is None:
        return None

    # Drop duplicate columns at the outset to avoid downstream alignment issues
    if not train_df.empty:
        train_df = train_df.loc[:, ~train_df.columns.duplicated()].copy()

    all_preds = []
    all_actuals = []
    positions = sorted([p for p in train_df["position"].dropna().unique().tolist()])
    for pos in positions:
        pos_train = train_df.loc[train_df["position"] == pos].copy()
        if not pos_train.empty:
            pos_train = pos_train.loc[:, ~pos_train.columns.duplicated()].copy()
        if pos_train.empty:
            continue
        model_key = f"{pos}_{int(target_gw)}"
        models = predictor.models.get(model_key, {})
        if not models:
            # Skip if model not present for this position
            continue
        # Compose ensemble prediction consistent with predict_player_points
        xgb_model = models.get("xgb")
        lgbm_model = models.get("lgbm")

        # Align features using trained model's feature names
        def align_for(model, df: pd.DataFrame) -> pd.DataFrame:
            if model is None:
                return pd.DataFrame(index=df.index)
            feat_names = getattr(model, "feature_names_in_", None)
            if feat_names is None:
                return pd.DataFrame(index=df.index)
            # Ensure no duplicate columns in source or targets
            df_nodup = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
            uniq_feats = list(pd.Index(feat_names).drop_duplicates())
            X = df_nodup.reindex(columns=uniq_feats, fill_value=0.0).copy()
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
            return X

        X_xgb = align_for(xgb_model, pos_train)
        X_lgb = align_for(lgbm_model, pos_train)

        # Compute predictions
        xgb_pred = xgb_model.predict(X_xgb) if xgb_model is not None and not X_xgb.empty else None
        lgb_pred = lgbm_model.predict(X_lgb) if lgbm_model is not None and not X_lgb.empty else None

        if xgb_pred is not None and lgb_pred is not None:
            pred = (np.asarray(xgb_pred).reshape(-1) + np.asarray(lgb_pred).reshape(-1)) / 2.0
        elif xgb_pred is not None:
            pred = np.asarray(xgb_pred).reshape(-1)
        elif lgb_pred is not None:
            pred = np.asarray(lgb_pred).reshape(-1)
        else:
            continue

        # Ensure actuals is a Series, not a scalar, and aligned with pos_train
        try:
            if "total_points" in pos_train.columns:
                tp = pos_train["total_points"]
                # Convert robustly to 1-D
                arr = np.asarray(tp)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                actual_series = pd.Series(arr, index=pos_train.index)
            else:
                actual_series = pd.Series(0, index=pos_train.index)
            actual = pd.to_numeric(actual_series, errors="coerce").fillna(0).to_numpy()
        except Exception:
            # Skip this position if we cannot form valid actuals
            continue
        # Guard shape alignment
        n = min(len(pred), len(actual))
        if n == 0:
            continue
        all_preds.append(pred[:n])
        all_actuals.append(actual[:n])

    if not all_preds:
        return None

    x = np.concatenate(all_preds)
    y = np.concatenate(all_actuals)
    # Avoid degenerate fit
    if len(x) < 20:
        return None
    try:
        # RMSE before
        rmse_before = float(np.sqrt(np.nanmean((x - y) ** 2))) if len(x) else None
        calibrator.fit(x, y)
        y_hat = calibrator.transform(x)
        rmse_after = float(np.sqrt(np.nanmean((y_hat - y) ** 2))) if len(y_hat) else None
        print(f"Calibration ({kind}) fit on training: RMSE before={rmse_before:.4f} after={rmse_after:.4f}")
        return calibrator
    except Exception:
        return None


def _apply_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply realistic FPL caps and floors to predicted_points in-place."""
    if df.empty or "predicted_points" not in df.columns:
        return df
    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce").fillna(0).clip(lower=0)
    if "position" in df.columns:
        gk_mask = df["position"].astype(str).str.upper().isin(["GK", "GKP"])
        df.loc[gk_mask, "predicted_points"] = df.loc[gk_mask, "predicted_points"].clip(upper=15)
        df.loc[~gk_mask, "predicted_points"] = df.loc[~gk_mask, "predicted_points"].clip(upper=10)
    else:
        df["predicted_points"] = df["predicted_points"].clip(upper=12)
    return df


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load panel
    db = FPLHistoricalDB(args.db_path) if args.db_path else FPLHistoricalDB()
    builder = GWPanelBuilder(db, PanelConfig())

    def maybe_drop_features(df: pd.DataFrame) -> pd.DataFrame:
        if not args.drop_features:
            return df
        try:
            import re as _re
            patterns = [p.strip() for p in args.drop_features.split(',') if p.strip()]
            drop_cols = []
            for col in df.columns:
                if col in {"player_id", "id", "season", "gw", "position", "element_type", "minutes", "total_points"}:
                    continue
                if any(_re.search(p, col) for p in patterns):
                    drop_cols.append(col)
            if drop_cols:
                df = df.drop(columns=drop_cols)
                print(f"Dropped {len(drop_cols)} features via ablation: first few {drop_cols[:10]}")
        except Exception:
            pass
        return df

    # Determine GW set
    if args.gw is not None:
        gw_list = [int(args.gw)]
    else:
        if args.gw_start is None or args.gw_end is None:
            raise SystemExit("Provide either --gw or both --gw-start and --gw-end")
        gw_list = list(range(int(args.gw_start), int(args.gw_end) + 1))

    for gw in gw_list:
        print(f"\n=== Generating predictions for GW {gw} ===")
        train_df, predict_df = builder.build(args.season, gw)
        print(f"[panel] GW {gw}: train_rows={len(train_df)}, predict_rows={len(predict_df)}")
        try:
            print(f"[panel] GW {gw}: train columns={list(train_df.columns)}")
            print(f"[panel] GW {gw}: predict columns={list(predict_df.columns)}")
            if 'position' in train_df.columns:
                print(f"[panel] GW {gw}: train position unique={sorted(train_df['position'].dropna().unique().tolist())}")
                print(f"[panel] GW {gw}: train position counts=\n{train_df['position'].value_counts(dropna=False).head(10)}")
            if 'position_name' in train_df.columns:
                print(f"[panel] GW {gw}: train position_name unique={sorted(train_df['position_name'].dropna().unique().tolist())}")
            if 'position' in predict_df.columns:
                print(f"[panel] GW {gw}: predict position unique={sorted(predict_df['position'].dropna().unique().tolist())}")
            if 'position_name' in predict_df.columns:
                print(f"[panel] GW {gw}: predict position_name unique={sorted(predict_df['position_name'].dropna().unique().tolist())}")
        except Exception:
            pass
        train_df = maybe_drop_features(train_df)
        predict_df = maybe_drop_features(predict_df)

        # --- Detailed Data Panel Debugging ---
        if gw == 1:
            print("\n--- GW1 PREDICT_DF DEBUG ---")
            print(predict_df[['player_id', 'minutes_roll_mean_5', 'pp90_roll_mean_5', 'prior_minutes_5gw', 'prior_pp90_5gw']].head(20).to_string())
            print("\n--- END GW1 PREDICT_DF DEBUG ---\n")

        # Drop duplicate columns to avoid pandas returning DataFrame for a single column
        if not train_df.empty:
            train_df = train_df.loc[:, ~train_df.columns.duplicated()].copy()
        if not predict_df.empty:
            predict_df = predict_df.loc[:, ~predict_df.columns.duplicated()].copy()

        ml_cfg = _default_ml_config(args.model_type, gw, args)
        predictor = MLPredictor(ml_cfg)
        _train_models(predictor, train_df, target_gw=gw)
# ... (rest of the code remains the same)
        preds = _predict_for_gw(predictor, predict_df, gw)
        print(f"[predict] GW {gw}: preds_rows={len(preds)} cols={list(preds.columns)}")

        if args.only_played and "player_id" in preds.columns:
            try:
                played_ids = set(predict_df.loc[predict_df["minutes"] > 0, "player_id"].dropna().astype(int).tolist())
                preds = preds[preds["player_id"].isin(played_ids)].copy()
            except Exception:
                pass

        # --- Diagnostic: Log raw predictions before calibration ---
        try:
            if not preds.empty:
                print(f"\n--- GW{gw} Raw Prediction Stats (Pre-Calibration) ---")
                print(preds['predicted_points'].describe())
                print("--------------------------------------------------\n")
        except Exception as e:
            print(f"[debug-log] Could not print raw prediction stats: {e}")

        if args.calibration != "none":
            calibrator = _fit_calibrator_on_training(predictor, train_df, gw, args.calibration)
            if calibrator is not None and len(preds) >= 1:
                try:
                    preds["predicted_points"] = calibrator.transform(preds["predicted_points"].to_numpy())
                except Exception:
                    pass

        preds = _apply_caps(preds)

        # --- Diagnostic: Print 10 random predictions and features ---
        try:
            if not preds.empty and not predict_df.empty:
                sample_ids = preds.sample(n=min(10, len(preds)), random_state=42)["player_id"]
                sample_preds = preds[preds["player_id"].isin(sample_ids)]
                sample_features = predict_df[predict_df["player_id"].isin(sample_ids)]

                print(f"\n--- GW{gw} Random Prediction Sample ---")
                for _, row in sample_preds.iterrows():
                    player_id = row["player_id"]
                    pred_points = row["predicted_points"]
                    features = sample_features[sample_features["player_id"] == player_id]
                    # Select only numeric feature columns for concise logging
                    feature_dict = features.select_dtypes(include=np.number).iloc[0].to_dict()
                    # Filter to non-zero features for readability
                    non_zero_features = {k: v for k, v in feature_dict.items() if v != 0}
                    print(f"PlayerID: {player_id}, Predicted: {pred_points:.4f}")
                    print(f"  Non-zero features: {non_zero_features}")
                print("-------------------------------------\n")
        except Exception as e:
            print(f"[debug-log] Could not print random predictions: {e}")
        # --- End Diagnostic ---

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"predictions_gw{gw:02d}_{ts}.csv"
        preds.to_csv(out_path, index=False)

        print(f"Wrote {len(preds)} predictions to {out_path}")
        if len(preds):
            print("Preview:")
            print(preds.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
