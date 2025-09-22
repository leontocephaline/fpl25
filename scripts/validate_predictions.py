#!/usr/bin/env python3
"""
Validation script to compare predicted fantasy football points against actual results
across multiple gameweeks.

Usage:
  python scripts/validate_predictions.py \
    --predictions-dir data/backtest_sanity_check \
    --actuals-file data/actual_results_2024-25.csv \
    --output-dir reports/validation_sanity

This script is robust to two prediction file formats:
- Minimal format (e.g., sanity check): columns include `player_id, gameweek, predicted_points[, actual_points]`
- Full format (e.g., backtest): many columns, uses `id` as player id, has `position`, `predicted_points`.
  In this case, gameweek is inferred from the filename pattern: predictions_gw<NN>_*.csv

Checks performed per gameweek:
- Column availability and type checks
- Duplicate keys in predictions (by player_id)
- Coverage vs actuals (missing_in_predictions, extra_in_predictions)
- If predictions contain `actual_points`, verify against actuals
- Accuracy metrics: MAE, RMSE, mean error, correlation (Pearson), and coverage rate

Outputs:
- <output-dir>/per_gw/<gw>_missing_in_predictions.csv
- <output-dir>/per_gw/<gw>_extra_in_predictions.csv
- <output-dir>/per_gw/<gw>_mismatched_actuals.csv (only if prediction has actual_points column)
- <output-dir>/summary.csv (aggregated metrics across gameweeks)
- Console log with a concise summary
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Ensure project root is on sys.path when executed from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


GW_FILENAME_RE = re.compile(r"predictions_gw(\d{1,2})_.*\.csv$", re.IGNORECASE)


@dataclass
class GWReport:
    gameweek: int
    n_predictions: int
    n_actuals: int
    coverage_rate: float
    n_missing_in_predictions: int
    n_extra_in_predictions: int
    n_duplicates: int
    mae: Optional[float]
    rmse: Optional[float]
    mean_error: Optional[float]
    pearson_corr: Optional[float]
    mismatched_actual_points: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate predictions vs actual results across gameweeks")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing prediction CSV files")
    parser.add_argument("--actuals-file", required=True, help="CSV file with actual results")
    parser.add_argument("--output-dir", required=True, help="Directory to write validation reports")
    parser.add_argument("--file-glob", default="predictions_gw*.csv", help="Glob/pattern to select prediction files (default: predictions_gw*.csv)")
    parser.add_argument("--season", default=None, help="Optional season string to filter actuals if the actuals file contains a season column (e.g., 2024-25)")
    parser.add_argument("--per-position", action="store_true", help="Emit per-position metrics per GW and an aggregated summary if position is available in predictions")
    parser.add_argument("--dist-stats", action="store_true", help="Emit prediction distribution statistics per GW (and by position if available)")
    return parser.parse_args()


def ensure_output_dirs(base_out: str) -> str:
    per_gw = os.path.join(base_out, "per_gw")
    os.makedirs(per_gw, exist_ok=True)
    return per_gw


def infer_gameweek_from_name(filename: str) -> Optional[int]:
    m = GW_FILENAME_RE.search(os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1))


def load_actuals(actuals_path: str, season: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(actuals_path)

    required_cols_any = [{"player_id", "gameweek", "actual_points"}, {"player_id", "season", "gameweek", "actual_points"}]
    cols = set(df.columns)
    if not any(req.issubset(cols) for req in required_cols_any):
        raise ValueError(
            f"Actuals file {actuals_path} must contain columns either (player_id, gameweek, actual_points) or (player_id, season, gameweek, actual_points). Found: {sorted(df.columns)}"
        )

    # Filter by season if provided and column exists
    if season and "season" in df.columns:
        df = df[df["season"].astype(str) == str(season)].copy()

    # Normalize dtypes
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")
    df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce")

    # Keep only necessary columns
    keep_cols = ["player_id", "gameweek", "actual_points"]
    df = df[keep_cols].dropna(subset=["player_id", "gameweek"])  # retain rows with valid keys
    return df


def read_predictions(pred_path: str, inferred_gw: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Load a predictions CSV and normalize columns.

    Returns:
        df_norm: DataFrame with columns [player_id, gameweek, predicted_points, position? (optional), actual_points? (optional)]
        meta: dict flags about columns present
    """
    df = pd.read_csv(pred_path, low_memory=False)
    cols = set(df.columns)

    # Map player id column
    if "player_id" in df.columns:
        player_id_col = "player_id"
    elif "id" in df.columns:
        player_id_col = "id"
    else:
        raise ValueError(f"{pred_path}: Could not find player id column. Expected one of ['player_id', 'id']. Found: {sorted(df.columns)}")

    # Determine gameweek
    if "gameweek" in df.columns:
        df_gw = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")
    else:
        if inferred_gw is None:
            raise ValueError(f"{pred_path}: No 'gameweek' column and could not infer from filename.")
        df_gw = pd.Series([inferred_gw] * len(df), index=df.index, dtype="Int64")

    # Predicted points
    if "predicted_points" not in df.columns:
        raise ValueError(f"{pred_path}: Missing required 'predicted_points' column.")

    df_out = pd.DataFrame(
        {
            "player_id": pd.to_numeric(df[player_id_col], errors="coerce").astype("Int64"),
            "gameweek": df_gw,
            "predicted_points": pd.to_numeric(df["predicted_points"], errors="coerce"),
        }
    )

    # Optional columns
    if "actual_points" in df.columns:
        df_out["actual_points_pred_file"] = pd.to_numeric(df["actual_points"], errors="coerce")

    # Position if available (direct or via element_type mapping)
    if "position" in df.columns:
        df_out["position"] = df["position"].astype(str)
    elif "element_type" in df.columns:
        # Map FPL element_type to position labels
        mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        df_out["position"] = pd.to_numeric(df["element_type"], errors="coerce").map(mapping).astype("string")

    meta = {
        "had_gameweek_col": "gameweek" in cols,
        "had_actual_points": "actual_points" in cols,
        "had_position": ("position" in cols) or ("element_type" in cols),
        "player_id_col": player_id_col,
    }

    # Drop rows with invalid keys
    df_out = df_out.dropna(subset=["player_id", "gameweek"]).copy()
    return df_out, meta


def accuracy_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if len(y_true) == 0:
        return None, None, None, None
    err = y_pred - y_true
    mae = float(np.nanmean(np.abs(err))) if np.isfinite(err).any() else None
    rmse = float(np.sqrt(np.nanmean(err ** 2))) if np.isfinite(err).any() else None
    mean_error = float(np.nanmean(err)) if np.isfinite(err).any() else None
    # Safe Pearson: avoid divide-by-zero when either series is constant
    pearson: Optional[float]
    try:
        ys = pd.Series(pd.to_numeric(y_true, errors="coerce"))
        ps = pd.Series(pd.to_numeric(y_pred, errors="coerce"))
        # Need at least 2 unique finite values in both series
        if ys.dropna().nunique() < 2 or ps.dropna().nunique() < 2 or len(ys) < 2:
            pearson = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pearson = float(ps.corr(ys))
    except Exception:
        pearson = None
    return mae, rmse, mean_error, pearson


def validate_one_gw(pred_path: str, actuals_df: pd.DataFrame) -> Tuple[GWReport, Dict[str, pd.DataFrame]]:
    inferred_gw = infer_gameweek_from_name(pred_path)
    pred_df, meta = read_predictions(pred_path, inferred_gw)
    gw = int(pred_df["gameweek"].iat[0]) if len(pred_df) else (inferred_gw or -1)

    # Duplicate check by player_id within GW
    dup_mask = pred_df.duplicated(subset=["player_id"], keep=False)
    n_duplicates = int(dup_mask.sum())

    # Merge with actuals for this GW
    act_gw = actuals_df[actuals_df["gameweek"] == gw].copy()

    merged = pred_df.merge(act_gw, on=["player_id", "gameweek"], how="outer", indicator=True)
    # Coverage
    missing_in_predictions = merged[merged["_merge"] == "right_only"][["player_id", "gameweek", "actual_points"]]
    extra_in_predictions = merged[merged["_merge"] == "left_only"][
        [c for c in ["player_id", "gameweek", "predicted_points", "position"] if c in merged.columns]
    ]

    # Rows present in both for metrics
    both = merged[merged["_merge"] == "both"].copy()

    # Compare actual_points if present in prediction file
    mismatched_actuals = pd.DataFrame()
    if "actual_points_pred_file" in both.columns:
        mismatched_actuals = both[
            (both["actual_points_pred_file"].notna())
            & (both["actual_points"].notna())
            & (both["actual_points_pred_file"] != both["actual_points"])
        ][["player_id", "gameweek", "actual_points_pred_file", "actual_points"]]

    # Compute metrics where we have both predicted and actuals
    metrics_df = both.dropna(subset=["predicted_points", "actual_points"]).copy()
    mae, rmse, mean_error, pearson = accuracy_metrics(metrics_df["actual_points"], metrics_df["predicted_points"]) if len(metrics_df) else (None, None, None, None)

    n_actuals = int(len(act_gw))
    n_preds = int(len(pred_df))
    coverage_rate = float(len(both) / n_actuals) if n_actuals > 0 else 0.0

    report = GWReport(
        gameweek=gw,
        n_predictions=n_preds,
        n_actuals=n_actuals,
        coverage_rate=coverage_rate,
        n_missing_in_predictions=int(len(missing_in_predictions)),
        n_extra_in_predictions=int(len(extra_in_predictions)),
        n_duplicates=n_duplicates,
        mae=mae,
        rmse=rmse,
        mean_error=mean_error,
        pearson_corr=pearson,
        mismatched_actual_points=int(len(mismatched_actuals)) if "actual_points_pred_file" in both.columns else None,
    )

    artifacts = {
        "missing_in_predictions": missing_in_predictions,
        "extra_in_predictions": extra_in_predictions,
        "mismatched_actuals": mismatched_actuals,
    }
    return report, artifacts


def write_artifacts(per_gw_dir: str, gw: int, artifacts: Dict[str, pd.DataFrame]) -> None:
    base = os.path.join(per_gw_dir, f"gw{gw:02d}")
    os.makedirs(base, exist_ok=True)
    for name, df in artifacts.items():
        if df is None or len(df) == 0:
            # Write an empty file to indicate no issues for traceability
            out_path = os.path.join(base, f"{name}.csv")
            pd.DataFrame().to_csv(out_path, index=False)
        else:
            out_path = os.path.join(base, f"{name}.csv")
            df.to_csv(out_path, index=False)


def summarize_reports(reports: List[GWReport]) -> pd.DataFrame:
    rows = []
    for r in sorted(reports, key=lambda x: x.gameweek):
        rows.append(
            {
                "gameweek": r.gameweek,
                "n_predictions": r.n_predictions,
                "n_actuals": r.n_actuals,
                "coverage_rate": r.coverage_rate,
                "n_missing_in_predictions": r.n_missing_in_predictions,
                "n_extra_in_predictions": r.n_extra_in_predictions,
                "n_duplicates": r.n_duplicates,
                "mae": r.mae,
                "rmse": r.rmse,
                "mean_error": r.mean_error,
                "pearson_corr": r.pearson_corr,
                "mismatched_actual_points": r.mismatched_actual_points,
            }
        )
    return pd.DataFrame(rows)


def compute_position_metrics(both_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute per-position metrics (on rows where both predicted and actuals are present).

    Returns None if position column is not available.
    """
    if "position" not in both_df.columns:
        return None
    df = both_df.dropna(subset=["predicted_points", "actual_points"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["position", "count", "mae", "rmse", "mean_error"])
    rows = []
    for pos, g in df.groupby("position"):
        y_true = pd.to_numeric(g["actual_points"], errors="coerce")
        y_pred = pd.to_numeric(g["predicted_points"], errors="coerce")
        err = y_pred - y_true
        mae = float(np.nanmean(np.abs(err))) if np.isfinite(err).any() else None
        rmse = float(np.sqrt(np.nanmean(err ** 2))) if np.isfinite(err).any() else None
        mean_error = float(np.nanmean(err)) if np.isfinite(err).any() else None
        rows.append({
            "position": str(pos),
            "count": int(len(g)),
            "mae": mae,
            "rmse": rmse,
            "mean_error": mean_error,
        })
    return pd.DataFrame(rows).sort_values(["position"]).reset_index(drop=True)


def compute_distribution_stats(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution statistics for predicted_points overall and by position if present."""
    qtiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    cols = ["min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std", "count"]

    def summarize(series: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(series, errors="coerce")
        desc = {
            "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
            "p10": float(np.nanquantile(s, 0.10)) if s.notna().any() else np.nan,
            "p25": float(np.nanquantile(s, 0.25)) if s.notna().any() else np.nan,
            "p50": float(np.nanquantile(s, 0.50)) if s.notna().any() else np.nan,
            "p75": float(np.nanquantile(s, 0.75)) if s.notna().any() else np.nan,
            "p90": float(np.nanquantile(s, 0.90)) if s.notna().any() else np.nan,
            "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
            "mean": float(np.nanmean(s)) if s.notna().any() else np.nan,
            "std": float(np.nanstd(s)) if s.notna().any() else np.nan,
            "count": int(s.notna().sum()),
        }
        return desc

    rows: List[Dict[str, float | str]] = []
    # Overall
    overall = summarize(pred_df["predicted_points"])
    rows.append({"position": "ALL", **overall})

    # By position if available
    if "position" in pred_df.columns:
        for pos, g in pred_df.groupby("position"):
            rows.append({"position": str(pos), **summarize(g["predicted_points"])})

    out = pd.DataFrame(rows)
    # Ensure column order
    return out[["position", *cols]]


def find_prediction_files(predictions_dir: str, pattern: str) -> List[str]:
    # Simple pattern filter without glob to keep dependencies minimal
    # Expect pattern like 'predictions_gw' and '.csv'
    files = [os.path.join(predictions_dir, f) for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    # Prioritize consistent naming: only files containing 'predictions_gw'
    if "predictions_gw" in pattern:
        files = [f for f in files if os.path.basename(f).startswith("predictions_gw")]

    # If multiple files exist for the same GW, select the latest by filename (timestamps embedded)
    by_gw: Dict[int, List[str]] = {}
    for f in files:
        gw = infer_gameweek_from_name(f)
        if gw is None:
            continue
        by_gw.setdefault(gw, []).append(f)

    selected: List[str] = []
    for gw, paths in by_gw.items():
        # Sort by basename to leverage the embedded timestamp and pick the lexicographically last (latest)
        latest = sorted(paths, key=lambda p: os.path.basename(p))[-1]
        selected.append(latest)

    # Include any files that didn't match the GW pattern (fallback)
    unmatched = [f for f in files if infer_gameweek_from_name(f) is None]
    selected.extend(unmatched)

    return sorted(selected)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    per_gw_dir = ensure_output_dirs(args.output_dir)

    # Load and normalize actuals
    actuals_df = load_actuals(args.actuals_file, args.season)

    # Discover prediction files
    pred_files = find_prediction_files(args.predictions_dir, args.file_glob)
    if not pred_files:
        raise SystemExit(f"No prediction files found in {args.predictions_dir} matching pattern {args.file_glob}")

    reports: List[GWReport] = []

    # Accumulators for aggregated diagnostics
    all_pos_metrics: List[pd.DataFrame] = []
    all_dist_stats: List[pd.DataFrame] = []

    for pred_path in pred_files:
        try:
            report, artifacts = validate_one_gw(pred_path, actuals_df)
            # Diagnostics: per-position metrics and distribution stats
            if args.per_position or args.dist_stats:
                # We need to recompute merge here to access both and pred_df
                inferred_gw = infer_gameweek_from_name(pred_path)
                pred_df, _ = read_predictions(pred_path, inferred_gw)
                gw = int(pred_df["gameweek"].iat[0]) if len(pred_df) else (inferred_gw or -1)
                act_gw = actuals_df[actuals_df["gameweek"] == gw].copy()
                merged = pred_df.merge(act_gw, on=["player_id", "gameweek"], how="outer", indicator=True)
                both = merged[merged["_merge"] == "both"].copy()

                if args.per_position:
                    pos_df = compute_position_metrics(both)
                    if pos_df is not None:
                        pos_df.insert(0, "gameweek", report.gameweek)
                        artifacts["position_metrics"] = pos_df
                        all_pos_metrics.append(pos_df)

                if args.dist_stats:
                    dist_df = compute_distribution_stats(pred_df)
                    dist_df.insert(0, "gameweek", report.gameweek)
                    artifacts["predicted_distribution"] = dist_df
                    all_dist_stats.append(dist_df)

            reports.append(report)
            write_artifacts(per_gw_dir, report.gameweek, artifacts)
            print(
                f"GW {report.gameweek:02d} | preds={report.n_predictions} actuals={report.n_actuals} "
                f"coverage={report.coverage_rate:.3f} dup={report.n_duplicates} "
                f"MAE={report.mae if report.mae is not None else 'NA'} RMSE={report.rmse if report.rmse is not None else 'NA'}"
            )
        except Exception as e:
            # Write an error marker file for traceability
            gw = infer_gameweek_from_name(os.path.basename(pred_path)) or -1
            err_dir = os.path.join(per_gw_dir, f"gw{gw:02d}")
            os.makedirs(err_dir, exist_ok=True)
            with open(os.path.join(err_dir, "error.txt"), "w", encoding="utf-8") as fh:
                fh.write(str(e))
            print(f"Error processing {pred_path}: {e}")

    # Summary
    summary_df = summarize_reports(reports)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Write aggregated diagnostics if requested
    if all_pos_metrics:
        pos_summary = pd.concat(all_pos_metrics, ignore_index=True)
        pos_summary_path = os.path.join(args.output_dir, "per_position_summary.csv")
        pos_summary.to_csv(pos_summary_path, index=False)
    if all_dist_stats:
        dist_summary = pd.concat(all_dist_stats, ignore_index=True)
        dist_summary_path = os.path.join(args.output_dir, "predicted_distribution_summary.csv")
        dist_summary.to_csv(dist_summary_path, index=False)

    if len(summary_df):
        overall = {
            "gameweeks": int(summary_df["gameweek"].nunique()),
            "avg_coverage": float(summary_df["coverage_rate"].mean(skipna=True)),
            "avg_mae": float(summary_df["mae"].mean(skipna=True)) if summary_df["mae"].notna().any() else None,
            "avg_rmse": float(summary_df["rmse"].mean(skipna=True)) if summary_df["rmse"].notna().any() else None,
        }
        print("\nOverall Summary:")
        for k, v in overall.items():
            print(f"- {k}: {v}")
        print(f"Summary written to: {summary_path}")
    else:
        print("No reports generated.")


if __name__ == "__main__":
    main()
