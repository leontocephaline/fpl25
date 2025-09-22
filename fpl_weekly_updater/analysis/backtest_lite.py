from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:  # pragma: no cover
    plt = None
    PdfPages = None


@dataclass
class BacktestResults:
    overall: Dict[str, float]
    by_position: Dict[str, Dict[str, float]]
    by_gameweek: Dict[int, Dict[str, float]]
    drift_alerts: List[str]
    md_path: Optional[Path] = None
    pdf_path: Optional[Path] = None


def _position_from_element_type(v: object) -> Optional[str]:
    try:
        et = int(v)  # type: ignore[arg-type]
    except Exception:
        return None
    return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(et)


def _load_predictions(predictions_dir: Path, gameweek_range: Tuple[int, int]) -> pd.DataFrame:
    files = sorted(predictions_dir.glob("predictions_gw*.csv"))
    if not files:
        return pd.DataFrame()
    gw_start, gw_end = gameweek_range
    frames: List[pd.DataFrame] = []
    for f in files:
        # Extract GW from filename: predictions_gw<NN>_timestamp.csv
        gw = None
        stem = f.stem
        try:
            parts = stem.split("_")
            gw = int(parts[1].replace("gw", "").lstrip("0") or "0")
        except Exception:
            gw = None
        if gw is None or gw < gw_start or gw > gw_end:
            continue
        try:
            df = pd.read_csv(f)
            df["__source_file"] = str(f)
            # Standardize identifiers
            if "id" not in df.columns and "player_id" in df.columns:
                df["id"] = df["player_id"]
            if "player_id" not in df.columns and "id" in df.columns:
                df["player_id"] = df["id"]
            # Standardize gameweek column
            if "gameweek" not in df.columns:
                df["gameweek"] = gw
            # Position fallback from element_type if needed
            if "position" not in df.columns and "element_type" in df.columns:
                df["position"] = df["element_type"].apply(_position_from_element_type)
            # Require minimal columns
            cols_needed = {"player_id", "gameweek", "predicted_points"}
            if not cols_needed.issubset(set(df.columns)):
                cand_cols = [c for c in df.columns if "pred" in c.lower() and "point" in c.lower()]
                if cand_cols:
                    df["predicted_points"] = df[cand_cols[0]]
            if cols_needed.issubset(set(df.columns)):
                frames.append(df[list(set(df.columns))])
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load predictions from %s: %s", f, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["actual_points"].astype(float).to_numpy()
    y_pred = df["predicted_points"].astype(float).to_numpy()
    n = len(y_true)
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "mean_bias": float("nan"), "n_predictions": 0}
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    mean_bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mean_bias": mean_bias, "n_predictions": int(n)}


def _auto_detect_gw_range(actuals: pd.DataFrame, predictions_dir: Path) -> Optional[Tuple[int, int]]:
    gws_pred = []
    for f in predictions_dir.glob("predictions_gw*.csv"):
        try:
            parts = f.stem.split("_")
            gw = int(parts[1].replace("gw", "").lstrip("0") or "0")
            gws_pred.append(gw)
        except Exception:
            continue
    gws_pred = sorted(set(gws_pred))
    if not gws_pred:
        return None
    if "gameweek" in actuals.columns:
        gws_act = sorted(set(int(x) for x in pd.to_numeric(actuals["gameweek"], errors="coerce").dropna().astype(int)))
        common = [gw for gw in gws_pred if gw in gws_act]
        if common:
            return (min(common), max(common))
    return (min(gws_pred), max(gws_pred))


def _render_pdf(results: BacktestResults, output_dir: Path) -> Optional[Path]:
    if plt is None or PdfPages is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_dir / f"backtest_analysis_lite_{ts}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        overall = results.overall
        lines: List[str] = []
        lines.append("FANTASY FOOTBALL ML PIPELINE REPORT (Lite)")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("Executive Summary")
        lines.append(f"Predictions merged: {int(overall.get('n_predictions', 0)):,}")
        lines.append(f"RMSE: {overall.get('rmse', float('nan')):.3f}")
        lines.append(f"MAE: {overall.get('mae', float('nan')):.3f}")
        lines.append(f"R²: {overall.get('r2', float('nan')):.3f}")
        lines.append(f"Mean Bias: {overall.get('mean_bias', float('nan')):+.3f}")
        lines.append("")
        if results.by_position:
            lines.append("By Position:")
            for pos, m in results.by_position.items():
                lines.append(f"- {pos}: RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, Bias={m['mean_bias']:+.3f}, N={m['n_predictions']:,}")
            lines.append("")
        if results.drift_alerts:
            lines.append("Drift Alerts:")
            for a in results.drift_alerts[:5]:
                lines.append(f"- {a}")
            lines.append("")
        # Render text
        y = 0.95
        for line in lines:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=12, va='top')
            y -= 0.04
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    return pdf_path


def run_backtest_lite(actual_csv: Path, predictions_dir: Path, output_dir: Path) -> BacktestResults:
    logger = logging.getLogger(__name__)
    actuals = pd.read_csv(actual_csv)
    # Normalize actuals
    if "player_id" not in actuals.columns and "id" in actuals.columns:
        actuals["player_id"] = actuals["id"]
    if "gameweek" not in actuals.columns:
        for c in ["gw", "round", "event"]:
            if c in actuals.columns:
                actuals["gameweek"] = actuals[c]
                break
    if "actual_points" not in actuals.columns:
        for c in ["total_points", "points", "score"]:
            if c in actuals.columns:
                actuals["actual_points"] = actuals[c]
                break
    # Determine range
    gw_range = _auto_detect_gw_range(actuals, predictions_dir)
    if not gw_range:
        raise RuntimeError(f"No prediction files found in {predictions_dir}")
    preds = _load_predictions(predictions_dir, gw_range)
    if preds.empty:
        raise RuntimeError(f"No predictions matched range {gw_range}")
    merged = pd.merge(preds, actuals, on=["player_id", "gameweek"], how="inner")
    if merged.empty:
        raise RuntimeError("No rows after merging predictions and actuals. Check IDs and gameweeks.")

    overall = _compute_metrics(merged)
    by_position: Dict[str, Dict[str, float]] = {}
    if "position" in merged.columns:
        for pos, g in merged.groupby("position"):
            by_position[str(pos)] = _compute_metrics(g)
    by_gameweek: Dict[int, Dict[str, float]] = {}
    for gw, g in merged.groupby("gameweek"):
        m = _compute_metrics(g)
        try:
            r_true = pd.Series(g["actual_points"].astype(float)).rank(method="average").to_numpy()
            r_pred = pd.Series(g["predicted_points"].astype(float)).rank(method="average").to_numpy()
            rx = r_true - np.mean(r_true)
            ry = r_pred - np.mean(r_pred)
            denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
            corr = float(np.sum(rx * ry) / denom) if denom > 0 else float("nan")
            m["xi_score"] = corr if corr == corr else None
        except Exception:
            m["xi_score"] = None
        by_gameweek[int(gw)] = m

    drift_alerts: List[str] = []
    for gw, m in sorted(by_gameweek.items()):
        mb = m.get("mean_bias")
        if mb is not None and abs(mb) > 0.5:
            drift_alerts.append(f"GW{gw}: mean bias {mb:+.2f} pts")

    # Write markdown summary
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"backtest_summary_lite_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Fantasy Football ML Pipeline Report (Lite)\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"Predictions merged: {overall['n_predictions']:,}\n\n")
        f.write("## Overall Performance\n\n")
        f.write(f"- RMSE: {overall['rmse']:.3f}\n")
        f.write(f"- MAE: {overall['mae']:.3f}\n")
        f.write(f"- R²: {overall['r2']:.3f}\n")
        f.write(f"- Mean Bias: {overall['mean_bias']:+.3f}\n\n")
        if by_position:
            f.write("## Performance by Position\n")
            for pos, m in by_position.items():
                f.write(f"- {pos}: RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, Bias={m['mean_bias']:+.3f}, N={m['n_predictions']:,}\n")
            f.write("\n")
        if drift_alerts:
            f.write("## Drift Alerts\n")
            for a in drift_alerts[:5]:
                f.write(f"- {a}\n")
            f.write("\n")
        f.write("## Gameweek Performance (sample)\n")
        for gw in sorted(list(by_gameweek.keys()))[:10]:
            m = by_gameweek[gw]
            xi = m.get("xi_score")
            xi_str = f"{xi:.3f}" if xi is not None else "N/A"
            f.write(f"- GW{gw}: RMSE={m['rmse']:.3f}, Bias={m['mean_bias']:+.3f}, N={m['n_predictions']}, XI={xi_str}\n")

    pdf_path = _render_pdf(BacktestResults(overall=overall, by_position=by_position, by_gameweek=by_gameweek, drift_alerts=drift_alerts), output_dir)
    logging.getLogger(__name__).info("Backtest lite summary: %s", md_path)
    if pdf_path:
        logging.getLogger(__name__).info("Backtest lite PDF: %s", pdf_path)
    return BacktestResults(
        overall=overall,
        by_position=by_position,
        by_gameweek=by_gameweek,
        drift_alerts=drift_alerts,
        md_path=md_path,
        pdf_path=pdf_path,
    )
