#!/usr/bin/env python3
"""
Utility script to run historical backtest analysis on FPL predictions.

Usage:
    python scripts/run_backtest_analysis.py --actual-data path/to/actual_results.csv
    python scripts/run_backtest_analysis.py --actual-data path/to/actual_results.csv --gameweek-range 1 10
    python scripts/run_backtest_analysis.py --actual-data path/to/actual_results.csv --generate-report
    python scripts/run_backtest_analysis.py --actual-data path/to/actual_results.csv --output-dir desktop
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
log_file_path = Path(__file__).parent.parent / 'logs' / 'backtest_analysis.log'
log_file_path.parent.mkdir(exist_ok=True)

# Set up root logger with WARNING level to reduce noise
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

# Set specific loggers to INFO level
logging.getLogger('models.ml_predictor').setLevel(logging.INFO)  # Increase ML predictor verbosity for debugging
logging.getLogger('run_backtest_analysis').setLevel(logging.INFO)   # Keep script's own logs at INFO

# Note: We avoid importing heavy ML dependencies (e.g., xgboost/lightgbm) here
# to keep packaging light. The local backtest implementation relies only on
# pandas/numpy/sklearn/scipy which are commonly available.

def _safe_float(s):
    try:
        return float(s)
    except Exception:
        return 0.0

def _position_from_element_type(v):
    try:
        et = int(v)
    except Exception:
        return None
    return {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(et)

def _load_predictions(predictions_dir: Path, gameweek_range: tuple[int, int]) -> pd.DataFrame:
    """Load predictions CSVs for the specified gameweek range and standardize columns."""
    files = sorted(predictions_dir.glob('predictions_gw*.csv'))
    if not files:
        return pd.DataFrame()
    gw_start, gw_end = gameweek_range
    frames: list[pd.DataFrame] = []
    for f in files:
        # Extract GW from filename: predictions_gw<NN>_timestamp.csv
        gw = None
        stem = f.stem
        try:
            parts = stem.split('_')
            gw = int(parts[1].replace('gw', '').lstrip('0') or '0')
        except Exception:
            gw = None
        if gw is None or gw < gw_start or gw > gw_end:
            continue
        try:
            df = pd.read_csv(f)
            df['__source_file'] = str(f)
            # Standardize identifiers
            if 'id' not in df.columns and 'player_id' in df.columns:
                df['id'] = df['player_id']
            if 'player_id' not in df.columns and 'id' in df.columns:
                df['player_id'] = df['id']
            # Standardize gameweek column
            if 'gameweek' not in df.columns:
                df['gameweek'] = gw
            # Position fallback from element_type if needed
            if 'position' not in df.columns and 'element_type' in df.columns:
                df['position'] = df['element_type'].apply(_position_from_element_type)
            # Require minimal columns
            cols_needed = {'player_id', 'gameweek', 'predicted_points'}
            if not cols_needed.issubset(set(df.columns)):
                # Try to derive predicted_points if available as another column
                cand_cols = [c for c in df.columns if 'pred' in c.lower() and 'point' in c.lower()]
                if cand_cols:
                    df['predicted_points'] = df[cand_cols[0]]
            if cols_needed.issubset(set(df.columns)):
                frames.append(df[list(set(df.columns))])
        except Exception as e:
            logging.getLogger('run_backtest_analysis').warning(f"Failed to load predictions from {f}: {e}")
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True)
    return pred

def _compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df['actual_points'].astype(float).to_numpy()
    y_pred = df['predicted_points'].astype(float).to_numpy()
    n = len(y_true)
    if n == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'mean_bias': float('nan'), 'n_predictions': 0}
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    # R^2: 1 - SS_res/SS_tot
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float('nan') if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    mean_bias = float(np.mean(diff))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mean_bias': mean_bias, 'n_predictions': int(n)}

def run_historical_backtest_local(actual_results: pd.DataFrame, gameweek_range: tuple[int, int], predictions_dir: str | Path) -> dict | None:
    """Local fallback backtest when predictor lacks run_historical_backtest()."""
    logger = logging.getLogger('run_backtest_analysis')
    pred_dir = Path(predictions_dir)
    preds = _load_predictions(pred_dir, gameweek_range)
    if preds.empty:
        logger.error("No prediction files found in %s for range %s", pred_dir, gameweek_range)
        return None
    # Standardize actuals
    ar = actual_results.copy()
    # Normalize id columns
    if 'player_id' not in ar.columns and 'id' in ar.columns:
        ar['player_id'] = ar['id']
    if 'gameweek' not in ar.columns:
        # Attempt fallback column names
        for c in ['gw', 'round', 'event']:
            if c in ar.columns:
                ar['gameweek'] = ar[c]
                break
    if 'actual_points' not in ar.columns:
        # Try common alternatives
        for c in ['total_points', 'points', 'score']:
            if c in ar.columns:
                ar['actual_points'] = ar[c]
                break
    # Filter to requested range
    gw_start, gw_end = gameweek_range
    ar = ar[(pd.to_numeric(ar['gameweek'], errors='coerce') >= gw_start) & (pd.to_numeric(ar['gameweek'], errors='coerce') <= gw_end)]
    # Merge on (player_id, gameweek)
    key_cols = ['player_id', 'gameweek']
    merged = pd.merge(preds, ar, on=key_cols, how='inner')
    if merged.empty:
        logger.error("No rows after merging predictions with actuals. Check IDs and gameweeks.")
        return None
    # Compute overall metrics
    overall = _compute_metrics(merged)
    # By position
    by_position: dict[str, dict] = {}
    if 'position' in merged.columns:
        for pos, group in merged.groupby('position'):
            by_position[str(pos)] = _compute_metrics(group)
    # By gameweek
    by_gameweek: dict[int, dict] = {}
    for gw, g in merged.groupby('gameweek'):
        m = _compute_metrics(g)
        # simple xi proxy: Spearman correlation of ranks (numpy/pandas only)
        try:
            r_true = pd.Series(g['actual_points'].astype(float)).rank(method='average').to_numpy()
            r_pred = pd.Series(g['predicted_points'].astype(float)).rank(method='average').to_numpy()
            rx = r_true - np.mean(r_true)
            ry = r_pred - np.mean(r_pred)
            denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
            corr = float(np.sum(rx * ry) / denom) if denom > 0 else float('nan')
            m['xi_score'] = corr if corr == corr else None
        except Exception:
            m['xi_score'] = None
        by_gameweek[int(gw)] = m
    # Drift alerts: mean bias > 0.5 magnitude
    drift_alerts: list[str] = []
    for gw, m in sorted(by_gameweek.items()):
        mb = m.get('mean_bias')
        if mb is not None and abs(mb) > 0.5:
            drift_alerts.append(f"GW{gw}: mean bias {mb:+.2f} pts")
    # xi mean
    xi_vals = [m.get('xi_score') for m in by_gameweek.values() if m.get('xi_score') is not None]
    if xi_vals:
        overall['xi_score_mean'] = float(np.mean(xi_vals))
    return {
        'overall': overall,
        'by_position': by_position,
        'by_gameweek': by_gameweek,
        'drift_alerts': drift_alerts,
    }


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_actual_results(file_path, season=None, filter_nonplaying: bool = False):
    """Load actual FPL results from a CSV file.

    If filter_nonplaying is True, exclude rows for non-playing players using
    minutes > 0 when available, otherwise actual_points > 0 as a proxy.
    """
    logger = logging.getLogger('run_backtest_analysis')
    try:
        df = pd.read_csv(file_path)

        # Optional filtering to exclude non-playing rows
        if filter_nonplaying:
            if 'minutes' in df.columns:
                initial_rows = len(df)
                df = df[df['minutes'] > 0]
                logger.info(f"Filtered out players who did not play. Kept {len(df)}/{initial_rows} records with minutes > 0.")
            elif 'actual_points' in df.columns:
                initial_rows = len(df)
                df = df[df['actual_points'] > 0]
                logger.info(f"Filtered out players with 0 actual points. Kept {len(df)}/{initial_rows} records.")
        else:
            logger.info("Including all actuals (no non-playing filter).")

        if 'season' in df.columns and season:
            original_rows = len(df)
            df = df[df['season'] == season]
            logger.info(f"Season filter applied: {season} -> {len(df)}/{original_rows} records")

        logger.info(f"Loaded {len(df)} actual result records for analysis.")
        return df
    except FileNotFoundError:
        logger.error(f"Actual results file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading actual results from {file_path}: {e}")
        return None

def assess_backtest_performance(backtest_results):
    """Assess if backtest results meet acceptable criteria"""
    overall = backtest_results['overall']
    
    # Define pass/fail criteria for FPL prediction models
    criteria = {
        'rmse_threshold': 3.0,      # RMSE should be < 3.0 points
        'mae_threshold': 2.0,       # MAE should be < 2.0 points  
        'r2_threshold': 0.15,       # R² should be > 0.15 (modest but realistic)
        'bias_threshold': 0.5,      # Mean bias should be < ±0.5 points
        'min_predictions': 100      # Need sufficient data points
    }
    
    # Check each criterion
    results = {
        'rmse_pass': overall['rmse'] < criteria['rmse_threshold'],
        'mae_pass': overall['mae'] < criteria['mae_threshold'],
        'r2_pass': overall['r2'] > criteria['r2_threshold'],
        'bias_pass': abs(overall['mean_bias']) < criteria['bias_threshold'],
        'sample_size_pass': overall['n_predictions'] >= criteria['min_predictions']
    }
    
    # Overall assessment
    critical_passes = results['rmse_pass'] and results['mae_pass'] and results['sample_size_pass']
    all_passes = all(results.values())
    
    if all_passes:
        status = "EXCELLENT"
        color = "green"
    elif critical_passes and (results['r2_pass'] or results['bias_pass']):
        status = "GOOD"
        color = "darkgreen"
    elif critical_passes:
        status = "ACCEPTABLE"
        color = "orange"
    else:
        status = "NEEDS_IMPROVEMENT"
        color = "red"
    
    return {
        'status': status,
        'color': color,
        'criteria': criteria,
        'results': results,
        'passes': sum(results.values()),
        'total': len(results)
    }

def generate_pdf_report(backtest_results, output_path):
    """Generate a comprehensive PDF report from backtest results"""
    try:
        assessment = assess_backtest_performance(backtest_results)
        overall = backtest_results['overall']
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Executive Summary
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('FPL Model Backtest Analysis - Executive Summary', fontsize=16, fontweight='bold')
            
            # Overall metrics summary
            ax1.axis('off')
            xi_text = overall.get('xi_score_mean')
            xi_text = f"{xi_text:.3f}" if xi_text is not None else "N/A"
            summary_text = f"""OVERALL PERFORMANCE METRICS
            
RMSE: {overall['rmse']:.3f} points
MAE: {overall['mae']:.3f} points
R²: {overall['r2']:.3f}
Mean Bias: {overall['mean_bias']:+.3f} points
Predictions: {overall['n_predictions']:,}
XI Score (mean): {xi_text}
            
VALIDATION STATUS: {assessment['status']}
Criteria Passed: {assessment['passes']}/{assessment['total']}"""
            ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            # Performance by position (if available)
            if backtest_results['by_position']:
                positions = list(backtest_results['by_position'].keys())
                rmse_values = [backtest_results['by_position'][pos]['rmse'] for pos in positions]
                
                bars = ax2.bar(positions, rmse_values, color='lightblue', edgecolor='navy')
                ax2.set_title('RMSE by Position', fontweight='bold')
                ax2.set_ylabel('RMSE (points)')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, val in zip(bars, rmse_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, 'Position data\nnot available', ha='center', va='center')
            
            # Performance by gameweek (if available)
            if backtest_results['by_gameweek']:
                gws = sorted(backtest_results['by_gameweek'].keys())
                bias_values = [backtest_results['by_gameweek'][gw]['mean_bias'] for gw in gws]
                
                ax3.plot(gws, bias_values, marker='o', linewidth=2, markersize=4)
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax3.set_title('Mean Bias by Gameweek', fontweight='bold')
                ax3.set_xlabel('Gameweek')
                ax3.set_ylabel('Mean Bias (points)')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.axis('off')
                ax3.text(0.5, 0.5, 'Gameweek data\nnot available', ha='center', va='center')
            
            # Pass/Fail criteria chart
            criteria_names = ['RMSE', 'MAE', 'R²', 'Bias', 'Sample']
            pass_status = [assessment['results'][key] for key in 
                          ['rmse_pass', 'mae_pass', 'r2_pass', 'bias_pass', 'sample_size_pass']]
            colors = ['green' if passed else 'red' for passed in pass_status]
            
            bars = ax4.bar(criteria_names, [1] * len(criteria_names), color=colors, alpha=0.7)
            ax4.set_title('Pass/Fail Criteria', fontweight='bold')
            ax4.set_ylabel('Status')
            ax4.set_ylim(0, 1.2)
            ax4.set_yticks([0, 1])
            ax4.set_yticklabels(['FAIL', 'PASS'])
            
            # Add checkmarks and X marks
            for i, (bar, passed) in enumerate(zip(bars, pass_status)):
                symbol = '✓' if passed else '✗'
                ax4.text(bar.get_x() + bar.get_width()/2, 0.5, symbol,
                        ha='center', va='center', fontsize=16, fontweight='bold', color='white')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Methodology and Detailed Results
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            methodology_text = f"""BACKTEST METHODOLOGY AND DETAILED ANALYSIS
            
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
METHODOLOGY:
This backtest compares historical model predictions against {"mock" if "mock" in str(output_path) else "actual"} FPL results.
Predictions were loaded from archived CSV files and merged with actual results by player_id and gameweek.

KEY METRICS EXPLAINED:
• RMSE (Root Mean Square Error): Measures average prediction error magnitude
• MAE (Mean Absolute Error): Average absolute difference between predicted and actual points  
• R² (Coefficient of Determination): Proportion of variance explained by the model
• Mean Bias: Average prediction bias (positive = over-prediction, negative = under-prediction)
• XI Score: Normalized 0..1 score comparing predicted top-11 vs random baseline (0) and best possible XI (1)
            
VALIDATION CRITERIA:
• RMSE < 3.0 points (CRITICAL)
• MAE < 2.0 points (CRITICAL) 
• R² > 0.15 (IMPORTANT)
• |Mean Bias| < 0.5 points (IMPORTANT)
• Sample Size ≥ 100 predictions (CRITICAL)

DETAILED RESULTS:"""
            
            # Add position breakdown if available
            if backtest_results['by_position']:
                methodology_text += "\n\nPERFORMANCE BY POSITION:"
                for pos, metrics in backtest_results['by_position'].items():
                    methodology_text += f"\n• {pos}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
                    methodology_text += f"Bias={metrics['mean_bias']:+.3f}, N={metrics['n_predictions']:,}"
            
            # Add drift analysis
            if backtest_results['drift_alerts']:
                methodology_text += f"\n\nDRIFT ALERTS ({len(backtest_results['drift_alerts'])} detected):"
                for alert in backtest_results['drift_alerts'][:5]:  # Show top 5
                    methodology_text += f"\n• {alert}"
                if len(backtest_results['drift_alerts']) > 5:
                    methodology_text += f"\n• ... and {len(backtest_results['drift_alerts'])-5} more"
            else:
                methodology_text += "\n\nDRIFT ANALYSIS:\nNo significant prediction drift detected across gameweeks."
            
            # Add recommendation
            methodology_text += f"\n\nRECOMMENDATION:\n"
            if assessment['status'] == 'EXCELLENT':
                methodology_text += "Model performance is EXCELLENT. All criteria passed with strong predictive power."
            elif assessment['status'] == 'GOOD':
                methodology_text += "Model performance is GOOD. Critical criteria passed with adequate predictive performance."
            elif assessment['status'] == 'ACCEPTABLE': 
                methodology_text += "Model performance is ACCEPTABLE. Meets minimum requirements but has room for improvement."
            else:
                methodology_text += "Model performance NEEDS IMPROVEMENT. Consider retraining or feature engineering."
            
            ax.text(0.05, 0.95, methodology_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace', wrap=True)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"PDF report generated: {output_path}")
        return assessment
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None


def generate_markdown_report(backtest_results, output_path):
    """Generate a markdown report from backtest results"""
    try:
        with open(output_path, 'w') as f:
            f.write("# FPL Model Backtest Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            overall = backtest_results['overall']
            f.write("## Overall Performance\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| RMSE | {overall['rmse']:.3f} |\n")
            f.write(f"| MAE | {overall['mae']:.3f} |\n")
            f.write(f"| R² | {overall['r2']:.3f} |\n")
            f.write(f"| Mean Bias | {overall['mean_bias']:+.3f} pts |\n")
            f.write(f"| Predictions | {overall['n_predictions']:,} |\n\n")
            xi_mean = overall.get('xi_score_mean')
            if xi_mean is not None:
                f.write(f"| XI Score (mean) | {xi_mean:.3f} |\n\n")
            
            # Position breakdown
            if backtest_results['by_position']:
                f.write("## Performance by Position\n\n")
                f.write("| Position | RMSE | MAE | Mean Bias | Predictions |\n")
                f.write("|----------|------|-----|-----------|-------------|\n")
                for pos, metrics in backtest_results['by_position'].items():
                    f.write(f"| {pos} | {metrics['rmse']:.3f} | {metrics['mae']:.3f} | ")
                    f.write(f"{metrics['mean_bias']:+.3f} | {metrics['n_predictions']:,} |\n")
                f.write("\n")
            
            # Drift alerts
            if backtest_results['drift_alerts']:
                f.write("## Drift Alerts\n\n")
                f.write("Gameweeks with significant prediction bias (>0.5 pts average error):\n\n")
                for alert in backtest_results['drift_alerts']:
                    f.write(f"- {alert}\n")
                f.write("\n")
            else:
                f.write("## Drift Analysis\n\n")
                f.write("No significant prediction drift detected (all gameweeks within ±0.5 pts bias).\n\n")
            
            # Gameweek performance
            if backtest_results['by_gameweek']:
                f.write("## Gameweek Performance\n\n")
                f.write("| GW | RMSE | Mean Bias | Predictions | XI Score |\n")
                f.write("|----|------|-----------|-------------|----------|\n")
                for gw in sorted(backtest_results['by_gameweek'].keys()):
                    metrics = backtest_results['by_gameweek'][gw]
                    f.write(f"| {gw} | {metrics['rmse']:.3f} | ")
                    xi_val = metrics.get('xi_score')
                    xi_str = f"{xi_val:.3f}" if xi_val is not None else "N/A"
                    f.write(f"{metrics['mean_bias']:+.3f} | {metrics['n_predictions']:,} | {xi_str} |\n")
                f.write("\n")
            
        print(f"Markdown report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating markdown report: {e}")


def main():
    logger = logging.getLogger('run_backtest_analysis')
    logger.info("Starting backtest analysis script.")

    parser = argparse.ArgumentParser(description="Run FPL prediction backtest analysis")
    parser.add_argument('--actual-data', required=True,
                       help='Path to CSV file with actual FPL results (required)')
    parser.add_argument('--gameweek-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Gameweek range to analyze (e.g., --gameweek-range 1 10)')
    parser.add_argument('--generate-report', action='store_true', default=True,
                       help='Generate PDF backtest report (default: True)')
    parser.add_argument('--output-dir', default='reports',
                       help='Output directory for reports (default: reports)')
    parser.add_argument('--season', default='2024-25',
                       help="Season to analyze (e.g., '2024-25'). Defaults to 2024-25. If the actuals file contains multiple seasons, it will be filtered to this one.")
    parser.add_argument('--predictions-dir', default='data/backtest',
                       help='Directory containing prediction files (default: data/backtest)')
    parser.add_argument('--filter-nonplaying', action='store_true', default=False,
                        help='If set, exclude non-playing rows from actuals (minutes==0 or actual_points==0). Default: include all actuals')
    
    args = parser.parse_args()
    logger.info(f"Arguments parsed: {args}")
    
    # Load configuration and setup predictor
    logger.info("Loading configuration...")
    config = load_config()
    logger.info("Initializing MLPredictor...")
    predictor = MLPredictor(config)
    
    # Load actual results (required)
    actual_results = None
    if args.actual_data and Path(args.actual_data).exists():
        logger.info(f"Loading actual results from: {args.actual_data}")
        actual_results = load_actual_results(args.actual_data, season=args.season, filter_nonplaying=args.filter_nonplaying)
    else:
        logger.error(f"Error: Actual results file not found: {args.actual_data}")
        return 1

    if actual_results is None or actual_results.empty:
        logger.error("Actual results are empty or failed to load. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(actual_results)} actual result records")
    
    # Run backtest analysis
    logger.info("Determining gameweek range for analysis...")
    gameweek_range = None
    if args.gameweek_range:
        gameweek_range = tuple(args.gameweek_range)
        logger.info(f"Using user-defined gameweek range: {gameweek_range}")
    else:
        logger.info("Auto-detecting gameweek range...")
        backtest_dir = Path(args.predictions_dir)
        if backtest_dir.exists():
            files = list(backtest_dir.glob('predictions_gw*.csv'))
            gws = []
            for f in files:
                try:
                    parts = f.stem.split('_')
                    gw = int(parts[1].replace('gw', '').lstrip('0') or '0')
                    gws.append(gw)
                except Exception:
                    continue
            
            if gws:
                gws = sorted(set(gws))
                logger.info(f"Found prediction files for gameweeks: {gws}")
                if actual_results is not None and 'gameweek' in actual_results.columns:
                    actual_gws = sorted(set(int(x) for x in actual_results['gameweek'].unique()))
                    logger.info(f"Found actuals for gameweeks: {actual_gws}")
                    common = [gw for gw in gws if gw in actual_gws]
                    if common:
                        gameweek_range = (min(common), max(common))
                        logger.info(f"Common gameweeks found. Setting range to: {gameweek_range}")
                    else:
                        logger.warning("No common gameweeks between predictions and actuals.")
                if gameweek_range is None:
                    gameweek_range = (min(gws), max(gws))
                    logger.info(f"Using prediction file gameweek range: {gameweek_range}")
            else:
                logger.warning("No prediction files found to auto-detect gameweek range.")
        else:
            logger.warning("Backtest directory 'data/backtest' not found.")

    if gameweek_range:
        logger.info(f"Final analysis range: gameweeks {gameweek_range[0]} to {gameweek_range[1]}")
    else:
        logger.error("Could not determine gameweek range. Exiting.")
        return 1
    
    logger.info("Running historical backtest...")
    if hasattr(predictor, 'run_historical_backtest') and callable(getattr(predictor, 'run_historical_backtest')):
        backtest_results = predictor.run_historical_backtest(actual_results, gameweek_range, predictions_dir=args.predictions_dir)
    else:
        logger.info("Predictor has no run_historical_backtest; using local backtest implementation.")
        backtest_results = run_historical_backtest_local(actual_results, gameweek_range, args.predictions_dir)
    
    if backtest_results is None:
        logger.error("Backtest analysis failed and returned None. Exiting.")
        return 1
    
    logger.info("Backtest analysis completed successfully.")
    
    # Print summary to console
    overall = backtest_results['overall']
    print("\n" + "="*60)
    print("BACKTEST ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall RMSE: {overall['rmse']:.3f}")
    print(f"Overall MAE: {overall['mae']:.3f}")
    print(f"Overall R²: {overall['r2']:.3f}")
    print(f"Mean Bias: {overall['mean_bias']:+.3f} points")
    print(f"Total Predictions: {overall['n_predictions']:,}")
    if overall.get('xi_score_mean') is not None:
        print(f"XI Score (mean): {overall['xi_score_mean']:.3f}")
    
    if backtest_results['drift_alerts']:
        print(f"\nDrift Alerts: {len(backtest_results['drift_alerts'])}")
        for alert in backtest_results['drift_alerts'][:3]:  # Show first 3
            print(f"  - {alert}")
        if len(backtest_results['drift_alerts']) > 3:
            print(f"  ... and {len(backtest_results['drift_alerts'])-3} more")
    else:
        print("\nNo significant drift detected")
    
    # Position breakdown
    if backtest_results['by_position']:
        print("\nPosition Performance:")
        for pos, metrics in backtest_results['by_position'].items():
            print(f"  {pos}: RMSE={metrics['rmse']:.3f}, Bias={metrics['mean_bias']:+.3f}")
    
    print("="*60)
    
    # Generate comprehensive PDF report
    if args.generate_report:
        logger.info("Generating reports...")
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # PDF Report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        season_tag = (args.season.replace('/', '-') if args.season else "")
        pdf_path = output_dir / f"backtest_analysis_{season_tag}_{timestamp}.pdf"
        assessment = generate_pdf_report(backtest_results, pdf_path)
        
        # Also generate markdown for compatibility
        md_path = output_dir / f"backtest_summary_{season_tag}_{timestamp}.md"
        generate_markdown_report(backtest_results, md_path)
        
        if assessment:
            print(f"\n{'='*60}")
            print(f"BACKTEST VALIDATION RESULT: {assessment['status']}")
            print(f"Criteria Passed: {assessment['passes']}/{assessment['total']}")
            print(f"PDF Report: {pdf_path}")
            print(f"{'='*60}")
    
    logger.info("Script finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred in main execution:")
        print(f"Exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        exit(1)
