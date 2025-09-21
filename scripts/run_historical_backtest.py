#!/usr/bin/env python3
"""
Historical Backtesting CLI Script

Runs comprehensive historical backtests using SQLite database with real FPL data.
Supports data import, benchmarks, progress tracking, and comprehensive reporting.

Usage Examples:
    # Import data and run full backtest
    python run_historical_backtest.py --import-data --seasons 2023-24 --model xgb
    
    # Run backtest with specific gameweek range
    python run_historical_backtest.py --seasons 2022-23 2023-24 --gw-range 10 20
    
    # Quick benchmark comparison
    python run_historical_backtest.py --benchmark-only --seasons 2023-24
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import json
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.historical_backtest import HistoricalBacktester
from src.data.historical_db import FPLHistoricalDB
from src.data.data_importer import FPLDataImporter

# Setup logging (default to warnings and above)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'historical_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_argument_parser():
    """Setup command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Historical FPL Backtesting with SQLite Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --import-data --seasons 2023-24
  %(prog)s --seasons 2022-23 2023-24 --model xgb --position ALL
  %(prog)s --benchmark-only --seasons 2023-24 --gw-range 1 10
  %(prog)s --status
        """
    )
    
    # Data management
    data_group = parser.add_argument_group('Data Management')
    data_group.add_argument(
        '--import-data',
        action='store_true',
        help='Import historical data from Vaastav\'s repository'
    )
    data_group.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh of existing data during import'
    )
    data_group.add_argument(
        '--status',
        action='store_true',
        help='Show database status and available data'
    )
    
    # Backtest configuration
    backtest_group = parser.add_argument_group('Backtest Configuration')
    backtest_group.add_argument(
        '--seasons',
        nargs='+',
        default=None,
        help='Seasons to backtest (e.g., 2021-22 2022-23). Default: all available'
    )
    backtest_group.add_argument(
        '--gw-range',
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        default=(1, 38),
        help='Gameweek range to test (default: 1 38)'
    )
    backtest_group.add_argument(
        '--model',
        choices=['xgb', 'lgbm', 'dt'],
        default='xgb',
        help='Primary model type (default: xgb)'
    )
    backtest_group.add_argument(
        '--position',
        choices=['ALL', 'GK', 'DEF', 'MID', 'FWD'],
        default='ALL',
        help='Position to focus on (default: ALL)'
    )
    backtest_group.add_argument(
        '--live-mode',
        action='store_true',
        default=True,
        help='Simulate live FPL conditions (default: enabled)'
    )
    backtest_group.add_argument(
        '--no-live-mode',
        action='store_false',
        dest='live_mode',
        help='Disable live simulation (use all available data)'
    )
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Run benchmark comparisons only (faster)'
    )
    exec_group.add_argument(
        '--retrain',
        action='store_true',
        help='Force retraining of existing models'
    )
    backtest_group.add_argument(
        '--recalibration-window',
        type=int,
        default=None,
        help='Use only the previous N GWs for training (“rolling window”).'
    )
    backtest_group.add_argument(
        '--retrain-frequency',
        type=int,
        help='Retrain every M GWs instead of every GW (reduces compute for live scenarios).')
    exec_group.add_argument(
        '--checkpoint-freq',
        type=int,
        default=5,
        help='Checkpoint frequency in gameweeks (default: 5)'
    )
    exec_group.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Custom database file path'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for reports (default: reports/)'
    )
    output_group.add_argument(
        '--generate-pdf',
        action='store_true',
        help='Generate PDF report in addition to JSON'
    )
    output_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose logging output'
    )
    
    return parser

def show_database_status(db_path: str = None):
    """Show current database status"""
    
    print("\n=== FPL Historical Database Status ===\n")
    
    try:
        with FPLHistoricalDB(db_path) as db:
            importer = FPLDataImporter(db)
            status = importer.get_import_status()
            
            if not status:
                print("No data found in database")
                print("   Run with --import-data to populate database")
                return
            
            print("Available Data:")
            print("-" * 50)
            
            for season, info in status.items():
                status_emoji = "[+]" if info['imported'] else "[-]"
                latest_gw = info['latest_gw'] or 0
                player_count = info['player_count']
                
                print(f"{status_emoji} {season:<10} | GW {latest_gw:2d} | {player_count:4d} players")
            
            # Overall stats
            total_gws = sum(info['latest_gw'] or 0 for info in status.values())
            total_players = sum(info['player_count'] for info in status.values())
            imported_seasons = sum(1 for info in status.values() if info['imported'])
            
            print("-" * 50)
            print(f"Total: {imported_seasons} seasons, {total_gws} gameweeks, {total_players} player-season records")
            
            # Check for recent backtests
            recent_runs = db.conn.execute("""
                SELECT run_id, timestamp, season_start, season_end, model_type 
                FROM backtest_runs 
                ORDER BY timestamp DESC 
                LIMIT 5
            """).fetchall()
            
            if recent_runs:
                print("\nRecent Backtest Runs:")
                print("-" * 50)
                for run in recent_runs:
                    run_id, timestamp, s_start, s_end, model = run
                    dt = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
                    print(f"   {run_id} | {dt} | {s_start}-{s_end} | {model}")
            
    except Exception as e:
        print(f"Error accessing database: {e}")
        print("   Database may not be initialized.")

def import_historical_data(seasons: List[str] = None, force_refresh: bool = False):
    """Import historical data from Vaastav's repository"""
    
    print("\n=== Importing FPL Historical Data ===\n")
    
    try:
        with FPLHistoricalDB() as db:
            importer = FPLDataImporter(db)
            
            if seasons is None:
                seasons = FPLDataImporter.SEASONS
                print(f"Importing all available seasons: {', '.join(seasons)}")
            else:
                print(f"Importing specified seasons: {', '.join(seasons)}")
            
            # Show what will be imported
            for season in seasons:
                existing_gw = db.get_latest_gw(season)
                if existing_gw and not force_refresh:
                    print(f"   {season}: Already exists (GW {existing_gw}) - will update")
                else:
                    status = "force refresh" if force_refresh else "new import"
                    print(f"   {season}: {status}")
            
            print()
            
            # Run import
            importer.import_all_seasons(seasons, force_refresh=force_refresh)
            
            print("\nData import completed!")
            show_database_status()
            
    except Exception as e:
        logger.error(f"Import failed: {e}")
        print(f"Import failed: {e}")
        sys.exit(1)


def run_historical_backtest(
    seasons: List[str] = None,
    gw_range: tuple = (1, 38),
    model_type: str = 'xgb',
    position: str = 'ALL',
    live_mode: bool = True,
    benchmark_only: bool = False,
    recalibration_window: int = None,
    checkpoint_freq: int = 5,
    retrain_frequency: int = None,
    db_path: str = None,
    output_dir: str = None,
    generate_pdf: bool = False,
    retrain: bool = False
):
    """Run historical backtest"""
    
    print("\n=== Running Historical Backtest ===\n")
    
    try:
        # Initialize backtester and ensure data is up-to-date
        backtester = HistoricalBacktester(db_path=db_path)
        print("\nChecking for latest data...")
        FPLDataImporter(backtester.db).auto_update_latest_data()

        backtester.checkpoint_frequency = checkpoint_freq
        
        # Show configuration
        print("Backtest Configuration:")
        print(f"   Seasons: {seasons or 'all available'}")
        print(f"   Gameweeks: {gw_range[0]}-{gw_range[1]}")
        print(f"   Model: {model_type}")
        print(f"   Position: {position}")
        print(f"   Live Mode: {'enabled' if live_mode else 'disabled'}")
        print(f"   Recalibration Window: {recalibration_window}")
        print(f"   Retrain Frequency: {retrain_frequency}")
        print(f"   Benchmark Only: {'yes' if benchmark_only else 'no'}")
        print()
        
        # Check data availability
        available_seasons = backtester.db.get_available_seasons()
        if not available_seasons:
            print("No data available in database")
            print("   Run with --import-data first")
            sys.exit(1)
        
        if seasons:
            missing_seasons = set(seasons) - set(available_seasons)
            if missing_seasons:
                print(f"Missing seasons: {', '.join(missing_seasons)}")
                print(f"   Available: {', '.join(available_seasons)}")
                sys.exit(2)
        
        print("Data validation passed")
        print()
        
        # Run backtest
        results = backtester.run_historical_backtest(
            seasons=seasons,
            gw_range=gw_range,
            model_type=model_type,
            position=position,
            recalibration_window = recalibration_window,
            retrain_frequency = retrain_frequency,
            use_live_mode=live_mode,
            retrain=retrain
        )
        
        if not results:
            print("Backtest failed - no results generated")
            sys.exit(3)
        
        # Display summary
        print("\n=== Backtest Results Summary ===\n")
        
        overall_stats = results.get('results', {}).get('overall', {})
        if isinstance(overall_stats, dict) and overall_stats:
            print(f"Overall Performance:")
            print(f"   RMSE: {overall_stats.get('overall_rmse', 0):.3f}")
            print(f"   MAE:  {overall_stats.get('overall_mae', 0):.3f}")
            print(f"   R²:   {overall_stats.get('overall_r2', 0):.3f}")
            print(f"   Bias: {overall_stats.get('overall_bias', 0):.3f}")
            print(f"        # Print comprehensive results summary")
        if results:
            print("\n=== Comprehensive Backtest Results ===")
            
            # Model performance comparison (use benchmarks)
            benchmarks = results.get('results', {}).get('benchmarks', {})
            if isinstance(benchmarks, dict) and benchmarks:
                print(f"\nModel Performance Comparison:")
                print("-" * 70)
                print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Bias':<8} {'Entropy':<8} {'Status':<10}")
                print("-" * 70)
                
                model_stats = []
                model_display_names = {
                    'primary': 'XGBoost',
                    'decision_tree': 'DecisionTree',
                    'random_forest': 'RandomForest',
                    'perfect_xi': 'Perfect XI',
                    'random': 'Random'
                }
                
                for model_name in ['primary', 'decision_tree', 'random_forest', 'perfect_xi', 'random']:
                    stats = benchmarks.get(model_name)
                    if isinstance(stats, dict):
                        rmse = stats.get('avg_rmse', 0.0)
                        mae = stats.get('avg_mae', 0.0)
                        r2 = stats.get('avg_r2', 0.0)
                        bias = stats.get('avg_bias', 0.0)
                        entropy = stats.get('avg_entropy', None)
                        status = "Good" if rmse < 3.0 else "High Error"
                        display_name = model_display_names.get(model_name, model_name)
                        model_stats.append((model_name, display_name, rmse, mae, r2, bias, entropy, status))
                
                model_stats.sort(key=lambda x: x[2])  # Sort by RMSE
                for model_name, display_name, rmse, mae, r2, bias, entropy, status in model_stats:
                    entropy_disp = f"{entropy:.3f}" if isinstance(entropy, (int, float)) else "-"
                    print(f"{display_name:<15} {rmse:<8.3f} {mae:<8.3f} {r2:<8.3f} {bias:<8.3f} {entropy_disp:<8} {status:<10}")
                if model_stats:
                    best_model, best_display, best_rmse = model_stats[0][0], model_stats[0][1], model_stats[0][2]
                    print(f"\nBest performer: {best_display} (RMSE: {best_rmse:.3f})")
            
            # Gameweek-by-gameweek breakdown (first few GWs)
            season_results = results.get('results', {}).get('by_season', {})
            if season_results:
                print(f"\nSample Gameweek Performance:")
                print("-" * 50)
                
                gw_shown = 0
                for season, season_data in season_results.items():
                    print(f"\n{season}:")
                    gw_data = season_data.get('by_gameweek', {}) if isinstance(season_data, dict) else {}
                    
                    for gw_key in sorted(gw_data.keys())[:3]:  # Show first 3 GWs
                        gw_results = gw_data[gw_key]
                        # gw_key is season-qualified (e.g., '2023-24-06'); extract trailing GW number if present
                        try:
                            gw_display = int(str(gw_key).split('-')[-1])
                        except Exception:
                            gw_display = str(gw_key)
                        print(f"  GW{gw_display}: ", end="")
                        
                        # Show RMSE for each model in this GW
                        for model in ['primary', 'decision_tree', 'random_forest', 'perfect_xi']:
                            metrics = gw_results.get('metrics', {}) if isinstance(gw_results, dict) else {}
                            if model in metrics and isinstance(metrics[model], dict):
                                gw_rmse = metrics[model].get('rmse', 0.0)
                                print(f"{model[:4]}={gw_rmse:.2f} ", end="")
                        print()
                        
                        gw_shown += 1
                        if gw_shown >= 6:  # Limit total shown
                            break
                    
                    if gw_shown >= 6:
                        break
                
                # Total gameweek summary
                total_gws = 0
                for s in season_results.values():
                    if isinstance(s, dict):
                        total_gws += len(s.get('by_gameweek', {}))
                print(f"\nTotal gameweeks analyzed: {total_gws}")

            # XI Selection vs Best XI (aggregate)
            if isinstance(benchmarks, dict) and benchmarks:
                print("\nXI Selection vs Best XI:")
                print("-" * 70)
                print(f"{'Model XI':<15} {'Overlap%':<10} {'AbsGap':<10}")
                print("-" * 70)
                for model_name, display_name in [('primary_xi','XGBoost XI'), ('decision_tree_xi','DecisionTree XI'), ('random_forest_xi','RandomForest XI')]:
                    stats = benchmarks.get(model_name, {})
                    if not isinstance(stats, dict) or not stats:
                        continue
                    overlap = stats.get('avg_xi_overlap')
                    gap = stats.get('avg_xi_points_gap_abs')
                    overlap_disp = f"{(overlap*100):.1f}%" if isinstance(overlap, (int,float)) else '-'
                    gap_disp = f"{gap:.2f}" if isinstance(gap, (int,float)) else '-'
                    print(f"{display_name:<15} {overlap_disp:<10} {gap_disp:<10}")
            
            # Data quality summary
            print(f"\nData Quality Summary:")
            print("-" * 30)
            if isinstance(overall_stats, dict) and overall_stats:
                print(f"   Seasons processed: {len(season_results)}")
                # Compare benchmark RMSEs for identical values
                if isinstance(benchmarks, dict) and benchmarks:
                    rmse_values = [v.get('avg_rmse', None) for v in benchmarks.values() if isinstance(v, dict)]
                    rmse_values = [v for v in rmse_values if v is not None]
                    unique_rmses = len(set(f"{r:.3f}" for r in rmse_values))
                    if rmse_values and unique_rmses < len(rmse_values):
                        print("   Warning: Some models show identical RMSE values")
        
        print(f"\nBacktest completed successfully (Run ID: {results['run_id']})")
        
        # Generate additional reports if requested
        if generate_pdf:
            print("Generating PDF report...")
            desktop_path = Path.home() / "Desktop"
            desktop_path.mkdir(exist_ok=True)
            pdf_path = desktop_path / f"backtest_report_{results['run_id']}.pdf"
            generate_pdf_report(results, pdf_path)
            print(f"PDF report saved to {pdf_path}")

        backtester.close()
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"Backtest failed: {e}")
        sys.exit(1)

def generate_pdf_report(results: dict, output_path: Path):
    """Generates a PDF report from backtest results."""
    doc = SimpleDocTemplate(str(output_path))
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("FPL Historical Backtest Report", styles['h1']))
    story.append(Spacer(1, 0.2*inch))

    # Run ID and Timestamp
    run_id = results['run_id']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Run ID: {run_id}", styles['Normal']))
    story.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Overall Performance (with fallback if not available)
    results_root = results.get('results', {}) if isinstance(results, dict) else {}
    overall_stats = results_root.get('overall', {}) if isinstance(results_root, dict) else {}
    benchmarks = results_root.get('benchmarks', {}) if isinstance(results_root, dict) else {}

    if not overall_stats:
        # Fallback: synthesize a minimal "overall" from benchmarks if present
        # Prefer primary; otherwise pick decision_tree; else any available
        for key in ['primary', 'decision_tree', 'random_forest', 'random', 'perfect_xi']:
            if isinstance(benchmarks.get(key), dict):
                b = benchmarks[key]
                overall_stats = {
                    'overall_rmse': b.get('avg_rmse'),
                    'overall_mae': b.get('avg_mae'),
                    'overall_r2': b.get('avg_r2'),
                    'overall_bias': b.get('avg_bias'),
                }
                break

    if isinstance(overall_stats, dict) and overall_stats:
        story.append(Paragraph("Overall Performance", styles['h2']))
        data = [
            ['Metric', 'Value'],
            ['RMSE', f"{overall_stats.get('overall_rmse', 0) if overall_stats.get('overall_rmse', None) is not None else 0:.3f}"],
            ['MAE', f"{overall_stats.get('overall_mae', 0) if overall_stats.get('overall_mae', None) is not None else 0:.3f}"],
            ['R²', f"{overall_stats.get('overall_r2', 0) if overall_stats.get('overall_r2', None) is not None else 0:.3f}"],
            ['Bias', f"{overall_stats.get('overall_bias', 0) if overall_stats.get('overall_bias', None) is not None else 0:.3f}"]
        ]
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))

    # XI Selection vs Best XI (aggregate)
    if isinstance(benchmarks, dict) and benchmarks:
        story.append(Paragraph("XI Selection vs Best XI", styles['h2']))
        story.append(Paragraph("Overlap is the proportion of players the model-selected XI matched with the actual Best XI for the gameweek. Points Gap is Best XI total minus model XI total (positive means behind).", styles['Italic']))
        xi_header = ['Model XI', 'Overlap %', 'Avg Abs Points Gap']
        xi_data = [xi_header]
        for model_name, display_name in [('primary_xi','XGBoost XI'), ('decision_tree_xi','DecisionTree XI'), ('random_forest_xi','RandomForest XI')]:
            stats = benchmarks.get(model_name, {})
            if not isinstance(stats, dict) or not stats:
                continue
            overlap = stats.get('avg_xi_overlap')
            gap = stats.get('avg_xi_points_gap_abs')
            xi_data.append([
                display_name,
                (f"{overlap*100:.1f}%" if isinstance(overlap, (int,float)) else '-'),
                (f"{gap:.2f}" if isinstance(gap, (int,float)) else '-')
            ])
        if len(xi_data) > 1:
            t = Table(xi_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2*inch))

    # Model Performance Comparison
    benchmarks = results_root.get('benchmarks', {})
    if isinstance(benchmarks, dict) and benchmarks:
        story.append(Paragraph("Model Performance Comparison", styles['h2']))
        story.append(Paragraph("Entropy/Skill Score: normalized so Perfect XI = 1.0 (oracle), Random = 0.0; other models are linearly scaled between these baselines using RMSE.", styles['Italic']))
        model_display_names = {
            'primary': 'XGBoost',
            'decision_tree': 'DecisionTree',
            'random_forest': 'RandomForest',
            'perfect_xi': 'Perfect XI',
            'random': 'Random',
            'lastweek_xi': 'LastWeek XI'
        }
        header = ['Model', 'RMSE', 'MAE', 'R²', 'Bias', 'Entropy']
        data = [header]
        for model_name in ['primary', 'decision_tree', 'random_forest', 'perfect_xi', 'random', 'lastweek_xi']:
            stats = benchmarks.get(model_name)
            if isinstance(stats, dict):
                display_name = model_display_names.get(model_name, model_name)
                entropy = stats.get('avg_entropy')
                data.append([
                    display_name,
                    f"{(stats.get('avg_rmse') if stats.get('avg_rmse', None) is not None else 0.0):.3f}",
                    f"{(stats.get('avg_mae') if stats.get('avg_mae', None) is not None else 0.0):.3f}",
                    f"{(stats.get('avg_r2') if stats.get('avg_r2', None) is not None else 0.0):.3f}",
                    f"{(stats.get('avg_bias') if stats.get('avg_bias', None) is not None else 0.0):.3f}",
                    (f"{entropy:.3f}" if isinstance(entropy, (int, float)) else '-')
                ])
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))

    # Best XI (recent gameweeks)
    by_season = results_root.get('by_season', {}) if isinstance(results_root, dict) else {}
    if isinstance(by_season, dict) and by_season:
        story.append(Paragraph("Best XI (Recent Gameweeks)", styles['h2']))

        # Collect the latest up to 3 gameweeks across seasons
        all_gws = []  # list of tuples (season, gw_key)
        for season_name, season_data in by_season.items():
            gws = season_data.get('by_gameweek', {}) if isinstance(season_data, dict) else {}
            for gw_key in gws.keys():
                all_gws.append((season_name, gw_key))
        # Sort by gw_key lexicographically which works for format 'YYYY-YY-##'
        all_gws_sorted = sorted(all_gws, key=lambda x: x[1])
        gameweeks_to_show = all_gws_sorted[-3:]

        # Lazy DB instance for metadata lookups
        db_instance = None

        for season, gw_key in gameweeks_to_show:
            season_data = by_season.get(season, {})
            gws = season_data.get('by_gameweek', {})
            gw_result = gws.get(gw_key, {})
            predictions_all = gw_result.get('predictions', {})
            perfect_xi_preds = predictions_all.get('perfect_xi', [])
            primary_preds_list = predictions_all.get('primary', [])
            dt_preds_list = predictions_all.get('decision_tree', [])
            rf_preds_list = predictions_all.get('random_forest', [])

            if not perfect_xi_preds:
                continue

            # Create a lookup for primary model predictions by player_id (may be empty)
            def _as_int(v):
                try:
                    return int(v)
                except Exception:
                    return None
            primary_preds_map = { _as_int(p.get('player_id')): p.get('predicted_points') for p in (primary_preds_list or []) if _as_int(p.get('player_id')) is not None }
            dt_preds_map = { _as_int(p.get('player_id')): p.get('predicted_points') for p in (dt_preds_list or []) if _as_int(p.get('player_id')) is not None }
            rf_preds_map = { _as_int(p.get('player_id')): p.get('predicted_points') for p in (rf_preds_list or []) if _as_int(p.get('player_id')) is not None }

            table_data = [['Season', 'GW', 'Name', 'Position', 'Predicted', 'Actual']]
            for row in perfect_xi_preds:
                player_id = row.get('player_id')
                pid_i = _as_int(player_id)
                name = row.get('name', '')
                pos = row.get('position', '')
                # If name/pos missing, try to fetch from players table for this season
                if (not isinstance(name, str)) or (isinstance(name, str) and name.strip() == ''):
                    try:
                        if db_instance is None:
                            db_instance = FPLHistoricalDB()
                        meta = db_instance.conn.execute(
                            "SELECT name, position FROM players WHERE season = ? AND player_id = ?",
                            (season, int(player_id) if player_id is not None else -1)
                        ).fetchone()
                        if meta is None:
                            # Fallback any season (most recent)
                            meta_any = db_instance.conn.execute(
                                "SELECT name, position, season FROM players WHERE player_id = ? ORDER BY season DESC LIMIT 1",
                                (int(player_id) if player_id is not None else -1,)
                            ).fetchone()
                            meta = meta_any
                        if meta is not None:
                            name_db = meta[0]
                            pos_db = meta[1]
                            name = name or (name_db if isinstance(name_db, str) else '')
                            # Normalize numeric pos
                            try:
                                pos_i = int(pos_db)
                                pos = pos or {1:'GK',2:'DEF',3:'MID',4:'FWD'}.get(pos_i, '')
                            except Exception:
                                pos = pos or (pos_db if isinstance(pos_db, str) else '')
                    except Exception:
                        # ignore DB errors here to avoid breaking report
                        pass
                # Get prediction: primary -> decision_tree -> random_forest -> '-'
                pred_val = primary_preds_map.get(pid_i, None)
                if pred_val is None:
                    pred_val = dt_preds_map.get(pid_i, None)
                if pred_val is None:
                    pred_val = rf_preds_map.get(pid_i, None)
                pred_disp = f"{float(pred_val):.2f}" if isinstance(pred_val, (int, float)) else '-'
                actual = row.get('actual_points', 0.0)
                try:
                    gw_display = int(str(gw_key).split('-')[-1])
                except Exception:
                    gw_display = str(gw_key)
                table_data.append([season, str(gw_display), name, pos, pred_disp, f"{(actual if actual is not None else 0.0):.2f}"])

            if len(table_data) > 1:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 8),
                    ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
                ]))
                story.append(t)
                story.append(Spacer(1, 0.15*inch))

        # Close DB instance if opened
        try:
            if db_instance is not None:
                db_instance.close()
        except Exception:
            pass

    doc.build(story)

def main():
    """Main CLI entry point"""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure logs directory exists
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    print(f"\nFPL Historical Backtesting System")
    print(f"   Database: {args.db_path or 'default (data/fpl_history.db)'}")
    print(f"   Project: {project_root}")
    
    try:
        # Handle different modes
        if args.status:
            show_database_status(args.db_path)
            
        elif args.import_data:
            import_historical_data(args.seasons, args.force_refresh)
            
            
        else:
            # Run backtest (default mode)
            run_historical_backtest(
                seasons=args.seasons,
                gw_range=tuple(args.gw_range),
                model_type=args.model,
                position=args.position,
                live_mode=args.live_mode,
                recalibration_window = args.recalibration_window,
                benchmark_only=args.benchmark_only,
                retrain_frequency = args.retrain_frequency,
                checkpoint_freq=args.checkpoint_freq,
                db_path=args.db_path,
                output_dir=args.output_dir,
                generate_pdf=args.generate_pdf,
                retrain=args.retrain
            )
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
