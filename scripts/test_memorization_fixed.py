#!/usr/bin/env python3
"""
Test script for the enhanced memorization diagnostics with actual data format.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
import base64
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memorization_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the fixed module
from src.analysis.memorization_diagnostics_fixed import (
    calculate_bits_of_memorization,
    rolling_mean_predictor
)
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the prediction data for analysis."""
    logger.info(f"Loading data from {filepath}")
    
    # Load the data
    df = pd.read_csv(filepath, low_memory=False)
    
    # Basic preprocessing - use the actual column names from the CSV
    df = df.rename(columns={
        'id': 'player_id',
        'total_points': 'y_true',
        'element_type': 'position_code',
        'team': 'team_id',
        'now_cost': 'value',
        'selected_by_percent': 'selected',
        'minutes': 'minutes_played'
    })
    
    # Add gameweek if not present (assuming all data is for the same gameweek)
    if 'gw' not in df.columns:
        # Extract gameweek from filename if possible
        import re
        match = re.search(r'gw(\d+)', filepath.lower())
        if match:
            gw = int(match.group(1))
            df['gw'] = gw
        else:
            # Default to gameweek 1 if can't determine from filename
            df['gw'] = 1
    
    # Map position codes to names
    position_map = {
        1: 'GK',
        2: 'DEF',
        3: 'MID',
        4: 'FWD'
    }
    df['position'] = df['position_code'].map(position_map)
    
    # Ensure required columns exist
    required_cols = ['player_id', 'gw', 'y_true', 'position']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by player and gameweek
    df = df.sort_values(['player_id', 'gw']).reset_index(drop=True)
    
    # Add some derived features
    if 'minutes_played' in df.columns:
        df['starts'] = (df['minutes_played'] >= 60).astype(int)
        
    # Filter out players with too few minutes
    if 'minutes_played' in df.columns:
        min_minutes = df['gw'].nunique() * 30  # At least 30 mins per gameweek on average
        player_minutes = df.groupby('player_id')['minutes_played'].sum()
        active_players = player_minutes[player_minutes >= min_minutes].index
        df = df[df['player_id'].isin(active_players)]
    
    logger.info(f"Loaded data with {len(df)} rows and {df['player_id'].nunique()} players")
    return df

def test_with_actual_data(data_path: str, output_dir: str = 'reports/memorization_tests'):
    """Run memorization diagnostics on actual prediction data."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    
    # Define feature columns - use only columns that exist in the data
    possible_features = [
        'value', 'selected', 'minutes_played', 'influence', 
        'creativity', 'threat', 'ict_index', 'form',
        'points_per_game', 'value_form', 'value_season',
        'transfers_in', 'transfers_out', 'selected',
        'starts', 'goals_scored', 'assists', 'clean_sheets',
        'saves', 'bonus', 'bps', 'influence', 'creativity',
        'threat', 'ict_index', 'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'expected_goals_conceded'
    ]
    
    # Only keep columns that exist in the data and have some variation
    feature_cols = []
    for col in possible_features:
        if col in df.columns and col not in feature_cols:  # Avoid duplicates
            try:
                # Check if the column is numeric and has more than one unique non-null value
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Drop NA values for the uniqueness check
                    non_null = df[col].dropna()
                    if len(non_null) > 0 and non_null.nunique() > 1:
                        # Fill any remaining NaN values with column mean
                        df[col] = df[col].fillna(df[col].mean())
                        feature_cols.append(col)
            except Exception as e:
                logger.warning(f"Skipping column {col} due to error: {str(e)}", exc_info=True)
    
    # Ensure we have at least some features
    if not feature_cols:
        raise ValueError("No valid features found in the dataset")
    
    logger.info(f"Using {len(feature_cols)} features: {', '.join(feature_cols)}")
    
    # Since we only have one gameweek, we'll use all data for training
    # but we need to ensure we have enough data points
    min_samples = 50  # Minimum number of samples needed
    if len(df) < min_samples:
        logger.warning(f"Only {len(df)} samples available, which is less than the recommended minimum of {min_samples}")
    
    train_df = df.copy()
    logger.info(f"Using {len(train_df)} samples for training")
    
    # Prepare features and target
    X_train = train_df[feature_cols]
    y_train = train_df['y_true']
    
    logger.info(f"Training model on {len(train_df)} samples...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Define a simple baseline predictor that uses the global mean for prediction
    def global_mean_predictor(df):
        """Baseline predictor that uses global mean for all predictions."""
        global_mean = df['y_true'].mean()
        return np.full(len(df), global_mean)
    
    logger.info("Calculating memorization metrics...")
    # Call the function with debug information
    logger.info("Calling calculate_bits_of_memorization...")
    try:
        # Get the results
        per_sample_df, metrics = calculate_bits_of_memorization(
            model=model,
            train_df=train_df,
            baseline_predictor=global_mean_predictor,
            target_col='y_true',
            group_col='player_id',
            time_col='gw',
            position_col='position',
            feature_cols=feature_cols,
            return_df=True,
            min_std_dev=1.0,
            max_std_dev=8.0,
            min_samples_for_std=5,
            window_size=5,
            clip_residuals=8.0,
            variance_estimator="homoscedastic"
        )
        
        # Log the structure of the results
        if per_sample_df is not None:
            logger.info(f"Per-sample results shape: {per_sample_df.shape}")
            logger.info(f"Per-sample columns: {per_sample_df.columns.tolist()}")
        else:
            logger.warning("No per-sample results returned")
            
        if metrics:
            logger.info(f"Metrics keys: {list(metrics.keys())}")
            
        # If we have per-sample results, merge them back with the original data
        if per_sample_df is not None and not per_sample_df.empty:
            # Ensure we have the index to merge on
            if per_sample_df.index.names == [None]:  # If index is not set
                per_sample_df = per_sample_df.reset_index(drop=True)
                train_df = train_df.reset_index(drop=True)
                results = pd.concat([train_df, per_sample_df], axis=1)
            else:
                results = train_df.join(per_sample_df, how='left')
        else:
            logger.warning("No per-sample results to merge")
            results = train_df.copy()
            
    except Exception as e:
        logger.error(f"Error in calculate_bits_of_memorization: {str(e)}", exc_info=True)
        raise
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'memorization_results_{timestamp}.csv'
    metrics_path = output_dir / f'metrics_{timestamp}.json'
    
    results.to_csv(results_path, index=False)
    pd.Series(metrics).to_json(metrics_path, indent=2)
    
    logger.info(f"\n=== Memorization Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate and save HTML report
    report_path = output_dir / f'report_{timestamp}.html'
    generate_html_report(metrics, results, report_path)
    logger.info(f"HTML report generated at {report_path}")
    
    return results, metrics

def generate_html_report(metrics: Dict[str, Any], results: pd.DataFrame, output_path: Path) -> None:
    """Generate an HTML report for the memorization analysis."""
    # Create visualizations
    plots = {}
    
    # Check if required columns exist - note the actual column is 'bits_of_memorization'
    required_cols = ['bits_of_memorization', 'y_pred_model', 'y_true']
    missing_cols = [col for col in required_cols if col not in results.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns in results: {', '.join(missing_cols)}")
        logger.info(f"Available columns: {results.columns.tolist()}")
        return
    
    # 1. Distribution of bits
    plt.figure(figsize=(10, 6))
    plt.hist(results['bits_of_memorization'], bins=30, alpha=0.7, color='skyblue')
    plt.title('Distribution of Bits of Memorization')
    plt.xlabel('Bits of Memorization')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    bits_dist = BytesIO()
    plt.savefig(bits_dist, format='png', bbox_inches='tight')
    plt.close()
    plots['bits_dist'] = base64.b64encode(bits_dist.getvalue()).decode('utf-8')
    
    # 2. Bits by position - handle cases where position might be a list or multi-index
    if 'position' in results.columns and 'bits_of_memorization' in results.columns:
        try:
            # Ensure position is a simple column (not a MultiIndex or list-like)
            position_series = results['position'].copy()
            
            # If position is a list or array-like, take the first element
            if position_series.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                position_series = position_series.apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
            
            # Convert to string and get unique positions
            position_series = position_series.astype(str)
            
            # Only proceed if we have valid positions
            if len(position_series.unique()) > 1:  # Need at least 2 positions to be meaningful
                plt.figure(figsize=(10, 6))
                
                # Create a temporary DataFrame with the cleaned position data
                temp_df = pd.DataFrame({
                    'position': position_series,
                    'bits': results['bits_of_memorization']
                })
                
                # Group by position and calculate mean bits
                position_bits = temp_df.groupby('position')['bits'].mean().sort_values(ascending=False)
                
                # Only plot if we have data
                if not position_bits.empty:
                    position_bits.plot(kind='bar', color='lightgreen')
                    plt.title('Average Bits of Memorization by Position')
                    plt.ylabel('Average Bits of Memorization')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    pos_bits = BytesIO()
                    plt.savefig(pos_bits, format='png', bbox_inches='tight')
                    plt.close()
                    plots['pos_bits'] = base64.b64encode(pos_bits.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Could not generate position plot: {str(e)}")
    
    # 3. Top and bottom performers
    top_n = min(10, len(results) // 2)
    top_players = results.nlargest(top_n, 'bits_of_memorization')
    bottom_players = results.nsmallest(top_n, 'bits_of_memorization')
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memorization Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 5px; }}
            .metrics-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); 
                gap: 15px; 
                margin-bottom: 20px;
            }}
            .metric-card {{ 
                background: white; 
                padding: 15px; 
                border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                color: #2c3e50;
                margin: 10px 0;
            }}
            .plot {{ 
                text-align: center; 
                margin: 20px 0;
            }}
            .plot img {{ 
                max-width: 100%; 
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 15px 0;
            }}
            th, td {{ 
                padding: 12px; 
                text-align: left; 
                border-bottom: 1px solid #ddd;
            }}
            th {{ 
                background-color: #f2f2f2; 
                font-weight: bold;
            }}
            tr:hover {{background-color: #f5f5f5;}}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Memorization Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div>Mean Bits per Sample</div>
                        <div class="metric-value">{metrics.get('mean_bits_per_sample', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div>Median Bits per Sample</div>
                        <div class="metric-value">{metrics.get('median_bits_per_sample', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div>Total Bits</div>
                        <div class="metric-value">{metrics.get('total_bits', 0):.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div>% Positive Bits</div>
                        <div class="metric-value">{metrics.get('frac_positive', 0) * 100:.1f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Distribution of Bits</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{plots.get('bits_dist', '')}" alt="Bits Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance by Position</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{plots.get('pos_bits', '')}" alt="Bits by Position">
                </div>
                
                <h3>Position-wise Metrics</h3>
                <table>
                    <tr>
                        <th>Position</th>
                        <th>Mean Bits</th>
                        <th>% Positive</th>
                        <th>Sample Count</th>
                    </tr>
    """
    
    # Add position-wise metrics from the metrics dictionary
    position_metrics = []
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        n_samples = metrics.get(f'n_samples_{pos}', 0)
        if n_samples > 0:
            mean_bits = metrics.get(f'mean_positive_bits_{pos}', 0) * metrics.get(f'frac_positive_{pos}', 0) + \
                       metrics.get(f'mean_negative_bits_{pos}', 0) * metrics.get(f'frac_negative_{pos}', 0)
            
            position_metrics.append({
                'position': pos,
                'mean_bits': mean_bits,
                'frac_positive': metrics.get(f'frac_positive_{pos}', 0),
                'count': int(n_samples)
            })
    
    # Sort by mean bits in descending order
    position_metrics.sort(key=lambda x: x['mean_bits'], reverse=True)
    
    # Add position metrics to the HTML
    for metric in position_metrics:
        html_content += f"""
                <tr>
                    <td>{metric['position']}</td>
                    <td>{metric['mean_bits']:.4f}</td>
                    <td>{metric['frac_positive']*100:.1f}%</td>
                    <td>{metric['count']}</td>
                </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Top Performers</h2>
                <table>
                    <tr>
                        <th>Player ID</th>
                        <th>Position</th>
                        <th>Bits</th>
                        <th>Model Pred</th>
                        <th>Actual</th>
                    </tr>
    """
    
    # Add top performers
    for _, row in top_players.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row.get('player_id', '')}</td>
                        <td>{row.get('position', '')}</td>
                        <td>{row.get('bits_of_memorization', 0):.4f}</td>
                        <td>{row.get('y_pred_model', 0):.2f}</td>
                        <td>{row.get('y_true', 0)}</td>
                    </tr>
        """
    
    html_content += """
                </table>
                
                <h3>Bottom Performers</h3>
                <table>
                    <tr>
                        <th>Player ID</th>
                        <th>Position</th>
                        <th>Bits</th>
                        <th>Model Pred</th>
                        <th>Actual</th>
                    </tr>
    """
    
    # Add bottom performers
    for _, row in bottom_players.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row.get('player_id', '')}</td>
                        <td>{row.get('position', '')}</td>
                        <td>{row.get('bits', 0):.4f}</td>
                        <td>{row.get('y_pred_model', 0):.2f}</td>
                        <td>{row.get('y_true', 0)}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Model Details</h2>
                <p><strong>Variance Method:</strong> {metrics.get('variance_method', 'N/A')}</p>
                <p><strong>Model Std Dev:</strong> {metrics.get('model_std_mean', 0):.4f} (min: {metrics.get('model_std_min', 0):.4f}, max: {metrics.get('model_std_max', 0):.4f})</p>
                <p><strong>Baseline Std Dev:</strong> {metrics.get('baseline_std_mean', 0):.4f}</p>
                <p><strong>Number of Samples:</strong> {metrics.get('n_samples', 0)}</p>
                <p><strong>Number of Features:</strong> {len([col for col in results.columns if col.startswith('feat_')])}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run memorization diagnostics on prediction data')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to the prediction data file')
    parser.add_argument('--output-dir', type=str, default='reports/memorization_tests',
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    test_with_actual_data(
        data_path=args.data,
        output_dir=args.output_dir
    )
