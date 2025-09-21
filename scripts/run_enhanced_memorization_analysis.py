#!/usr/bin/env python3
"""
Enhanced memorization analysis script with improved diagnostics and reporting.

This script analyzes model memorization vs. generalization in time-series data,
with support for both squared error and bits-of-memorization metrics.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our package
from src.analysis.memorization_diagnostics_fixed import (
    calculate_squared_error_memorization_score,
    calculate_bits_of_memorization,
    generate_memorization_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memorization_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for analysis."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format for {data_path}")
    
    # Basic validation
    required_columns = ['player_id', 'gw', 'total_points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Ensure data types are correct
    df['player_id'] = df['player_id'].astype(str)
    df['gw'] = pd.to_numeric(df['gw'])
    
    # Sort by player and gameweek
    df = df.sort_values(['player_id', 'gw']).reset_index(drop=True)
    
    return df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """Train an XGBoost regression model."""
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Log feature importance
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    logger.info("Top 10 most important features:")
    for feat, imp in feature_importance.head(10).items():
        logger.info(f"  {feat}: {imp:.4f}")
    
    return model


def run_analysis(
    df: pd.DataFrame,
    target_col: str = 'total_points',
    position_col: str = 'position',
    output_dir: str = 'reports',
    n_samples: int = 5,
    min_gw: int = 6,  # Skip first few gameweeks to allow for rolling statistics
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Run the full memorization analysis pipeline."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data
    df = df[df['gw'] >= min_gw].copy()
    
    # Prepare features and target
    non_feature_cols = ['player_id', 'gw', target_col, position_col, 'name', 'team']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Ensure we only use numeric features
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    logger.info(f"Using {len(numeric_cols)} numeric features for modeling")
    
    # Split into train/test by time (not random for time series)
    max_gw = df['gw'].max()
    test_split_gw = max_gw - int((max_gw - min_gw) * test_size)
    
    train_mask = df['gw'] < test_split_gw
    test_mask = df['gw'] >= test_split_gw
    
    X_train = df.loc[train_mask, numeric_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, numeric_cols]
    y_test = df.loc[test_mask, target_col]
    
    logger.info(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Prepare data for memorization analysis
    train_df = df[train_mask].copy()
    train_df['y_pred'] = model.predict(X_train)
    
    # Calculate memorization scores
    logger.info("Calculating squared error memorization scores...")
    se_results = calculate_squared_error_memorization_score(
        model=model,
        train_df=train_df,
        target_col=target_col,
        group_col='player_id',
        time_col='gw',
        position_col=position_col,
        feature_cols=numeric_cols,
        return_df=True
    )
    
    logger.info("Calculating bits of memorization...")
    bits_results = calculate_bits_of_memorization(
        model=model,
        train_df=train_df,
        target_col=target_col,
        group_col='player_id',
        time_col='gw',
        position_col=position_col,
        feature_cols=numeric_cols,
        return_df=True,
        variance_estimator='heteroscedastic_rolling',
        window_size=5
    )
    
    # Generate report
    logger.info("Generating report...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'memorization_report_{timestamp}.md')
    
    report = generate_memorization_report(
        results_list=[se_results, bits_results],
        metric_names=['Squared Error', 'Bits of Memorization'],
        output_path=report_path,
        n_samples=n_samples
    )
    
    # Calculate test metrics
    test_preds = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    
    # Prepare results
    results = {
        'model': model,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'se_results': se_results,
        'bits_results': bits_results,
        'report_path': os.path.abspath(report_path),
        'feature_importance': dict(zip(numeric_cols, model.feature_importances_))
    }
    
    logger.info(f"Analysis complete! Report saved to {results['report_path']}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run enhanced memorization analysis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data file (CSV, Parquet, or Pickle)')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory to save output files')
    parser.add_argument('--target', type=str, default='total_points',
                       help='Name of the target column')
    parser.add_argument('--position-col', type=str, default='position',
                       help='Name of the position column')
    parser.add_argument('--min-gw', type=int, default=6,
                       help='Minimum gameweek to include in analysis')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of top/bottom samples to show in report')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load and prepare data
        df = load_data(args.data)
        
        # Run analysis
        results = run_analysis(
            df=df,
            target_col=args.target,
            position_col=args.position_col,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            min_gw=args.min_gw,
            test_size=args.test_size
        )
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in memorization analysis: {str(e)}", exc_info=True)
        sys.exit(1)
