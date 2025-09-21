#!/usr/bin/env python3
"""Debug script to test baseline model train/test split fixes"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.historical_db import FPLHistoricalDB
from src.analysis.historical_backtest import HistoricalBacktester
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_baseline_training():
    """Test if baseline models are using proper train/test splits"""
    
    # Initialize components
    db = FPLHistoricalDB()
    backtester = HistoricalBacktester(db)
    
    # Test data for 2023-24 GW5 (should have GW1-4 as training data)
    season = "2023-24"
    test_gw = 5
    
    # Get training and test data
    train_data = db.get_historical_data_for_training(season, test_gw)
    test_data = db.get_player_data(season, test_gw)
    
    print(f"\n🔍 Debug Results for {season} GW{test_gw}:")
    print(f"   Training data: {len(train_data)} records (GW1-{test_gw-1})")
    print(f"   Test data: {len(test_data)} records (GW{test_gw})")
    
    if train_data.empty:
        print("❌ No training data found!")
        return
    
    if test_data.empty:
        print("❌ No test data found!")
        return
        
    print(f"   Training GWs: {sorted(train_data['gw'].unique())}")
    print(f"   Test GW: {sorted(test_data['gw'].unique())}")
    
    # Test the baseline prediction method
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    
    try:
        predictions = backtester._run_benchmark_prediction(model, train_data, test_data)
        
        if predictions:
            pred_points = [p['predicted_points'] for p in predictions]
            actual_points = test_data['total_points'].values
            
            print(f"   Predictions: min={min(pred_points):.2f}, max={max(pred_points):.2f}, mean={sum(pred_points)/len(pred_points):.2f}")
            print(f"   Actuals: min={min(actual_points):.2f}, max={max(actual_points):.2f}, mean={sum(actual_points)/len(actual_points):.2f}")
            
            # Calculate RMSE manually
            import numpy as np
            rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(pred_points, actual_points)]))
            print(f"   RMSE: {rmse:.3f}")
            
            if rmse < 1.0:
                print("❌ RMSE suspiciously low - likely data leakage!")
            else:
                print("✅ RMSE looks realistic")
        else:
            print("❌ No predictions returned")
            
    except Exception as e:
        print(f"❌ Error in prediction: {e}")

if __name__ == "__main__":
    test_baseline_training()
