"""Test script to verify global feature superset implementation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.models.ml_predictor import MLPredictor
from src.analysis.historical_backtest import HistoricalBacktester
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_superset_generation():
    """Test that superset generation works"""
    logger.info("Testing superset generation...")
    
    # Create a minimal config
    config = {
        'database': {'path': 'data/fpl.db'},
        'ml': {'model_type': 'xgb', 'quiet_feature_logs': False}
    }
    
    # Initialize backtest
    backtest = HistoricalBacktester(config)
    
    # Generate superset for a small range
    seasons = ['2023-24']
    gw_range = (1, 5)
    
    features = backtest._generate_feature_superset(seasons, gw_range, sample_size=3)
    
    logger.info(f"Generated {len(features)} features in superset")
    logger.info(f"First 10 features: {features[:10]}")
    
    # Check file was created
    superset_path = Path('models') / 'fpl_safe_superset_features.txt'
    assert superset_path.exists(), "Superset file was not created"
    
    return features

def test_ml_predictor_uses_superset():
    """Test that MLPredictor loads and uses the superset"""
    logger.info("\nTesting MLPredictor superset usage...")
    
    config = {
        'ml': {'model_type': 'xgboost', 'quiet_feature_logs': False}
    }
    
    # Create MLPredictor
    predictor = MLPredictor(config)
    
    # Check if superset was loaded
    assert predictor.global_features is not None, "Global features not loaded"
    logger.info(f"MLPredictor loaded {len(predictor.global_features)} global features")
    
    return predictor.global_features

def main():
    try:
        # Test 1: Generate superset
        superset_features = test_superset_generation()
        
        # Test 2: Verify MLPredictor uses it
        loaded_features = test_ml_predictor_uses_superset()
        
        # Verify they match
        assert len(superset_features) == len(loaded_features), "Feature count mismatch"
        assert set(superset_features) == set(loaded_features), "Feature sets don't match"
        
        logger.info("\n✅ All tests passed! Global feature superset is working correctly.")
        logger.info(f"Total features in superset: {len(superset_features)}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
