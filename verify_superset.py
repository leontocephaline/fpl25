"""Simple verification of global feature superset implementation"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_feature_superset_creation():
    """Test creating a feature superset file manually"""
    print("Testing feature superset creation...")
    
    # Create a sample feature superset file
    superset_path = Path('models') / 'fpl_safe_superset_features.txt'
    superset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample features that would be found in FPL data
    sample_features = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'goals_conceded', 'goals_scored', 'ict_index', 'influence',
        'minutes', 'own_goals', 'penalties_missed', 'penalties_saved',
        'red_cards', 'saves', 'threat', 'yellow_cards',
        'form', 'points_per_game', 'selected_by_percent', 'transfers_in',
        'transfers_out', 'value', 'now_cost', 'cost_change_start',
        'cost_change_event', 'transfers_in_event', 'transfers_out_event'
    ]
    
    with open(superset_path, 'w') as f:
        for feature in sorted(sample_features):
            f.write(f"{feature}\n")
    
    print(f"✅ Created superset file with {len(sample_features)} features at {superset_path}")
    return superset_path

def test_ml_predictor_loads_superset():
    """Test that MLPredictor loads the superset"""
    print("\nTesting MLPredictor superset loading...")
    
    try:
        from src.models.ml_predictor import MLPredictor
        
        config = {'ml': {'model_type': 'xgb', 'quiet_feature_logs': False}}
        predictor = MLPredictor(config)
        
        if predictor.global_features:
            print(f"✅ MLPredictor loaded {len(predictor.global_features)} global features")
            print(f"First 5 features: {predictor.global_features[:5]}")
            return True
        else:
            print("❌ MLPredictor did not load global features")
            return False
            
    except Exception as e:
        print(f"❌ Error loading MLPredictor: {e}")
        return False

def main():
    print("=== Global Feature Superset Verification ===\n")
    
    # Test 1: Create superset file
    superset_path = test_feature_superset_creation()
    
    # Test 2: Verify MLPredictor loads it
    success = test_ml_predictor_loads_superset()
    
    if success:
        print("\n🎉 Global feature superset implementation is working!")
        print("\nNext steps:")
        print("1. Run a backtest to generate the real superset from data")
        print("2. Verify XGBoost uses consistent features across gameweeks")
    else:
        print("\n❌ Implementation needs debugging")
    
    return success

if __name__ == "__main__":
    main()
