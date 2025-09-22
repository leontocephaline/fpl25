#!/usr/bin/env python3

"""
Test script to verify that the ML predictor can work without XGBoost.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Mock XGBoost to test fallback behavior
class MockXGBoostModule:
    """Mock XGBoost module that raises ImportError when imported"""
    def __getattr__(self, name):
        raise ImportError(f"No module named 'xgboost'")

# Temporarily replace xgboost in sys.modules
import sys
original_modules = {}
for module_name in list(sys.modules.keys()):
    if module_name.startswith('xgboost'):
        original_modules[module_name] = sys.modules[module_name]

# Remove XGBoost modules to simulate missing XGBoost
for module_name in list(sys.modules.keys()):
    if module_name.startswith('xgboost'):
        del sys.modules[module_name]

# Import our ML predictor with XGBoost unavailable
try:
    from models.ml_predictor import MLPredictor
    print("✅ Successfully imported MLPredictor without XGBoost")

    # Test that we can create an instance
    config = {
        'ml': {
            'model_type': 'lightgbm'
        }
    }
    predictor = MLPredictor(config)
    print(f"✅ Successfully created MLPredictor instance")
    print(f"✅ XGBoost available: {predictor._xgboost_available}")

    # Test with ensemble mode (should fall back to LightGBM only)
    config['ml']['model_type'] = 'ensemble'
    predictor2 = MLPredictor(config)
    print(f"✅ Successfully created ensemble MLPredictor without XGBoost")
    print(f"✅ XGBoost available: {predictor2._xgboost_available}")

except Exception as e:
    print(f"❌ Failed to import or create MLPredictor: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore original XGBoost modules
    for module_name, module in original_modules.items():
        sys.modules[module_name] = module
