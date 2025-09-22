#!/usr/bin/env python3
"""
Test script to verify XGBoost works in the built executable
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import xgboost as xgb
    print("✅ XGBoost imported successfully!")
    print(f"Version: {xgb.__version__}")

    # Try to create a simple model
    import numpy as np
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
    model.fit(X, y)
    print("✅ XGBoost model created and trained successfully!")

    # Test prediction
    pred = model.predict([[2, 3]])
    print(f"✅ Prediction successful: {pred}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
