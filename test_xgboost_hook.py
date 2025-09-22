#!/usr/bin/env python3

import xgboost
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

print(f"XGBoost version: {xgboost.__version__}")
print(f"XGBoost lib path: {xgboost.libpath}")

# Check what collect_dynamic_libs finds
binaries = collect_dynamic_libs('xgboost')
print(f"PyInstaller found {len(binaries)} XGBoost binaries:")
for binary in binaries[:10]:  # Show first 10
    print(f"  {binary}")

# Also check the actual files in the lib path
if os.path.exists(xgboost.libpath):
    print(f"\nActual DLL files in {xgboost.libpath}:")
    for root, dirs, files in os.walk(xgboost.libpath):
        for file in files:
            if file.endswith('.dll'):
                print(f"  {os.path.join(root, file)}")
