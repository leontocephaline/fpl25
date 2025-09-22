from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_all
import os
import sys

# Include non-Python package data (e.g., VERSION) and native binaries (xgboost.dll)
datas = collect_data_files('xgboost', include_py_files=False)
binaries = collect_dynamic_libs('xgboost')

# Also collect all XGBoost submodules and their binaries
try:
    # Get all XGBoost-related packages and modules
    xgb_packages = collect_all('xgboost')
    if 'binaries' in xgb_packages:
        binaries.extend(xgb_packages['binaries'])
    if 'datas' in xgb_packages:
        datas.extend(xgb_packages['datas'])
except Exception as e:
    print(f"Warning: Could not collect all XGBoost packages: {e}")

# Try to explicitly find XGBoost DLL files
try:
    import xgboost
    lib_path = xgboost.libpath
    if lib_path and os.path.exists(lib_path):
        for root, dirs, files in os.walk(lib_path):
            for file in files:
                if file.endswith('.dll'):
                    full_path = os.path.join(root, file)
                    # Add to binaries with proper destination
                    binaries.append((full_path, 'xgboost'))
except Exception as e:
    print(f"Warning: Could not find XGBoost library files: {e}")

# Also try to find XGBoost DLLs in common locations
try:
    # Check for XGBoost DLL in the package directory
    import xgboost
    xgb_dir = os.path.dirname(xgboost.__file__)
    if xgb_dir:
        for root, dirs, files in os.walk(xgb_dir):
            for file in files:
                if file.endswith('.dll') and 'xgboost' in file.lower():
                    full_path = os.path.join(root, file)
                    binaries.append((full_path, '.'))
                    print(f"Found XGBoost DLL: {full_path}")
except Exception as e:
    print(f"Warning: Could not search for XGBoost DLLs in package directory: {e}")

print(f"XGBoost hook collected {len(datas)} data files and {len(binaries)} binaries")
