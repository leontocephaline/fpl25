from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os

# Include non-Python package data (e.g., VERSION) and native binaries (xgboost DLL)
datas = collect_data_files('xgboost', include_py_files=False)
binaries = collect_dynamic_libs('xgboost')

# Explicitly search for XGBoost DLLs without importing test submodules
try:
    import xgboost
    # libpath may be a function in new versions
    try:
        lib_path = xgboost.libpath()
    except TypeError:
        lib_path = getattr(xgboost, 'libpath', None)
    if lib_path:
        paths = lib_path if isinstance(lib_path, (list, tuple)) else [lib_path]
        for p in paths:
            if p and os.path.exists(p):
                for root, dirs, files in os.walk(p):
                    for file in files:
                        if file.lower().endswith('.dll'):
                            binaries.append((os.path.join(root, file), 'xgboost'))
    # Fallback: search package dir
    xgb_dir = os.path.dirname(xgboost.__file__)
    if xgb_dir and os.path.exists(xgb_dir):
        for root, dirs, files in os.walk(xgb_dir):
            # Skip testing and spark subpackages to avoid importing pytest/hypothesis
            if any(skip in root.replace('\\','/').lower() for skip in ['testing', 'spark']):
                continue
            for file in files:
                if file.lower().endswith('.dll') and 'xgboost' in file.lower():
                    binaries.append((os.path.join(root, file), '.'))
except Exception as e:
    print(f"Warning: XGBoost hook DLL scan issue: {e}")

# Tell PyInstaller to exclude testing/spark submodules (avoid hypothesis/pytest import)
excludedimports = [
    'xgboost.testing',
    'xgboost.spark',
    'pytest',
    'hypothesis',
]

print(f"XGBoost hook collected {len(datas)} data files and {len(binaries)} binaries")
