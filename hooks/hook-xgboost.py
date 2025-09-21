from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Include non-Python package data (e.g., VERSION) and native binaries (xgboost.dll)
datas = collect_data_files('xgboost', include_py_files=False)
binaries = collect_dynamic_libs('xgboost')
