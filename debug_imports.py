print("--- Starting import debugger ---")

try:
    print("Importing pandas...")
    import pandas
    print("Importing numpy...")
    import numpy
    print("Importing sklearn...")
    import sklearn
    print("Importing xgboost...")
    import xgboost
    print("Importing lightgbm...")
    import lightgbm
    print("Importing scipy...")
    import scipy
    print("Importing joblib...")
    import joblib
    print("Importing reportlab...")
    import reportlab
    print("Importing seaborn...")
    import seaborn
    print("Importing matplotlib...")
    import matplotlib
    print("Importing yaml...")
    import yaml
    print("Importing requests...")
    import requests
    print("Importing pulp...")
    import pulp
    # boruta is a module within the boruta package, not a top-level import
    # print("Importing boruta...")
    # import boruta
    print("Importing tqdm...")
    import tqdm
    print("Importing sqlite_utils...")
    import sqlite_utils
    print("Importing onnx...")
    import onnx
    print("Importing skl2onnx...")
    import skl2onnx
    print("Importing onnxmltools...")
    import onnxmltools
    print("Importing onnxruntime...")
    import onnxruntime
    print("Importing shap...")
    import shap
    print("Importing pydantic...")
    import pydantic
    print("Importing pydantic_settings...")
    import pydantic_settings
    print("Importing tenacity...")
    import tenacity
    print("Importing dotenv...")
    import dotenv
    print("Importing fpl...")
    import fpl
    print("Importing keyring...")
    import keyring
    print("Importing aiohttp...")
    import aiohttp
    print("Importing selenium...")
    import selenium
    print("--- All imports successful ---")

except Exception as e:
    print(f"!!! An error occurred during import: {e} !!!")
    import traceback
    traceback.print_exc()
