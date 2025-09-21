# Global Feature Superset

## Overview

The global feature superset is a mechanism to ensure consistent feature usage across all ML models during training and prediction in the Fantasy Premier League backtesting system. This solves the "shrinking feature-set" problem where early gameweeks would train models on very limited features, causing later predictions to use only those minimal features despite richer data becoming available.

## Problem Solved

**Before**: XGBoost models trained in early gameweeks (GW1-3) would only see basic features like `creativity`, `ict_index`, `influence`, etc. Later in the season, when lagged and rolling features became available, the models would still only use the original minimal feature set, wasting valuable engineered features.

**After**: All models use a consistent superset of safe features throughout the entire backtest, ensuring fair comparisons and optimal feature utilization.

## Implementation

### 1. Feature Superset Generation

The `HistoricalBacktester._generate_feature_superset()` method:
- Samples gameweeks across seasons to discover all available safe features
- Applies the same safety filtering used during normal training
- Collects all numeric features excluding identifiers and targets
- Saves the sorted superset to `models/fpl_safe_superset_features.txt`

### 2. MLPredictor Integration

The `MLPredictor` class now:
- Loads the global feature superset on initialization
- Uses the superset for both training and prediction feature alignment
- Falls back to per-model expected features if no superset exists
- Fills missing features with zeros and drops extra features consistently

### 3. Automatic Generation

The superset is generated automatically:
- At the start of each backtest run (if missing or `--retrain` is used)
- Before any model training begins
- Using a sample of gameweeks to avoid excessive data scanning

## Files

- **`models/fpl_safe_superset_features.txt`**: Contains the sorted list of safe features (one per line)
- **`src/analysis/historical_backtest.py`**: Contains superset generation logic
- **`src/models/ml_predictor.py`**: Contains superset loading and usage logic

## Configuration

The feature superset respects existing configuration:
- `ml.quiet_feature_logs`: Controls verbosity of feature alignment logging
- Safety filtering still applies (excludes outlier-prone features)
- Model-specific hyperparameters remain unchanged

## Benefits

1. **Consistent Features**: All models use the same feature set across all gameweeks
2. **Fair Benchmarking**: Models can be compared on equal footing
3. **Optimal Feature Usage**: Later gameweeks benefit from all available engineered features
4. **Stable Performance**: Eliminates feature-set shrinkage issues
5. **Automatic Management**: No manual intervention required

## Example Usage

```bash
# Generate new superset and retrain models
.venv/Scripts/python.exe -m scripts.run_historical_backtest --seasons 2022-23 --gw-range 1 10 --retrain

# Use existing superset
.venv/Scripts/python.exe -m scripts.run_historical_backtest --seasons 2022-23 --gw-range 1 10
```

## Current Superset Features

As of the latest generation, the superset includes:
- `creativity`
- `ict_index` 
- `influence`
- `threat`
- `transfers_balance`
- `transfers_in`
- `transfers_out`
- `value`

This list will expand as more seasons and gameweeks are processed, automatically capturing new engineered features like lagged statistics and rolling averages.
