# Fantasy Premier League Prediction System - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Initial Setup](#initial-setup)
   - [System Requirements](#system-requirements)
   - [Installation](#installation)
   - [Configuration](#configuration)
4. [Running the System](#running-the-system)
   - [Initial Data Collection](#initial-data-collection)
   - [Weekly Updates](#weekly-updates)
   - [Running Backtests](#running-backtests)
5. [Understanding the Output](#understanding-the-output)
   - [Report Locations](#report-locations)
   - [Interpreting Results](#interpreting-results)
6. [Advanced Usage](#advanced-usage)
   - [Customizing Features](#customizing-features)
   - [Model Tuning](#model-tuning)
   - [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Introduction

Welcome to the Fantasy Premier League Prediction System! This powerful tool combines machine learning and optimization to help you make data-driven decisions for your Fantasy Premier League team. The system provides:

- **Accurate Predictions**: Uses XGBoost and LightGBM models trained on historical data
- **Optimal Team Selection**: Implements Mixed Integer Linear Programming (MILP) to select the best team within FPL rules
- **Comprehensive Analysis**: Provides detailed reports and visualizations of predictions and performance

## Quick Start

1. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API keys** (see [Configuration](#configuration))

3. **Run the initial setup** (optional backtest warm-up):
   ```bash
   python -m fpl_weekly_updater backtest --actual-data data/actuals_2024-25.csv --gameweek-range 1 1 --generate-report
   ```

4. **Get your first weekly report**:
   ```bash
   python -m fpl_weekly_updater weekly --quiet
   ```

## Initial Setup

### System Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- 2GB+ free disk space
- Internet connection for API access

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Fantasy-Football
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # OR
   source .venv/bin/activate  # Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Copy the example configuration file:
   ```bash
   # Windows (PowerShell)
   Copy-Item .\docs\env.example .\.env
   # macOS/Linux
   cp ./docs/env.example ./.env
   ```

2. Edit `.env` and add your API keys:
   ```
   # Required: Fantasy Premier League API (no key needed)
   FPL_EMAIL=your.email@example.com
   FPL_PASSWORD=your_password
   
   # Optional: External APIs for additional data
   FOOTBALL_DATA_API_KEY=your_key_here
   ```

3. Configure model settings in `config.yaml`:
   ```yaml
   ml:
     model_type: xgb  # or 'lgbm' for LightGBM
     enable_feature_importance: true
     quiet_feature_logs: false
   
   optimization:
     budget: 100.0
     strategy:
       premium_limit: 3
       premium_thresholds:
         GK: 5.5
         DEF: 6.0
         MID: 9.0
         FWD: 10.0
   ```

## Running the System

### Initial Data Collection

Before making predictions, you need to collect historical data:

```bash
# Collect data for the 2022-23 season, gameweeks 1-38
python -m scripts.run_historical_backtest --seasons 2022-23 --gw-range 1 38 --retrain
```

This will:
1. Download historical FPL data
2. Train the prediction models
3. Generate a global feature superset
4. Save the trained models to the `models/` directory

### Weekly Updates

1. **Update player data** (run after each gameweek):
   ```bash
   python -m fpl_weekly_updater.main
   ```

2. **Generate new predictions** (before each deadline):
   ```bash
   python -m fpl_weekly_updater weekly --quiet
   ```
   
   This will output the optimal team selection to the console and save detailed reports to `reports/`.

### Running Backtests

To evaluate model performance on historical data via the unified CLI:

```bash
# Run backtest for a range of gameweeks
python -m fpl_weekly_updater backtest --actual-data data/actuals_2024-25.csv --gameweek-range 1 10 --generate-report

# Use a custom predictions directory or season tag (optional)
python -m fpl_weekly_updater backtest --actual-data data/actuals_2024-25.csv --predictions-dir data/backtest --season 2024-25 --generate-report
```

## Understanding the Output

### Report Locations

- **Team Selections**: `reports/team_selection_<timestamp>.json`
- **Backtest Results**: `reports/historical_backtest_<timestamp>.json`
- **Model Performance**: `reports/model_performance_<timestamp>.csv`
- **Feature Importance**: `reports/feature_importance_<timestamp>.csv`
- **Logs**: `logs/fpl_optimizer.log`

### Interpreting Results

1. **Team Selection Report** (`team_selection_*.json`):
   - `squad`: Your optimal 15-player squad
   - `starting_xi`: Recommended starting lineup
   - `captain`: Selected captain (double points)
   - `vice_captain`: Vice-captain (replaces captain if they don't play)
   - `transfers`: Recommended transfers (if any)
   - `expected_points`: Predicted points for the next gameweek

2. **Backtest Results** (`historical_backtest_*.json`):
   - `metrics`: Performance metrics (RMSE, MAE, R²)
   - `gameweeks`: Detailed results for each gameweek
   - `feature_importance`: Most influential features

## Advanced Usage

### Customizing Features

To modify the feature set:

1. Edit `src/models/feature_engineering.py`
2. Add/remove feature transformations
3. Regenerate the feature superset:
   ```bash
   python -m scripts.run_historical_backtest --seasons 2022-23 --retrain
   ```

### Model Tuning

1. Adjust hyperparameters in `config.yaml`:
   ```yaml
   ml:
     xgb_params:
       max_depth: 6
       learning_rate: 0.01
       n_estimators: 500
     lgbm_params:
       num_leaves: 31
       learning_rate: 0.05
       n_estimators: 500
   ```

2. Retrain the models:
   ```bash
   python -m scripts.run_historical_backtest --seasons 2022-23 --retrain
   ```

### Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Solution**: Ensure all dependencies are installed
  ```bash
  pip install -r requirements.txt
  ```

**Issue**: API rate limiting
- **Solution**: Add delays between requests in `fpl_weekly_updater/apis/fpl_auth.py`

**Issue**: Model performance issues
- **Solution**: Check `logs/fpl_optimizer.log` for warnings/errors
- Try increasing training data with more seasons

## FAQ

**Q: How often should I update the data?**
A: Run the weekly updater after each gameweek, typically on Monday or Tuesday.

**Q: Can I use this for other fantasy football leagues?**
A: The system is designed for FPL but can be adapted for other leagues with similar rules.

**Q: How accurate are the predictions?**
A: The model achieves an RMSE of ~3.4 points per player per gameweek, which is competitive with other prediction models.

**Q: Can I customize the team selection strategy?**
A: Yes, edit the optimization settings in `config.yaml` to adjust budget allocation, formation preferences, and player selection criteria.

**Q: How do I interpret the feature importance?**
A: Higher values indicate features that have a stronger influence on the model's predictions. Look for patterns in `reports/feature_importance_*.csv`.
