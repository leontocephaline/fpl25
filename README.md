# Fantasy Premier League Optimizer

A sophisticated Fantasy Premier League (FPL) team optimizer that uses machine learning and mathematical optimization to select the best possible team according to official FPL rules.

## Features

- **ML-Based Player Prediction**: Uses XGBoost and LightGBM models to predict player performance
- **MILP Optimization**: Mixed Integer Linear Programming for optimal team selection
- **Captain/Vice-Captain Selection**: Automatic selection based on highest predicted points
- **Comprehensive Rule Compliance**: Fully implements all official FPL rules
- **2025/26 Rule Support**: Includes updated defensive contributions, revised assists, and new chip system
- **AFCON Transfer Support**: Handles additional transfers during African Cup of Nations period

## Official FPL Rules Implementation

### Squad Composition

The optimizer enforces the official FPL squad requirements:

- **2 Goalkeepers**
- **5 Defenders**
- **5 Midfielders**
- **3 Forwards**
- **Total squad size: 15 players**

### Budget Constraints

- **£100 million budget limit**
- Player prices change during the season based on transfer market popularity
- Selling price may be less than purchase price due to 50% sell-on fee on profits

### Team Restrictions

- **Maximum 3 players per Premier League team**

### Starting XI Requirements

- **Exactly 11 players** selected from the 15-player squad
- **Formation flexibility** with minimum requirements:
  - 1 goalkeeper
  - At least 3 defenders
  - At least 2 midfielders
  - At least 1 forward
- **Any valid formation** within these constraints (e.g., 3-4-3, 3-5-2, 4-4-2, 4-5-1, 5-3-2)

### Scoring System

During the season, your fantasy football players will be allocated points based on their performance in the Premier League.

| Action                                                  | Points |
| ------------------------------------------------------- | ------ |
| For playing up to 60 minutes                            | 1      |
| For playing 60 minutes or more (excluding stoppage time)| 2      |
| For each goal scored by a goalkeeper                    | 10     |
| For each goal scored by a defender                      | 6      |
| For each goal scored by a midfielder                    | 5      |
| For each goal scored by a forward                       | 4      |
| For each goal assist                                    | 3      |
| For a clean sheet by a goalkeeper or defender           | 4      |
| For a clean sheet by a midfielder                       | 1      |
| For every 3 shot saves by a goalkeeper                  | 1      |
| For each penalty save                                   | 5      |
| For each penalty miss                                   | -2     |
| Bonus points for the best players in a match            | 1-3    |
| For every 2 goals conceded by a goalkeeper or defender  | -1     |
| For each yellow card                                    | -1     |
| For each red card                                       | -3     |
| For each own goal                                       | -2     |

### Transfer Rules

- **Initial squad**: Unlimited free transfers with no cost
- **Subsequent gameweeks**: 1 free transfer per gameweek
- **Additional transfers**: -4 points penalty per extra transfer
- **Saved transfers**: Can accumulate up to 5 free transfers
- **Transfer limit**: Maximum 20 transfers per gameweek (unless using chips)

### Chips

All chips are now available for both the first and second part of the season:

- **Bench Boost**: Bench players' points included in total
- **Free Hit**: Unlimited free transfers for one gameweek
- **Triple Captain**: Captain points tripled instead of doubled
- **Wildcard**: All transfers free of charge

Each chip type is available twice per season:
- First set: Available from start of season until Gameweek 19 deadline
- Second set: Available from Gameweek 20 until end of season

Bench Boost and Triple Captain chips can be cancelled before the Gameweek deadline. Free Hit and Wildcard chips cannot be cancelled once confirmed.

### African Cup of Nations (AFCON) Transfers

With AFCON taking place from December 21 to January 18, managers are given additional transfers to deal with player absence during this period:

- Following Gameweek 15 deadline, managers are topped up to maximum of 5 free transfers
- These free transfers can be carried over between gameweeks and used at any time

### Club Badge Creator with Adobe Express

New for the 2025/26 Fantasy season, you can create a club badge using Adobe Express and its Firefly generative AI tool:

- Upon entering your FPL team, you'll be prompted to generate a custom team badge
- Adobe Express will open and ask you to enter a prompt for the Firefly AI to use when generating images
- You'll need to log in or create an Adobe account to view your images and continue

## Technical Implementation

### Data Pipeline

1. **FPL API Integration**: Fetches real-time player and fixture data
2. **Data Processing**: Cleans and engineers features for ML models
3. **ML Prediction**: XGBoost/LightGBM models predict player points
4. **Optimization**: MILP solver selects optimal team composition
5. **Captain Selection**: Chooses captain/vice-captain from starting XI
6. **Risk Analysis**: Evaluates player risk factors

### ML Models

- **Target Variable**: `total_points` (official FPL scoring)
- **Features**: Form metrics, goal/assist rates, expected goals/assists, defensive contributions
- **Models**: XGBoost and LightGBM ensemble
- **Prediction Horizon**: Configurable gameweeks (default: 4)

### Optimization Engine

- **Constraints**: Budget, position requirements, team limits, formation rules
- **Objective**: Maximize expected points including captain bonus
- **Solver**: PuLP with CBC backend

### Model Diagnostics

To improve model interpretability and performance, several diagnostic tools are available, controlled via `config.yaml`:

- **Feature Importance**: Saves a CSV report of the most influential features for each model after training. Enable with `save_feature_importance: true`.
- **Per-GW/Position Diagnostics**: Generates a detailed CSV report and a heatmap plot of model performance (RMSE, MAE, R²) for each gameweek and position. Enable with `enable_extended_diagnostics: true`.
- **Residual Analysis**: Performs residual analysis by bucketing players based on their average minutes played. Produces a summary CSV and a box plot to identify systematic biases. Enable with `enable_residual_analysis: true`.

## Usage

Use the unified CLI with subcommands. Quiet is the default recommendation for console runs.

```bash
# Weekly report (quiet)
python -m fpl_weekly_updater weekly --quiet

# Weekly + appendix (also generates the appendix PDF)
python -m fpl_weekly_updater weekly --appendix --quiet

# Backtest (wraps existing analysis script)
python -m fpl_weekly_updater backtest --actual-data path/to/actuals.csv --gameweek-range 1 10 --generate-report

# Store FPL password securely in OS keyring
python -m fpl_weekly_updater set-password --email you@example.com

# Build an initial 15-player squad (JSON and CSV outputs)
python -m fpl_weekly_updater init-team --budget 100.0 --lock "Haaland,Saka" --report-dir reports
```

Executable (optional) after building with PyInstaller:

```powershell
# Build one-file EXE
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1

# Run weekly (quiet) via EXE
.\dist\FPLWeeklyUpdater.exe weekly --quiet
```

See detailed command help in `docs/CLI.md`.

## Quick Run Cheatsheet (EXE)

- **Build onedir EXE (with xgboost/lightgbm hooks)**

```powershell
# From the project root (PowerShell)
$proj = (Get-Location).Path
.\.venv\Scripts\python.exe -m PyInstaller `
  --noconfirm --onedir --name FPLWeeklyUpdater --console `
  --additional-hooks-dir .\hooks `
  --add-data "$proj\config.yaml;." `
  --add-data "$proj\scripts;scripts" `
  --add-data "$proj\src;src" `
  --collect-all xgboost `
  --collect-binaries xgboost `
  --collect-all lightgbm `
  --collect-binaries lightgbm `
  scripts/entry_cli.py
```

- **Run (appendix)**
.\dist\FPLWeeklyUpdater\FPLWeeklyUpdater.exe appendix `
  --actual-data 'reports\actuals_2024-25.csv' `
  --predictions-dir 'data\backtest' `
  --output-dir 'reports'

- **Run (appendix) with custom output paths**
.\out\dist\FPLWeeklyUpdater\FPLWeeklyUpdater.exe appendix `
  --actual-data 'reports\actuals_2024-25.csv' `
  --predictions-dir 'data\backtest' `
  --output-dir 'reports'

- **Run (weekly, no news)**

```powershell
.\dist\FPLWeeklyUpdater\FPLWeeklyUpdater.exe weekly `
  --no-news `
  --report-dir 'reports'

# or if using out\dist
.\out\dist\FPLWeeklyUpdater\FPLWeeklyUpdater.exe weekly `
  --no-news `
  --report-dir 'reports'
```

- **Verify outputs**

```powershell
Get-ChildItem .\reports | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime,Length
```

### Notes & Troubleshooting

- **Do not run** the EXE from the `build/` folder; always run from `dist/` (or your configured `out/dist/`).
- The **hooks** directory (`hooks/hook-xgboost.py`) ensures xgboost's data (`VERSION`) and the native DLL are bundled. Keep `--additional-hooks-dir .\hooks` in your build command.
- If you see `XGBoostLibraryNotFound` or missing `xgboost\VERSION`, rebuild using the command above. As a fallback, you can copy the entire `xgboost` package from `.venv/Lib/site-packages/xgboost` into `_internal/xgboost` under the EXE folder.
- If pytest plugins cause build noise, the build script sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` during analysis.
- If you don't have `Activate.ps1` in `.venv`, that's fine—call the venv Python directly: `.\.venv\Scripts\python.exe -m <module>`.

## Output

The optimizer provides detailed team information including:
- Optimal team formation
- Captain and vice-captain selections
- Starting XI with positions, costs, and predicted points
- Bench players
- Team player counts (with per-team limit checking)
- Expected points and total cost

## Configuration

The system is highly configurable via `config.yaml`, allowing for strategic adjustments to the optimization process. Key settings include:

- **API & System Settings**: Configure API endpoints, rate limits, and logging levels.
- **ML Model Parameters**: Adjust parameters for XGBoost, LightGBM, and the ensemble model.
- **Optimization Constraints**: Fine-tune the core rules and strategic levers:
  - `budget`: Set the total team budget (e.g., `100.0` million).
  - `squad_size`, `starting_xi_size`, `max_per_team`: Define the fundamental squad rules.
  - `positions`: Specify exact squad counts and starting XI minimums/maximums for each position (GK, DEF, MID, FWD).
  - `formations`: Assign weights to preferred formations (e.g., `3-5-2`, `4-4-2`) to guide the optimizer's selection.
  - `bench_weights`: Assign weights to the four substitutes to influence their priority in the objective function.
  - `strategy`: Control advanced strategic choices:
    - `premium_limit`: Set the maximum number of high-cost premium players.
    - `premium_thresholds`: Define the cost thresholds for a player to be considered premium, specified per position.

## Benchmarking

The model's performance was benchmarked using historical backtests for the `2023-24` season. The introduction of lagged, rolling, and aggregate features resulted in a notable improvement in predictive accuracy:

- **Model with new features**: RMSE = **3.432**
- **Model without new features**: RMSE = **3.515**

This demonstrates the value of the enhanced feature engineering in reducing prediction error.

## Testing

Unit tests ensure correctness of:
- Data processing pipeline
- ML prediction models
- Optimization constraints
- Captain selection logic

Run tests with:

```bash
python -m pytest tests/
```

## Weekly Automation (Windows Task Scheduler)

This project includes a unified PowerShell script that runs both the weekly optimizer and the historical backtest (with PDF generation). Use this single script for manual runs and for scheduling.

Script path: `run_weekly_update.ps1`

What it does:

- Activates the local virtual environment (`.venv`).
- Runs `main.py` (weekly optimizer). Optional `--retrain` is supported via the script parameter.
- Runs `scripts/run_historical_backtest.py --generate-pdf` to produce a fresh PDF report on your Desktop.

Manual usage examples (PowerShell):

```powershell
# Default (uses team_data.json if present, 1 free transfer, £0 bank)
./run_weekly_update.ps1

# Custom team JSON and bank balance
./run_weekly_update.ps1 -TeamDataJsonPath ".\team_data.json" -FreeTransfers 2 -Bank 1.5

# Force model retraining
./run_weekly_update.ps1 -Retrain
```

Scheduling via Windows Task Scheduler:

1. Open Task Scheduler and create a new task.
2. General tab:
   - Name: `FPL Weekly Update`
   - Run whether user is logged on or not
   - Configure for: Windows 10/11
3. Triggers tab:
   - New…
   - Weekly, select preferred day/time before the FPL deadline
4. Actions tab:
   - New…
   - Action: Start a program
   - Program/script: `powershell.exe`
   - Add arguments: `-ExecutionPolicy Bypass -File "<ABSOLUTE_PATH_TO_PROJECT>\run_weekly_update.ps1" -TeamDataJsonPath "<ABSOLUTE_PATH_TO_PROJECT>\team_data.json"`
   - Start in: `<ABSOLUTE_PATH_TO_PROJECT>` (project root directory)
5. Conditions and Settings tabs: configure to taste (e.g., wake computer, stop task if runs longer than X hours).
6. Save. When prompted, provide credentials so it can run while logged out if desired.

Notes:

- Ensure the `.venv` exists. The script will create one if missing and install requirements from `requirements.txt`.
- The backtest PDF is written to your Desktop as `backtest_report_<RUN_ID>.pdf`.
- To debug, run the script manually from a PowerShell terminal first to validate output.
