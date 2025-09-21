# Data Dictionary

This document provides a comprehensive overview of the features used in the Fantasy Premier League (FPL) prediction models. It details the source of each feature, the transformations applied, and its role in the model.

## Feature List

The following table lists all features used as inputs for the XGBoost and LightGBM models.

| Feature Name | Description | Source(s) | Transformation |
| :--- | :--- | :--- | :--- |
| `form_numeric` | Player's current form score. | `form` | Numeric conversion of the 'form' string. |
| `points_per_game` | Average points scored per 90 minutes played. | `total_points`, `minutes` | `total_points / (minutes / 90)` |
| `goals_per_90` | Goals scored per 90 minutes played. | `goals_scored`, `minutes` | `goals_scored * 90 / minutes` |
| `assists_per_90` | Assists made per 90 minutes played. | `assists`, `minutes` | `assists * 90 / minutes` |
| `xG_per_90` | Expected goals per 90 minutes played. | `expected_goals`, `minutes` | `expected_goals * 90 / minutes` |
| `xA_per_90` | Expected assists per 90 minutes played. | `expected_assists`, `minutes` | `expected_assists * 90 / minutes` |
| `clean_sheets_per_game` | Average number of clean sheets per 90 minutes. | `clean_sheets`, `minutes` | `clean_sheets / (minutes / 90)` |
| `saves_per_90` | Saves made per 90 minutes (GK only). | `saves`, `minutes` | `saves * 90 / minutes` |
| `bonus_per_game` | Average bonus points per 90 minutes. | `bonus`, `minutes` | `bonus / (minutes / 90)` |
| `fixture_difficulty_next_4` | Average fixture difficulty over the next 4 gameweeks. | `fixtures_df` | Mean difficulty of upcoming fixtures. |
| `fixture_count_next_4` | Number of fixtures in the next 4 gameweeks. | `fixtures_df` | Count of upcoming fixtures. |
| `points_per_million` | Total points divided by the player's current cost. | `total_points`, `now_cost` | `total_points / now_cost` |
| `form_per_million` | Current form score divided by the player's cost. | `form`, `now_cost` | `form / now_cost` |
| `minutes_per_game` | Average minutes played per gameweek. | `minutes` | `minutes / total_gameweeks` |
| `team_strength_overall` | Overall strength rating of the player's team. | `teams_df` | Joined from teams data. |
| `team_strength_home` | Home strength rating of the player's team. | `teams_df` | Joined from teams data. |
| `team_strength_away` | Away strength rating of the player's team. | `teams_df` | Joined from teams data. |
| `attacking_returns` | Sum of goals scored and assists. | `goals_scored`, `assists` | `goals_scored + assists` |
| `defensive_returns` | Clean sheets (for defenders and goalkeepers). | `clean_sheets`, `element_type` | `clean_sheets` if player is GK or DEF. |
| `now_cost` | The player's current price in the game. | `now_cost` | Raw value. |
| `selected_by_percent` | Percentage of FPL managers who have selected the player. | `selected_by_percent` | Raw value. |
| `element_type` | A numeric code representing the player's position. | `element_type` | Raw value (1: GK, 2: DEF, 3: MID, 4: FWD). |

## Data Cleaning and Clipping

To ensure model stability and reduce the impact of outliers, the following fields are cleaned and clipped:

- `selected_by_percent`: Clipped to a range of [0, 100].
- `form`: Clipped to a range of [0, 15].
- `now_cost`: Clipped to a range of [3.0, 15.0] (in millions).

## Diagnostic Output Files

When enabled in `config.yaml`, the ML pipeline generates several diagnostic files in the `reports/` and `plots/` directories. These files provide insights into model performance and interpretability.

### Feature Importance

- **File**: `reports/feature_importance_{model_key}.csv`
- **Description**: A CSV file listing the features and their corresponding importance scores for a given model (e.g., `FWD_4_xgb`). Higher scores indicate a greater influence on the model's predictions.

### Per-Gameweek/Position Diagnostics

- **File**: `reports/extended_diagnostics_summary.csv`
- **Description**: A CSV summary of model performance metrics, grouped by gameweek, position, and model type (XGBoost or LightGBM).
- **Columns**:
  - `gameweek`: The gameweek number.
  - `position`: The player position (e.g., MID, DEF).
  - `model_type`: The model used for prediction (`xgb` or `lgbm`).
  - `rmse`: Root Mean Squared Error.
  - `mae`: Mean Absolute Error.
  - `r2`: R-squared score.

- **File**: `plots/diagnostics_rmse_heatmap.png`
- **Description**: A heatmap visualizing the RMSE for each position across different gameweeks, providing a quick overview of where the model performs well or poorly.

### Residual Analysis

- **File**: `reports/residual_analysis_summary.csv`
- **Description**: A CSV file summarizing the prediction residuals (actual - predicted points), grouped by player minutes buckets.
- **Columns**:
  - `minutes_bucket`: The bucket of average minutes played (e.g., '61-90 mins').
  - `model_type`: The model used for prediction.
  - `mean`: The average residual for that bucket.
  - `std`: The standard deviation of the residuals.
  - `count`: The number of players in the bucket.

- **File**: `plots/residual_analysis_boxplot.png`
- **Description**: A box plot visualizing the distribution of residuals for each minutes-played bucket, helping to identify systematic over- or under-prediction.
