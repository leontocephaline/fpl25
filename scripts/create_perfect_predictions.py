import pandas as pd
import argparse
from pathlib import Path
import shutil

def create_perfect_predictions(actuals_path: str, season: str, output_dir: str):
    """
    Generates 'perfect' prediction files from actual results for backtesting sanity checks.
    A 'perfect' prediction is one where predicted_points == actual_points.
    """
    print(f"Loading actuals from {actuals_path} for season {season}")
    try:
        actuals = pd.read_csv(actuals_path)
    except FileNotFoundError:
        print(f"Error: Actuals file not found at {actuals_path}")
        return

    # Filter for the specified season
    season_actuals = actuals[actuals['season'] == season].copy()
    if season_actuals.empty:
        print(f"No actuals found for season {season}. Exiting.")
        return

    gameweeks = sorted(season_actuals['gameweek'].unique())
    print(f"Found gameweeks: {gameweeks}")

    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Clearing existing directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for gw in gameweeks:
        gw_actuals = season_actuals[season_actuals['gameweek'] == gw].copy()

        # Create perfect predictions
        gw_actuals['predicted_points'] = gw_actuals['actual_points']

        # Use a consistent naming scheme that the backtester will recognize
        # The timestamp helps avoid caching issues and indicates it's a generated file
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"predictions_gw{gw:02d}_{timestamp}.csv"
        filepath = output_path / filename

        # Define the columns to be saved in the perfect prediction file.
        COLS_TO_SAVE = ['player_id', 'gameweek', 'predicted_points', 'actual_points', 'position']

        # Ensure player_id column exists, renaming from 'id' if necessary.
        if 'player_id' not in gw_actuals.columns and 'id' in gw_actuals.columns:
            gw_actuals.rename(columns={'id': 'player_id'}, inplace=True)

        # Filter to only the columns that exist in the dataframe to avoid errors.
        final_cols = [col for col in COLS_TO_SAVE if col in gw_actuals.columns]
        
        # Add any missing columns and fill with a default value (e.g., NaN or 0)
        for col in COLS_TO_SAVE:
            if col not in gw_actuals.columns:
                gw_actuals[col] = pd.NA

        if 'player_id' not in gw_actuals.columns:
            print(f"Warning: 'player_id' column missing for GW {gw}. Skipping.")
            continue

        # Save the selected columns to the CSV file.
        gw_actuals[COLS_TO_SAVE].to_csv(filepath, index=False)
        print(f"Generated perfect predictions for GW {gw} at {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perfect prediction files for backtesting sanity check.")
    parser.add_argument('--actuals-path', type=str, default='data/actual_results.csv', help='Path to the actual results CSV file.')
    parser.add_argument('--season', type=str, required=True, help='The season to generate predictions for (e.g., 2024-25).')
    parser.add_argument('--output-dir', type=str, default='data/backtest_sanity_check', help='Directory to save the perfect prediction files.')
    args = parser.parse_args()

    create_perfect_predictions(args.actuals_path, args.season, args.output_dir)
