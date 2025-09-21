#!/usr/bin/env python3
"""
Exports historical actual results from the SQLite database to a CSV file.

This script is used to create the input file required by run_backtest_analysis.py.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys
import argparse

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.historical_db import FPLHistoricalDB

def export_actual_results(season: str | None = None):
    """Connects to the database, queries actual results, and saves to CSV.
    
    Args:
        season: Optional season filter (e.g., '2024-25'). If provided, only export that season.
    """
    db_path = Path(__file__).parent.parent / 'data' / 'fpl_history.db'
    output_path = Path(__file__).parent.parent / 'data' / 'actuals.csv'

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        # Base query
        base_query = (
            "SELECT player_id, season, gw AS gameweek, total_points AS actual_points FROM gameweeks"
        )

        params = None
        if season:
            query = base_query + " WHERE season = ?"
            params = (season,)
            output_path = output_path.with_name(f"actual_results_{season.replace('/', '-')}.csv")
            print(f"Filtering export to season: {season}")
        else:
            query = base_query

        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            print("No data found in gameweeks table.")
            return

        # Ensure output directory exists
        output_path.parent.mkdir(exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Successfully exported {len(df)} records to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export actual FPL results from DB to CSV")
    parser.add_argument("--season", type=str, default=None, help="Season to export (e.g., 2024-25). If omitted, exports all seasons.")
    args = parser.parse_args()
    export_actual_results(args.season)

