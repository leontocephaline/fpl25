import csv
from collections import defaultdict
import os

print("--- Script started ---")

# Path to the merged sample file
file_path = 'reports/merged_sample_20250904_144150.csv'

# Dictionaries to hold our analysis
player_predictions = defaultdict(set)
player_gameweeks = defaultdict(int)
position_errors = defaultdict(list)

print(f"Analyzing {file_path} for static predictions...")

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print("--- Script finished --- ")
else:
    print("File found. Starting processing...")
    # Read the CSV and populate dictionaries
    rows_processed = 0
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                player_id = int(row['player_id'])
                predicted_points = float(row['predicted_points'])
                actual_points = float(row['actual_points'])
                position = row['position']

                player_predictions[player_id].add(predicted_points)
                player_gameweeks[player_id] += 1
                position_errors[position].append(predicted_points - actual_points)
                rows_processed += 1
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e} - Row: {row}")
    
    print(f"Finished processing file. {rows_processed} rows processed.")

    # --- Analysis ---
    print("\n--- Analysis started ---")

    # 1. Static Prediction Analysis
    static_players = 0
    multi_prediction_players = 0

    for player_id, predictions in player_predictions.items():
        # Consider players with predictions in more than one gameweek
        if player_gameweeks[player_id] > 1:
            if len(predictions) == 1:
                static_players += 1
            else:
                multi_prediction_players += 1

    print("--- Static Prediction Analysis ---")
    if (static_players + multi_prediction_players) > 0:
        static_percentage = (static_players / (static_players + multi_prediction_players)) * 100
        print(f"Players with multiple gameweek predictions: {static_players + multi_prediction_players}")
        print(f"Players with only 1 unique prediction value: {static_players}")
        print(f"Percentage of players with static predictions: {static_percentage:.2f}%\n")
    else:
        print("No players with multiple gameweek predictions found to analyze.\n")

    # 2. Per-Position Bias Analysis
    print("--- Per-Position Bias (Prediction - Actual) ---")
    for position, errors in position_errors.items():
        if errors:
            average_bias = sum(errors) / len(errors)
            print(f"{position}: Average Bias = {average_bias:.2f} (based on {len(errors)} predictions)")
    print("----------------------------------------\n")

    print("Analysis complete.")

print("--- Script finished ---")
