#!/usr/bin/env python3
"""
Test script for the enhanced memorization diagnostics.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the fixed module
from src.analysis.memorization_diagnostics_fixed import (
    calculate_bits_of_memorization,
    rolling_mean_predictor
)
from sklearn.ensemble import RandomForestRegressor

def generate_test_data(n_players=100, n_gws=30, seed=42):
    """Generate test data with realistic FPL-like patterns."""
    np.random.seed(seed)
    
    # Generate player IDs and positions
    positions = ['GK', 'DEF', 'MID', 'FWD']
    position_weights = [0.15, 0.35, 0.35, 0.15]
    
    players = []
    for i in range(n_players):
        position = np.random.choice(positions, p=position_weights)
        base_ability = np.random.normal(loc=2.0, scale=1.0)
        players.append({
            'player_id': f'P{i:03d}',
            'position': position,
            'base_ability': base_ability
        })
    
    # Generate gameweek data
    data = []
    for player in players:
        player_id = player['player_id']
        position = player['position']
        base_ability = player['base_ability']
        
        # Position-specific adjustments
        if position == 'GK':
            base_points = np.random.normal(3.0, 1.0)
            home_boost = 0.2
        elif position == 'DEF':
            base_points = np.random.normal(2.5, 0.8)
            home_boost = 0.3
        elif position == 'MID':
            base_points = np.random.normal(3.5, 1.2)
            home_boost = 0.4
        else:  # FWD
            base_points = np.random.normal(3.0, 1.5)
            home_boost = 0.5
        
        # Generate gameweek data with some seasonality
        for gw in range(1, n_gws + 1):
            # Weekly variation
            week_effect = 0.5 * np.sin(2 * np.pi * gw / 10)
            
            # Home/away effect
            is_home = gw % 2 == 0
            home_effect = home_boost if is_home else 0.0
            
            # Form effect (weighted average of last 3 games)
            form_effect = 0.0
            if gw > 3:
                last_3_games = [d['total_points'] for d in data[-3:]]
                form_effect = np.mean(last_3_games) / 2.0
            
            # Random noise
            noise = np.random.normal(0, 1.5)
            
            # Calculate points (clipped to be positive)
            points = max(0.1, base_points + week_effect + home_effect + form_effect + noise)
            
            # Add some features
            data.append({
                'player_id': player_id,
                'position': position,
                'gw': gw,
                'total_points': points,
                'minutes_played': 90 if np.random.random() > 0.1 else np.random.randint(1, 90),  # 10% chance of not playing full match
                'was_home': is_home,
                'influence': base_ability + np.random.normal(0, 0.5),
                'creativity': base_ability + np.random.normal(0, 0.5),
                'threat': base_ability + np.random.normal(0, 0.5) + (0.5 if position in ['MID', 'FWD'] else 0),
                'ict_index': base_ability * 2 + np.random.normal(0, 0.5),
                'form': np.clip(base_ability / 2 + np.random.normal(0, 0.3), 0, 10),
                'selected': np.random.randint(1000, 1000000),
                'value': np.random.normal(50, 10)
            })
    
    return pd.DataFrame(data)

def test_memorization_diagnostics():
    """Run the memorization diagnostics on test data."""
    print("Generating test data...")
    df = generate_test_data(n_players=100, n_gws=30)
    
    # Sort by player and gameweek
    df = df.sort_values(['player_id', 'gw']).reset_index(drop=True)
    
    # Split into train/test (last 5 GWs as test)
    test_gws = df['gw'].max() - 4
    train_df = df[df['gw'] < test_gws].copy()
    
    # Prepare features and target
    feature_cols = ['minutes_played', 'was_home', 'influence', 'creativity', 'threat', 
                   'ict_index', 'form', 'selected', 'value']
    
    X_train = train_df[feature_cols]
    y_train = train_df['total_points']
    
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("Calculating memorization...")
    results, metrics = calculate_bits_of_memorization(
        model=model,
        train_df=train_df,
        target_col='total_points',
        group_col='player_id',
        time_col='gw',
        position_col='position',
        feature_cols=feature_cols,
        min_std_dev=0.5,
        max_std_dev=10.0,
        window_size=5,
        clip_residuals=10.0
    )
    
    # Print metrics
    print("\n=== Memorization Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    output_dir = Path('reports/memorization_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'memorization_results_{timestamp}.csv'
    metrics_path = output_dir / f'metrics_{timestamp}.json'
    
    results.to_csv(results_path, index=False)
    pd.Series(metrics).to_json(metrics_path, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")
    
    return results, metrics

if __name__ == "__main__":
    test_memorization_diagnostics()
