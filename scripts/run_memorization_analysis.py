"""
Script to run memorization diagnostics on the full dataset.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_importer import FPLDataImporter
from src.data.historical_db import FPLHistoricalDB
from src.analysis.memorization_diagnostics import (
    calculate_squared_error_memorization_score,
    calculate_bits_of_memorization,
    generate_memorization_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memorization_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    logger.info("Loading data...")
    
    # Initialize the database connection
    db_path = os.path.join('data', 'fpl_history.db')
    db = FPLHistoricalDB(db_path)
    
    # Get available seasons and select the most recent one
    available_seasons = db.get_available_seasons()
    if not available_seasons:
        raise ValueError("No seasons found in the database")
        
    current_season = available_seasons[-1]  # Get most recent season
    logger.info(f"Using season: {current_season}")
    
    # Get the latest gameweek for this season
    latest_gw = db.get_latest_gw(season=current_season)
    if latest_gw is None:
        raise ValueError(f"No gameweeks found for season {current_season}")
        
    logger.info(f"Latest gameweek available: {latest_gw}")
    
    # Load the data for this gameweek
    df = db.get_gameweek_data(gw=latest_gw, season=current_season)
    
    # Ensure we have the required columns
    required_columns = ['player_id', 'gw', 'season', 'name', 'position', 'team_id', 'total_points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the data: {', '.join(missing_columns)}")
    
    # Remove any rows with missing values in key columns
    initial_count = len(df)
    df = df.dropna(subset=['player_id', 'gw', 'total_points'])
    if len(df) < initial_count:
        logger.info(f"Filtered {initial_count - len(df)} records with missing values from {initial_count} total")
    
    logger.info(f"Loaded {len(df)} records for season {current_season}, GW {latest_gw}")
    
    # Ensure we have the target variable
    df['y_true'] = df['total_points']
    
    # Add season and gameweek as features
    df['season'] = current_season
    df['gw'] = latest_gw
    
    # Ensure position is properly encoded
    if 'position' in df.columns:
        # Map position to string if it's not already
        if not pd.api.types.is_string_dtype(df['position']):
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            df['position'] = df['position'].map(position_map)
        
        # Create one-hot encoded position columns
        position_dummies = pd.get_dummies(df['position'], prefix='pos')
        
        # Ensure we have all expected position columns
        expected_positions = ['pos_GK', 'pos_DEF', 'pos_MID', 'pos_FWD']
        for pos in expected_positions:
            if pos not in position_dummies.columns:
                position_dummies[pos] = 0
        
        # Add position dummies to the dataframe
        df = pd.concat([df, position_dummies], axis=1)
    
    return df

def select_features(df):
    """Select and preprocess features for the model."""
    # Make a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Define numeric features to use (excluding IDs and target variables)
    numeric_features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
        'threat', 'ict_index', 'starts', 'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'expected_goals_conceded', 'value',
        'transfers_balance', 'selected', 'transfers_in', 'transfers_out'
    ]
    
    # Only keep features that exist in the dataframe
    features = [f for f in numeric_features if f in df_processed.columns]
    
    # Add position columns if they exist
    position_columns = ['pos_GK', 'pos_DEF', 'pos_MID', 'pos_FWD']
    for pos in position_columns:
        if pos in df_processed.columns:
            features.append(pos)
    
    # Select only the features we want
    features_df = df_processed[features].copy()
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    return features_df, features, position_columns

def train_model(df):
    """Train a model for memorization analysis."""
    logger.info("Preparing data for modeling...")
    
    # Select features
    X, feature_names, position_cols = select_features(df)
    y = df['y_true'].values
    
    logger.info(f"Using {len(feature_names)} features: {', '.join(feature_names[:5])}...")
    
    # Simple train-test split (you might want to use time-based split in production)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Log model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model trained. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    # Store the feature names in the model for later use
    model.feature_names_ = feature_names
    
    return model, X, feature_names

def run_analysis():
    """Run the full memorization analysis pipeline."""
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Create a copy with all the original columns plus any we'll add
        analysis_df = df.copy()
        
        # Train a model and get features
        model, X, feature_cols = train_model(analysis_df)
        
        # Prepare the features for prediction
        X_pred, _, _ = select_features(analysis_df)
        
        # Ensure the features are in the same order as during training
        if hasattr(model, 'feature_names_'):
            X_pred = X_pred[model.feature_names_]
        
        # Add predictions to the analysis dataframe
        analysis_df['y_pred_model'] = model.predict(X_pred)
        
        # Ensure we have the required columns for analysis
        if 'player_id' not in analysis_df.columns:
            raise ValueError("player_id column not found in the dataframe")
            
        if 'gw' not in analysis_df.columns:
            raise ValueError("gw (gameweek) column not found in the dataframe")
            
        # Ensure we have all the feature columns in the analysis dataframe
        missing_cols = [col for col in feature_cols if col not in analysis_df.columns]
        if missing_cols:
            for col in missing_cols:
                if col.startswith('pos_'):
                    analysis_df[col] = 0
                    logger.warning(f"Added missing position column: {col}")
                else:
                    analysis_df[col] = 0
                    logger.warning(f"Added missing feature column with default value 0: {col}")
        
        # Run memorization analysis
        logger.info("Running squared error memorization analysis...")
        logger.debug(f"Features being used: {feature_cols}")
        logger.debug(f"Analysis dataframe columns: {analysis_df.columns.tolist()}")
        
        # Ensure the target column exists
        if 'y_true' not in analysis_df.columns:
            raise ValueError("y_true column not found in the dataframe")
        
        # Ensure all feature columns exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in analysis_df.columns]
        if missing_cols:
            raise ValueError(f"The following feature columns are missing from the dataframe: {missing_cols}")
        
        try:
            se_df, se_aggs = calculate_squared_error_memorization_score(
                model=model,
                train_df=analysis_df,
                feature_cols=feature_cols,
                target_col='y_true',
                group_col='player_id',
                time_col='gw',
                return_df=True
            )
        except Exception as e:
            logger.error(f"Error in calculate_squared_error_memorization_score: {str(e)}")
            logger.error(f"Feature columns: {feature_cols}")
            logger.error(f"DataFrame columns: {analysis_df.columns.tolist()}")
            raise
        
        logger.info("Running bits of memorization analysis...")
        bits_df, bits_aggs = calculate_bits_of_memorization(
            model=model,
            train_df=df,
            feature_cols=feature_cols,
            target_col='y_true',
            group_col='player_id',
            time_col='gw',
            variance_estimator="heteroscedastic",
            return_df=True
        )
        
        # Generate and save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/memorization_analysis_{timestamp}.html"
        
        logger.info(f"Generating report at {report_path}...")
        report = generate_memorization_report(
            results_list=[(se_df, se_aggs), (bits_df, bits_aggs)],
            metric_names=["Squared Error", "Bits of Memorization"],
            output_path=report_path
        )
        
        logger.info("Analysis complete!")
        return report_path
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    report_path = run_analysis()
    print(f"Report generated at: {report_path}")
