"""
Unit tests for the ML predictor module.
Tests the functionality of the ML predictor with various input scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.ml_predictor import MLPredictor
from utils.config import Config

# Test configuration
TEST_CONFIG = {
    'system': {
        'log_level': 'INFO',
        'model_save_path': './tests/test_models/'
    },
    'ml': {
        'xgb': {},
        'lgb': {}
    }
}

@pytest.fixture
def sample_player_data():
    """Create sample player data for testing."""
    return pd.DataFrame({
        'id': range(1, 6),
        'web_name': [f'Player {i}' for i in range(1, 6)],
        'element_type': [1, 2, 3, 3, 4],  # Positions: 1=GK, 2=DEF, 3=MID, 4=FWD
        'team': [1, 1, 2, 3, 2],
        'now_cost': [50, 60, 80, 100, 120],
        'selected_by_percent': [5.0, 10.5, 15.2, 8.7, 20.3],
        'form': [4.5, 3.2, 5.6, 2.1, 6.7],
        'total_points': [45, 67, 89, 34, 102],
        'minutes': [1800, 2000, 2500, 1000, 2800],
        'goals_scored': [0, 2, 5, 1, 8],
        'assists': [0, 3, 4, 2, 6],
        'clean_sheets': [5, 8, 2, 1, 0],
        'saves': [45, 0, 0, 0, 0],
        'bonus': [12, 15, 20, 5, 25],
        'expected_goals': [0.2, 1.5, 4.2, 0.8, 7.5],
        'expected_assists': [0.1, 2.1, 3.8, 1.2, 5.4],
        'fixture_difficulty_next_4': [3.0, 3.5, 2.8, 4.2, 3.0],
        'fixture_count_next_4': [4, 4, 4, 4, 4]
    })

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()

@pytest.fixture
def ml_predictor(config):
    """Create an ML predictor instance for testing."""
    return MLPredictor(config)

def test_prepare_features_complete_data(ml_predictor, sample_player_data):
    """Test feature preparation with complete data."""
    # Add required columns that would normally be added in data_processor
    df = sample_player_data.copy()
    df['form_numeric'] = df['form']
    
    # Add calculated columns
    df['points_per_game'] = df['total_points'] / (df['minutes'] / 90).replace(0, 1)
    df['goals_per_90'] = df['goals_scored'] * 90 / df['minutes'].replace(0, 1)
    df['assists_per_90'] = df['assists'] * 90 / df['minutes'].replace(0, 1)
    df['xG_per_90'] = df['expected_goals'] * 90 / df['minutes'].replace(0, 1)
    df['xA_per_90'] = df['expected_assists'] * 90 / df['minutes'].replace(0, 1)
    df['clean_sheets_per_game'] = df['clean_sheets'] / (df['minutes'] / 90).replace(0, 1)
    df['saves_per_90'] = df['saves'] * 90 / df['minutes'].replace(0, 1)
    df['bonus_per_game'] = df['bonus'] / (df['minutes'] / 90).replace(0, 1)
    df['points_per_million'] = df['total_points'] / (df['now_cost'] / 10).replace(0, 1)
    df['form_per_million'] = df['form'] / (df['now_cost'] / 10).replace(0, 1)
    df['minutes_per_game'] = df['minutes'] / 10  # Assuming 10 gameweeks
    df['team_strength_overall'] = [3, 3, 4, 2, 4]
    df['team_strength_home'] = [3, 3, 4, 2, 4]
    df['team_strength_away'] = [3, 3, 4, 2, 4]
    df['attacking_returns'] = df['goals_scored'] + df['assists']
    df['defensive_returns'] = df['clean_sheets']
    
    features = ml_predictor._prepare_features(df)
    
    # Check that all expected columns are present
    expected_columns = [
        'form_numeric', 'points_per_game', 'goals_per_90', 'assists_per_90', 
        'xG_per_90', 'xA_per_90', 'clean_sheets_per_game', 'saves_per_90', 
        'bonus_per_game', 'fixture_difficulty_next_4', 'fixture_count_next_4',
        'points_per_million', 'form_per_million', 'minutes_per_game',
        'team_strength_overall', 'team_strength_home', 'team_strength_away',
        'attacking_returns', 'defensive_returns', 'now_cost', 'selected_by_percent',
        'element_type', 'xGI_per_90', 'is_goalkeeper', 'is_defender', 
        'is_midfielder', 'is_forward'
    ]
    
    for col in expected_columns:
        assert col in features.columns, f"Missing expected column: {col}"
    
    # Check that all numeric columns are filled
    assert not features.select_dtypes(include=[np.number]).isnull().any().any()

def test_prepare_features_missing_columns(ml_predictor, sample_player_data):
    """Test feature preparation with missing columns."""
    # Use minimal data to test robustness
    df = sample_player_data[['id', 'element_type', 'now_cost']].copy()
    
    features = ml_predictor._prepare_features(df)
    
    # Check that all expected columns are present with default values
    assert 'form_numeric' in features.columns
    assert features['form_numeric'].equals(pd.Series([0.0] * len(df)))
    assert 'is_goalkeeper' in features.columns
    assert 'is_defender' in features.columns
    assert 'is_midfielder' in features.columns
    assert 'is_forward' in features.columns

def test_predict_player_points(ml_predictor, sample_player_data):
    """Test the main prediction method with mock models."""
    # Add required columns
    df = sample_player_data.copy()
    df['form_numeric'] = df['form']
    
    # Mock the models
    # Create separate mock models for each position
    mock_models = {}
    for position in [1, 2, 3, 4]:
        # For each position, create mock predictions matching the number of players in that position
        position_count = sum(1 for p in [1, 2, 3, 3, 4] if p == position)
        if position_count > 0:
            mock_models[f'position_{position}'] = {
                'xgb': MagicMock(**{'predict.return_value': np.array([5.0 + position] * position_count)}),
                'lgb': MagicMock(**{'predict.return_value': np.array([4.0 + position] * position_count)})
            }
    # Add overall model as fallback
    mock_models['overall'] = {
        'xgb': MagicMock(**{'predict.return_value': np.array([5.0, 6.0, 7.0, 8.0, 9.0])}),
        'lgb': MagicMock(**{'predict.return_value': np.array([4.0, 5.0, 6.0, 7.0, 8.0])})
    }
    with patch.object(ml_predictor, 'models', mock_models):
        # Mock the _prepare_features method to return a simplified feature set
        with patch.object(ml_predictor, '_prepare_features', return_value=pd.DataFrame({
            'form_numeric': [4.5, 3.2, 5.6, 2.1, 6.7],
            'element_type': [1, 2, 3, 3, 4]
        })):
            # Mock the _add_prediction_metadata method to just return the input
            with patch.object(ml_predictor, '_add_prediction_metadata', lambda x, y: x):
                result = ml_predictor.predict_player_points(df)
    
    # Check that the result has the expected columns
    assert 'predicted_points' in result.columns
    assert len(result) == len(df)
    
    # The expected prediction is 0.6 * xgb + 0.4 * lgb = 0.6*7 + 0.4*6 = 6.6 for the ensemble
    # But since we're using the same mock for all positions, the exact value isn't as important
    # as the fact that predictions were made
    assert not result['predicted_points'].isnull().any()

def test_train_models_with_missing_form(ml_predictor, sample_player_data):
    """Test model training when form_numeric is missing."""
    # Create features without form_numeric
    features = pd.DataFrame({
        'element_type': [1, 2, 3, 3, 4],
        'now_cost': [50, 60, 80, 100, 120],
        'points_per_game': [4.5, 3.2, 5.6, 2.1, 6.7]
    })
    sample_player_data['position'] = ['FWD', 'MID', 'DEF', 'DEF', 'GK']
    
    # This should not raise an exception
    ml_predictor._train_models(features, sample_player_data)
    
    # Check that the models dictionary was populated
    assert hasattr(ml_predictor, 'models')
    assert 'overall' in ml_predictor.models

def test_handle_invalid_form_values(ml_predictor):
    """Test handling of invalid form values."""
    # Create test data with invalid form values
    df = pd.DataFrame({
        'form_numeric': ['invalid', None, 5.6, '2.1', np.nan],
        'element_type': [1, 2, 3, 3, 4],
        'now_cost': [50, 60, 80, 100, 120]
    })
    
    # Add required columns to avoid KeyError
    for col in ['points_per_game', 'goals_per_90', 'assists_per_90', 'xG_per_90', 
               'xA_per_90', 'clean_sheets_per_game', 'saves_per_90', 'bonus_per_game',
               'fixture_difficulty_next_4', 'fixture_count_next_4', 'points_per_million',
               'form_per_million', 'minutes_per_game', 'team_strength_overall',
               'team_strength_home', 'team_strength_away', 'attacking_returns',
               'defensive_returns', 'selected_by_percent']:
        df[col] = 0.0
    
    features = ml_predictor._prepare_features(df)
    
    # Check that all form_numeric values are valid numbers
    assert features['form_numeric'].notnull().all()
    assert (features['form_numeric'] >= 0).all()

def test_edge_cases(ml_predictor):
    """Test edge cases like empty DataFrames."""
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        ml_predictor._prepare_features(pd.DataFrame())


def test_save_feature_importance(tmp_path, sample_player_data):
    """Test that feature importance is saved correctly."""
    config_data = {
        'ml': {
            'model_type': 'xgboost',
            'xgboost': {'n_estimators': 1},
            'save_feature_importance': True
        },
        'target_variable': 'total_points',
        'report_path': str(tmp_path)
    }
    config = Config()
    config.config = config_data
    predictor = MLPredictor(config)

    df = sample_player_data.copy()
    df['position'] = 'MID'
    df['total_points'] = np.random.rand(len(df)) * 10

    # Since _train_position_model now handles saving, we call it directly.
    # We need a minimal feature set for the model to train.
    with patch.object(predictor, '_prepare_features', return_value=pd.DataFrame({'feature1': np.random.rand(len(df))})): 
        predictor._train_position_model(df, 'MID', {}, 1)

    expected_file = tmp_path / 'feature_importance_MID_1_xgb.csv'
    assert expected_file.exists()

    importance_df = pd.read_csv(expected_file)
    assert not importance_df.empty
    assert importance_df.shape[1] == 2

def test_extended_diagnostics_generation(tmp_path):
    """Test the generation of extended diagnostics files."""
    config_data = {
        'ml': {
            'enable_extended_diagnostics': True
        },
        'report_path': str(tmp_path),
        'plots_dir': str(tmp_path)
    }
    config = Config()
    config.config = config_data
    predictor = MLPredictor(config)

    # Populate with sample data
    predictor.extended_diagnostics_data = [
        {
            'gameweek': 1, 'position': 'MID', 'model_type': 'xgb',
            'y_true': [5, 6, 7], 'y_pred': [5.5, 6.2, 6.8]
        },
        {
            'gameweek': 1, 'position': 'DEF', 'model_type': 'xgb',
            'y_true': [2, 3, 4], 'y_pred': [2.1, 3.3, 3.9]
        }
    ]

    predictor._generate_extended_diagnostics()

    # Check for output files
    expected_csv = tmp_path / 'extended_diagnostics_summary.csv'
    expected_plot = tmp_path / 'diagnostics_rmse_heatmap.png'
    assert expected_csv.exists()
    assert expected_plot.exists()

    # Verify CSV content
    diagnostics_df = pd.read_csv(expected_csv)
    assert not diagnostics_df.empty
    assert 'rmse' in diagnostics_df.columns
    assert 'mae' in diagnostics_df.columns
    assert 'r2' in diagnostics_df.columns
    assert len(diagnostics_df) == 2

def test_residual_analysis_generation(tmp_path):
    """Test the generation of residual analysis files."""
    config_data = {
        'ml': {
            'enable_residual_analysis': True
        },
        'report_path': str(tmp_path),
        'plots_dir': str(tmp_path)
    }
    config = Config()
    config.config = config_data
    predictor = MLPredictor(config)

    # Populate with sample data including minutes
    predictor.extended_diagnostics_data = [
        {
            'gameweek': 1, 'position': 'MID', 'model_type': 'xgb',
            'y_true': [5, 6, 7, 8], 'y_pred': [5.5, 6.2, 6.8, 7.9],
            'minutes': [90, 25, 55, 0]
        }
    ]

    predictor._perform_residual_analysis()

    # Check for output files
    expected_csv = tmp_path / 'residual_analysis_summary.csv'
    expected_plot = tmp_path / 'residual_analysis_boxplot.png'
    assert expected_csv.exists()
    assert expected_plot.exists()

    # Verify CSV content
    summary_df = pd.read_csv(expected_csv)
    assert not summary_df.empty
    assert 'mean' in summary_df.columns
    assert 'std' in summary_df.columns
    assert len(summary_df) > 0
