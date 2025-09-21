"""
Unit tests for the data processor module.
Tests data cleaning, feature engineering, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_processor import DataProcessor
from utils.config import Config

@pytest.fixture
def sample_players_data():
    """Create sample players data for testing."""
    return {
        'elements': [
            {
                'id': 1,
                'web_name': 'Player 1',
                'element_type': 1,  # GK
                'team': 1,
                'now_cost': 50,
                'selected_by_percent': '5.0',
                'form': '4.5',
                'total_points': 45,
                'minutes': 1800,
                'goals_scored': 0,
                'assists': 0,
                'clean_sheets': 5,
                'saves': 45,
                'bonus': 12,
                'expected_goals': '0.2',
                'expected_assists': '0.1',
                'chance_of_playing_next_round': 100,
                'chance_of_playing_this_round': 100
            },
            {
                'id': 2,
                'web_name': 'Player 2',
                'element_type': 2,  # DEF
                'team': 1,
                'now_cost': 60,
                'selected_by_percent': '10.5',
                'form': '3.2',
                'total_points': 67,
                'minutes': 2000,
                'goals_scored': 2,
                'assists': 3,
                'clean_sheets': 8,
                'saves': 0,
                'bonus': 15,
                'expected_goals': '1.5',
                'expected_assists': '2.1',
                'chance_of_playing_next_round': 100,
                'chance_of_playing_this_round': 100
            }
        ],
        'teams': [
            {'id': 1, 'name': 'Team 1', 'strength': 3, 'strength_overall_home': 3, 'strength_overall_away': 3},
            {'id': 2, 'name': 'Team 2', 'strength': 4, 'strength_overall_home': 4, 'strength_overall_away': 4}
        ]
    }

@pytest.fixture
def sample_fixtures_data():
    """Create sample fixtures data for testing."""
    return pd.DataFrame([
        {'event': 1, 'team_h': 1, 'team_a': 2, 'team_h_difficulty': 3, 'team_a_difficulty': 4},
        {'event': 2, 'team_h': 2, 'team_a': 1, 'team_h_difficulty': 4, 'team_a_difficulty': 3},
        {'event': 3, 'team_h': 1, 'team_a': 3, 'team_h_difficulty': 2, 'team_a_difficulty': 4},
        {'event': 4, 'team_h': 3, 'team_a': 1, 'team_h_difficulty': 4, 'team_a_difficulty': 3}
    ])

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()

@pytest.fixture
def data_processor(config):
    """Create a DataProcessor instance for testing."""
    return DataProcessor(config)

@pytest.fixture
def sample_historical_players_data():
    """Fixture with historical player data over several gameweeks."""
    data = {
        'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'gameweek': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'team': [1, 1, 1, 2, 2, 2, 1, 1, 1],
        'total_points': [2, 5, 3, 6, 0, 8, 1, 1, 1],
        'ict_index': [5.0, 10.0, 8.0, 12.0, 2.0, 15.0, 3.0, 3.0, 3.0],
        'transfers_in': [100, 200, 150, 50, 10, 300, 0, 0, 0],
        'transfers_out': [10, 20, 15, 5, 1, 30, 0, 0, 0],
        'minutes': [90, 90, 90, 90, 45, 90, 90, 90, 90],
        'goals_scored': [0, 1, 0, 1, 0, 1, 0, 0, 0],
        'assists': [0, 0, 1, 0, 0, 1, 0, 0, 0],
        'clean_sheets': [0, 1, 0, 1, 0, 0, 0, 1, 0],
        'saves': [3, 1, 2, 0, 0, 0, 0, 0, 0],
        'bonus': [0, 1, 0, 2, 0, 3, 0, 0, 0],
        'goals_conceded': [1, 0, 2, 0, 1, 2, 1, 0, 2]
    }
    return pd.DataFrame(data)

def test_process_all_data_complete(data_processor, sample_players_data, sample_fixtures_data):
    """Test the main processing pipeline with complete data."""
    data_processor.config['feature_engineering'] = {
        'enable_lagged_features': True,
        'enable_fixture_trends': True,
        'enable_team_rolling_features': True
    }

    # The sample data needs a 'gameweek' column for team rolling features
    players_df = pd.json_normalize(sample_players_data, record_path='elements')
    players_df['gameweek'] = 1 # Mock current gameweek
    sample_players_data['elements'] = players_df.to_dict(orient='records')

    df, _, _ = data_processor.process_all_data(sample_players_data, sample_fixtures_data)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    expected_columns = [
        'id', 'web_name', 'element_type', 'team', 'now_cost', 'selected_by_percent',
        'form', 'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'saves', 'bonus', 'expected_goals', 'expected_assists', 'points_per_game',
        'goals_per_90', 'assists_per_90', 'xG_per_90', 'xA_per_90', 'clean_sheets_per_game',
        'saves_per_90', 'bonus_per_game', 'form_numeric', 'fixture_difficulty_next_4',
        'fixture_count_next_4', 'historical_points_per_million', 'form_per_million', 'minutes_per_game',
        'is_nailed', 'rotation_risk', 'position', 'attacking_returns', 'defensive_returns',
        'is_premium', 'data_timestamp'
    ]

    if data_processor.config.get('feature_engineering', {}).get('enable_lagged_features'):
        expected_columns.extend(['total_points_lag_1', 'total_points_rolling_mean_3'])

    if data_processor.config.get('feature_engineering', {}).get('enable_fixture_trends'):
        expected_columns.extend(['fixture_next_1_difficulty', 'fixture_difficulty_trend'])

    if data_processor.config.get('feature_engineering', {}).get('enable_team_rolling_features'):
        expected_columns.extend(['team_goals_scored_rolling_mean_3'])
    
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert not df[numeric_cols].isnull().any().any(), "Numeric columns should not contain NaNs"

def test_process_all_data_empty_players(data_processor, sample_fixtures_data):
    """Test with empty players data."""
    # Test with empty elements list, which is a valid case
    result = data_processor.process_all_data({'elements': []}, sample_fixtures_data)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    # Test with empty elements list
    result = data_processor.process_all_data({'elements': []}, sample_fixtures_data)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_process_all_data_missing_fixtures(data_processor, sample_players_data):
    """Test with missing or empty fixtures data."""
    # Test with None
    result = data_processor.process_all_data(sample_players_data, None)
    assert not result.empty
    assert 'fixture_difficulty_next_4' in result.columns
    
    # Test with empty DataFrame
    result = data_processor.process_all_data(sample_players_data, pd.DataFrame())
    assert not result.empty
    assert 'fixture_difficulty_next_4' in result.columns

def test_clean_player_data(data_processor, sample_players_data):
    """Test the player data cleaning method."""
    df = pd.json_normalize(sample_players_data['elements'])
    # Store original now_cost values before cleaning
    original_now_cost = df['now_cost'].copy()
    cleaned = data_processor._clean_player_data(df)
    
    # Check that required columns are present
    required_columns = ['id', 'web_name', 'element_type', 'team', 'now_cost']
    for col in required_columns:
        assert col in cleaned.columns
    
    # Check that now_cost was divided by 10
    expected_now_cost = original_now_cost / 10.0
    assert (cleaned['now_cost'] == expected_now_cost).all()
    
    # Check that selected_by_percent is numeric
    assert pd.api.types.is_numeric_dtype(cleaned['selected_by_percent'])
    
    # Check that players with no team are removed
    df_with_invalid_team = df.copy()
    df_with_invalid_team.loc[0, 'team'] = 0  # Invalid team
    cleaned = data_processor._clean_player_data(df_with_invalid_team)
    assert 0 not in cleaned['team'].unique()
    assert len(cleaned) == len(df) - 1

def test_engineer_form_features(data_processor, sample_players_data):
    """Test the form feature engineering."""
    df = pd.json_normalize(sample_players_data['elements'])
    df = data_processor._engineer_form_features(df)
    
    # Check that form_numeric was created and is numeric
    assert 'form_numeric' in df.columns
    assert pd.api.types.is_numeric_dtype(df['form_numeric'])
    
    # Check that per-90 stats were calculated correctly
    for idx, row in df.iterrows():
        if row['minutes'] > 0:
            assert abs(row['points_per_game'] - (row['total_points'] / (row['minutes'] / 90))) < 0.01
            assert abs(row['goals_per_90'] - (row['goals_scored'] * 90 / row['minutes'])) < 0.01
            assert abs(row['assists_per_90'] - (row['assists'] * 90 / row['minutes'])) < 0.01

def test_engineer_fixture_features_with_trends(data_processor, sample_players_data, sample_fixtures_data):
    """Test the fixture feature engineering with trend analysis."""
    df = pd.json_normalize(sample_players_data['elements'])
    data_processor.config['feature_engineering'] = {'enable_fixture_trends': True}
    
    # Mock current gameweek as 0 to look at GWs 1, 2, 3, 4
    with patch.object(data_processor, '_get_current_gameweek_from_fixtures', return_value=0):
        result = data_processor._engineer_fixture_features(df, sample_fixtures_data, [])

    assert 'fixture_difficulty_next_4' in result.columns
    assert 'fixture_next_1_difficulty' in result.columns
    assert 'fixture_difficulty_trend' in result.columns

    # Team 1 has fixtures with difficulty [3, 3, 2, 3]
    team_1_features = result[result['team'] == 1]
    assert np.isclose(team_1_features['fixture_difficulty_next_4'].iloc[0], 2.75)
    assert np.isclose(team_1_features['fixture_next_1_difficulty'].iloc[0], 3)
    assert np.isclose(team_1_features['fixture_next_2_difficulty'].iloc[0], 3)
    # Trend of [3, 3, 2, 3] should have a negative slope
    assert team_1_features['fixture_difficulty_trend'].iloc[0] < 0


def test_engineer_team_rolling_features(data_processor, sample_historical_players_data):
    """Test team-level rolling aggregate features."""
    data_processor.config['feature_engineering'] = {'enable_team_rolling_features': True}

    result = data_processor._engineer_team_features(sample_historical_players_data, pd.DataFrame(), [])

    assert 'team_goals_scored_rolling_mean_3' in result.columns
    assert 'team_total_points_rolling_std_5' in result.columns

    # Check values for Team 1, Player 1 at Gameweek 3
    player_1_gw3 = result[(result['id'] == 1) & (result['gameweek'] == 3)]
    # GW1 team goals: 0 (p1) + 0 (p3) = 0. GW2 team goals: 1 (p1) + 0 (p3) = 1.
    # Rolling mean at GW3 should be mean of GW1 and GW2 team goals: (0+1)/2 = 0.5
    assert np.isclose(player_1_gw3['team_goals_scored_rolling_mean_3'].iloc[0], 0.5)

def test_engineer_value_features(data_processor, sample_players_data):
    """Test the value feature engineering."""
    df = pd.json_normalize(sample_players_data['elements'])
    df['now_cost'] = df['now_cost'] / 10  # Clean data first
    
    result = data_processor._engineer_value_features(df)
    
    # Check that value features were created
    for col in ['points_per_million', 'form_per_million', 'value_rank']:
        assert col in result.columns
    
    # Check that value_rank was calculated correctly
    for pos in result['element_type'].unique():
        pos_df = result[result['element_type'] == pos]
        assert pos_df['value_rank'].min() == 1
        assert pos_df['value_rank'].max() <= len(pos_df)

def test_engineer_premium_features(data_processor, sample_players_data):
    """Test the premium player identification."""
    df = pd.json_normalize(sample_players_data['elements'])
    result = data_processor._engineer_premium_features(df)
    
    # Check that is_premium was added
    assert 'is_premium' in result.columns
    
    # Check that premium thresholds are applied correctly
    thresholds = {
        1: 5.5,  # GK
        2: 6.0,  # DEF
        3: 8.0,  # MID
        4: 8.0   # FWD
    }
    
    for pos, threshold in thresholds.items():
        pos_df = result[result['element_type'] == pos]
        if not pos_df.empty:
            # Players at or above threshold should be marked as premium
            assert all(pos_df[pos_df['now_cost'] >= threshold]['is_premium'] == True)
            # Players below threshold should not be marked as premium
            assert all(pos_df[pos_df['now_cost'] < threshold]['is_premium'] == False)

def test_edge_cases(data_processor, sample_players_data):
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    
    # Create a minimal valid empty DataFrame with required columns
    required_columns = [
        'id', 'web_name', 'element_type', 'team', 'now_cost', 'form', 
        'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'saves', 'bonus', 'expected_goals', 'expected_assists'
    ]
    
    empty_df_with_cols = pd.DataFrame(columns=required_columns)
    
    # Test methods that can handle completely empty DataFrames
    for method in [
        data_processor._engineer_fixture_features,
    ]:
        if method == data_processor._engineer_fixture_features:
            result = method(empty_df, pd.DataFrame())
        else:
            result = method(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    # Test methods that require certain columns
    for method in [
        (data_processor._engineer_form_features, False),
        (data_processor._engineer_value_features, False),
        (data_processor._engineer_minutes_features, False),
        (data_processor._engineer_team_features, True),  # Requires teams_df
        (data_processor._engineer_position_features, False),
        (data_processor._engineer_premium_features, False)
    ]:
        # These should handle empty DataFrames with the right columns
        if method[1]:  # If method requires teams_df
            result = method[0](empty_df_with_cols, pd.DataFrame())
        else:
            result = method[0](empty_df_with_cols)
            
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    # Test with None input
    result = data_processor._clean_player_data(None)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    # Test with invalid data types
    with pytest.raises(AttributeError):
        data_processor._clean_player_data("not a dataframe")
    
    # Test with sample data that has missing required columns
    invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})
    with pytest.raises(ValueError):
        data_processor._clean_player_data(invalid_df)

def test_engineer_lagged_features(data_processor, sample_historical_players_data):
    """Test lagged and rolling window feature engineering."""
    result = data_processor._engineer_lagged_features(sample_historical_players_data, [])

    assert 'total_points_lag_1' in result.columns
    assert 'ict_index_rolling_mean_3' in result.columns

    # Check specific values for Player 1 at Gameweek 3
    player_1_gw3 = result[(result['id'] == 1) & (result['gameweek'] == 3)]
    assert player_1_gw3['total_points_lag_1'].iloc[0] == 5  # From GW2
    assert player_1_gw3['total_points_lag_2'].iloc[0] == 2  # From GW1

    # Rolling mean of points at GW3 is mean of GW1, GW2: (2+5)/2 = 3.5
    assert np.isclose(player_1_gw3['total_points_rolling_mean_3'].iloc[0], 3.5)

    # Momentum: 5 (GW2) - 2 (GW1) = 3
    assert player_1_gw3['points_delta_1_2'].iloc[0] == 3

    # Check that NaNs are filled
    assert not result.isnull().any().any()
