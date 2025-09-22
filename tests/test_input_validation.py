"""
Unit tests for input validation.
Tests validation of user inputs, API responses, and model outputs.
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
from models.ml_predictor import MLPredictor
from utils.config import Config

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()

class TestUserInputValidation:
    """Tests for validating user inputs."""
    
    def test_validate_team_selection(self, config):
        """Test validation of user's team selection."""
        # Valid team selection
        valid_team = {
            'starting_xi': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'bench': [12, 13, 14, 15],
            'captain': 1,
            'vice_captain': 2
        }
        
        # Should not raise an exception
        DataProcessor(config)._validate_team_structure(valid_team)
        
        # Test invalid structures
        invalid_teams = [
            {'starting_xi': list(range(1, 12)), 'bench': list(range(12, 15))},  # Missing captain
            {'starting_xi': list(range(1, 10)), 'bench': list(range(10, 15)), 'captain': 1},  # Not enough starters
            {'starting_xi': list(range(1, 12)), 'bench': list(range(12, 16)), 'captain': 1}  # Too many bench players
        ]
        
        for team in invalid_teams:
            with pytest.raises(ValueError):
                DataProcessor(config)._validate_team_structure(team)
    
    def test_validate_transfer_inputs(self, config):
        """Test validation of transfer inputs."""
        # Valid transfer
        valid_transfer = {
            'player_out': 1,
            'player_in': 2,
            'is_free_hit': False,
            'is_wildcard': False
        }
        
        # Should not raise an exception
        DataProcessor(config)._validate_transfer(valid_transfer)
        
        # Test invalid transfers
        invalid_transfers = [
            {'player_out': 1},  # Missing player_in
            {'player_in': 2},   # Missing player_out
            {'player_out': 1, 'player_in': 1}  # Same player in and out
        ]
        
        for transfer in invalid_transfers:
            with pytest.raises(ValueError):
                DataProcessor(config)._validate_transfer(transfer)

class TestAPIResponseValidation:
    """Tests for validating API responses."""
    
    def test_validate_bootstrap_static(self):
        """Test validation of bootstrap-static API response."""
        # Valid response
        valid_response = {
            'elements': [{'id': 1, 'web_name': 'Player 1', 'element_type': 3, 'team': 1, 'now_cost': 50}],
            'teams': [{'id': 1, 'name': 'Team 1'}],
            'events': [{'id': 1, 'name': 'Gameweek 1'}],
            'game_settings': {}
        }
        
        # Should not raise an exception
        DataProcessor._validate_bootstrap_static(valid_response)
        
        # Test invalid responses
        invalid_responses = [
            {},  # Empty response
            {'elements': []},  # Missing required sections
            {'elements': [{}], 'teams': [{}], 'events': [{}], 'game_settings': {}}  # Missing required fields
        ]
        
        for response in invalid_responses:
            with pytest.raises(ValueError):
                DataProcessor._validate_bootstrap_static(response)
    
    def test_validate_fixtures_response(self):
        """Test validation of fixtures API response."""
        # Valid response
        valid_response = [
            {'id': 1, 'event': 1, 'team_h': 1, 'team_a': 2, 'team_h_difficulty': 3, 'team_a_difficulty': 4}
        ]
        
        # Should not raise an exception
        DataProcessor._validate_fixtures_response(valid_response)
        
        # Test invalid responses
        invalid_responses = [
            [],  # Empty list
            [{'id': 1}],  # Missing required fields
            [{'id': 1, 'event': 1, 'team_h': 1, 'team_a': 2}]  # Missing difficulty fields
        ]
        
        for response in invalid_responses:
            with pytest.raises(ValueError):
                DataProcessor._validate_fixtures_response(response)

class TestModelOutputValidation:
    """Tests for validating model outputs."""
    
    def test_validate_prediction_output(self, config):
        """Test validation of model prediction outputs."""
        ml_predictor = MLPredictor(config)
        
        # Valid prediction output
        valid_output = pd.DataFrame({
            'id': [1, 2, 3],
            'predicted_points': [5.0, 6.0, 7.0],
            'confidence_interval_low': [4.0, 5.0, 6.0],
            'confidence_interval_high': [6.0, 7.0, 8.0]
        })
        
        # Should not raise an exception
        ml_predictor._validate_prediction_output(valid_output)
        
        # Test invalid outputs
        invalid_outputs = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'id': [1, 2, 3]}),  # Missing prediction columns
            pd.DataFrame({'id': [1, 2, 3], 'predicted_points': [5.0, 6.0, np.nan]}),  # NaN in predictions
            pd.DataFrame({'id': [1, 2, 3], 'predicted_points': [5.0, 6.0, -1.0]})  # Negative predictions
        ]
        
        for output in invalid_outputs:
            with pytest.raises(ValueError):
                ml_predictor._validate_prediction_output(output)
    
    def test_validate_model_weights(self, config):
        """Test validation of model ensemble weights."""
        ml_predictor = MLPredictor(config)
        
        # Valid weights
        valid_weights = {
            'xgb': 0.6,
            'lgb': 0.4
        }
        
        # Should not raise an exception
        ml_predictor._validate_model_weights(valid_weights, ['xgb', 'lgb'])
        
        # Test invalid weights
        invalid_weights = [
            {'xgb': 0.5, 'lgb': 0.4},  # Doesn't sum to 1
            {'xgb': 1.1, 'lgb': -0.1},  # Weights outside [0,1] range
            {'xgb': 1.0},  # Missing model
            {}  # Empty weights
        ]
        
        for weights in invalid_weights:
            with pytest.raises(ValueError):
                ml_predictor._validate_model_weights(weights, ['xgb', 'lgb'])

class TestDataValidation:
    """Tests for data validation utilities."""
    
    def test_validate_player_data(self):
        """Test validation of player data structure."""
        # Valid player data
        valid_player = {
            'id': 1,
            'web_name': 'Player 1',
            'element_type': 1,
            'team': 1,
            'now_cost': 50,
            'selected_by_percent': '5.0',
            'form': '4.5',
            'total_points': 45,
            'minutes': 1800
        }
        
        # Should not raise an exception
        DataProcessor._validate_player_data(valid_player)
        
        # Test invalid player data
        invalid_players = [
            {},  # Empty
            {'id': 1},  # Missing required fields
            {**valid_player, 'now_cost': -10},  # Invalid cost
            {**valid_player, 'element_type': 5}  # Invalid position
        ]
        
        for player in invalid_players:
            with pytest.raises(ValueError):
                DataProcessor._validate_player_data(player)
    
    def test_validate_fixture_difficulty(self):
        """Test validation of fixture difficulty values."""
        # Valid difficulties
        valid_difficulties = [1, 2, 3, 4, 5]
        
        for diff in valid_difficulties:
            DataProcessor._validate_fixture_difficulty(diff)
        
        # Test invalid difficulties
        invalid_difficulties = [0, 6, -1, 1.5]
        
        for diff in invalid_difficulties:
            with pytest.raises(ValueError):
                DataProcessor._validate_fixture_difficulty(diff)

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_handle_missing_data(self, config):
        """Test handling of missing or malformed data."""
        processor = DataProcessor(config)
        
        # Test with None input
        with pytest.raises(ValueError):
            processor.process_all_data(None, None)
        
        # Test with empty data
        with pytest.raises(ValueError):
            processor.process_all_data({}, pd.DataFrame())
    
    def test_handle_invalid_types(self, config):
        """Test handling of invalid data types."""
        ml_predictor = MLPredictor(config)
        
        # Test with invalid input types
        invalid_inputs = [
            None,
            "not a dataframe",
            123,
            {'a': 1, 'b': 2}
        ]
        
        for invalid in invalid_inputs:
            with pytest.raises((ValueError, AttributeError)):
                ml_predictor.predict_player_points(invalid)
    
    def test_handle_extreme_values(self, config):
        """Test handling of extreme or edge case values."""
        # Test with extreme values in the data
        extreme_data = pd.DataFrame({
            'id': [1],
            'web_name': ['Test Player'],
            'element_type': [1],
            'team': [1],
            'now_cost': [1e6],  # Extremely high cost
            'selected_by_percent': ['1000.0'],  # Invalid percentage
            'form': ['999.9'],  # Extremely high form
            'total_points': [1e6],  # Extremely high points
            'minutes': [0],  # Zero minutes
            'goals_scored': [1e6],
            'assists': [1e6],
            'clean_sheets': [1e6],
            'saves': [1e6],
            'bonus': [1e6],
            'expected_goals': ['999.9'],
            'expected_assists': ['999.9']
        })
        
        # This should not raise an exception
        processor = DataProcessor(config)
        result = processor._clean_player_data(extreme_data)
        
        # Check that values were clamped to reasonable ranges
        assert result['now_cost'].iloc[0] < 1e6
        assert 0 <= result['selected_by_percent'].iloc[0] <= 100
        assert 0 <= result['form'].iloc[0] <= 15  # Form is capped at 15 in _clean_player_data
