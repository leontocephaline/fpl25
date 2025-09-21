"""
Test suite for memorization_diagnostics module.

This file contains unit, integration, and analytical tests to validate the
correctness and robustness of the memorization diagnostics implementation.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import math
import warnings

from src.analysis.memorization_diagnostics import (
    calculate_squared_error_memorization_score,
    calculate_bits_of_memorization,
    rolling_mean_predictor,
    generate_memorization_report
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data
def create_test_data(
    n_players: int = 5, 
    n_gws: int = 10,
    noise_std: float = 0.1,
    trend_strength: float = 0.5,
    seed: int = 42,
    include_position: bool = False
) -> pd.DataFrame:
    """Create synthetic test data with known properties.
    
    Args:
        n_players: Number of players to generate data for
        n_gws: Number of gameweeks per player
        noise_std: Standard deviation of noise to add to target
        trend_strength: Strength of the linear trend in the target
        seed: Random seed for reproducibility
        include_position: Whether to include a position column
    """
    np.random.seed(seed)
    
    # Create player IDs and gameweeks
    player_ids = [f"player_{i}" for i in range(n_players)]
    gws = list(range(1, n_gws + 1))
    
    # Generate data for each player
    data = []
    for i, player_id in enumerate(player_ids):
        # Base trend increases over time
        base_trend = np.linspace(0, trend_strength * (i + 1), n_gws)
        
        # Add some noise
        noise = np.random.normal(0, noise_std, n_gws)
        
        # Create target with trend and noise
        y_true = base_trend + noise
        
        # Create features (all numeric for model training)
        feature1 = np.random.normal(i, 1, n_gws)  # Player-specific feature
        feature2 = np.random.normal(0, 1, n_gws)  # Common feature
        
        # Add to data
        for j in range(n_gws):
            row = {
                'player_id': player_id,
                'gw': gws[j],
                'y_true': y_true[j],
                'feature1': feature1[j],
                'feature2': feature2[j]
            }
            
            # Add position if requested (for testing position-specific functionality)
            if include_position:
                row['position'] = 'FWD' if i % 3 == 0 else 'MID' if i % 3 == 1 else 'DEF'
                
            data.append(row)
    
    return pd.DataFrame(data)

# Fixtures
@pytest.fixture
def test_data():
    """Create test data fixture."""
    return create_test_data(n_players=3, n_gws=5, include_position=True)

@pytest.fixture
def trained_model():
    """Fixture that returns a trained linear regression model."""
    model = LinearRegression()
    X = pd.DataFrame(np.random.randn(100, 2), columns=['feature1', 'feature2'])
    y = 2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(100) * 0.1
    model.fit(X, y)
    return model

# Unit Tests
def test_rolling_mean_predictor(test_data):
    """Test rolling mean baseline predictor."""
    # Test with window=2
    preds = rolling_mean_predictor(
        test_data, 
        target_col='y_true', 
        window=2,
        group_col='player_id',
        time_col='gw'
    )
    
    # Check shape
    assert len(preds) == len(test_data)
    
    # Check first prediction is global mean (no history)
    global_mean = test_data['y_true'].mean()
    assert np.isclose(preds[0], global_mean, rtol=1e-5)
    
    # Check subsequent predictions use rolling mean
    for i, (_, row) in enumerate(test_data.iterrows()):
        if i == 0:
            continue  # Already checked first prediction
            
        # Get previous data for this player
        player_data = test_data[test_data['player_id'] == row['player_id']]
        past_data = player_data[player_data['gw'] < row['gw']]
        
        if len(past_data) == 0:
            # First game for player, should use global mean
            assert np.isclose(preds[i], global_mean, rtol=1e-5)
        else:
            # Should use mean of past 2 games, or all past games if fewer than 2
            if len(past_data) >= 2:
                expected = past_data['y_true'].iloc[-2:].mean()
            else:
                expected = past_data['y_true'].mean()
            
            assert np.isclose(preds[i], expected, rtol=1e-5, equal_nan=True)

def test_squared_error_memorization_score_basic(trained_model, test_data):
    """Test basic functionality of squared error memorization score."""
    # Calculate scores
    df, aggs = calculate_squared_error_memorization_score(
        model=trained_model,
        train_df=test_data,
        feature_cols=['feature1', 'feature2'],  # Explicitly specify numeric features
        return_df=True
    )
    
    # Check outputs
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(test_data)
    assert 'memorization_score' in df.columns
    assert 'y_true' in df.columns
    
    # Check aggregates
    required_aggs = ['mean', 'median', 'p95', 'frac_positive']
    for agg in required_aggs:
        assert agg in aggs
    
    # Check scores are in expected range
    assert (df['memorization_score'] >= 0).all()  # Should be non-negative

def test_bits_of_memorization_homoscedastic(trained_model, test_data):
    """Test bits of memorization with homoscedastic variance."""
    # Calculate bits
    df, aggs = calculate_bits_of_memorization(
        model=trained_model,
        train_df=test_data,
        feature_cols=['feature1', 'feature2'],  # Explicitly specify numeric features
        variance_estimator="homoscedastic",
        return_df=True
    )
    
    # Check outputs
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(test_data)
    assert 'bits_memorized' in df.columns
    
    # Check aggregates
    required_aggs = ['mean_bits_per_sample', 'total_bits', 'frac_positive']
    for agg in required_aggs:
        assert agg in aggs

def test_variance_estimators(trained_model, test_data):
    """Test different variance estimators."""
    # Test homoscedastic
    _, aggs_homo = calculate_bits_of_memorization(
        model=trained_model, 
        train_df=test_data, 
        feature_cols=['feature1', 'feature2'],
        variance_estimator="homoscedastic"
    )
    
    # Test heteroscedastic
    _, aggs_hetero = calculate_bits_of_memorization(
        model=trained_model, 
        train_df=test_data,
        feature_cols=['feature1', 'feature2'],
        variance_estimator="heteroscedastic"
    )
    
    # Should get different results
    assert not np.isclose(aggs_homo['mean_bits_per_sample'], 
                         aggs_hetero['mean_bits_per_sample'])

def test_missing_data_handling(trained_model, test_data):
    """Test handling of missing data."""
    # Create a clean copy with no missing values
    clean_data = test_data.dropna().copy()
    
    # Test with missing feature value - should drop that row
    data_missing_feature = clean_data.copy()
    row_with_nan = 1
    data_missing_feature.loc[row_with_nan, 'feature1'] = np.nan
    
    # Should drop the row with NaN feature
    df, _ = calculate_squared_error_memorization_score(
        model=trained_model,
        train_df=data_missing_feature,
        feature_cols=['feature1', 'feature2'],
        return_df=True
    )
    
    # Should have one less row than input (the one with missing feature)
    assert len(df) == len(clean_data) - 1
    
    # Test with missing target value - should drop that row
    data_missing_target = clean_data.copy()
    row_with_nan = 0
    data_missing_target.loc[row_with_nan, 'y_true'] = np.nan
    
    df2, _ = calculate_squared_error_memorization_score(
        model=trained_model,
        train_df=data_missing_target,
        feature_cols=['feature1', 'feature2'],
        return_df=True
    )
    
    # Should have one less row than input (the one with missing target)
    assert len(df2) == len(clean_data) - 1

# Integration Tests
def test_end_to_end_workflow(trained_model, test_data):
    """Test full workflow from calculation to report generation."""
    # Calculate both metrics
    se_df, se_aggs = calculate_squared_error_memorization_score(
        model=trained_model,
        train_df=test_data,
        feature_cols=['feature1', 'feature2'],
        return_df=True
    )
    
    bits_df, bits_aggs = calculate_bits_of_memorization(
        model=trained_model,
        train_df=test_data,
        feature_cols=['feature1', 'feature2'],
        return_df=True
    )
    
    # Generate report
    report = generate_memorization_report(
        results_list=[(se_df, se_aggs), (bits_df, bits_aggs)],
        metric_names=["Squared Error", "Bits of Memorization"]
    )
    
    # Basic checks on report
    assert isinstance(report, str)
    assert "Memorization Analysis Report" in report
    assert "Squared Error" in report
    assert "Bits of Memorization" in report

# Edge Case Tests
def test_empty_input(trained_model):
    """Test with empty input DataFrame."""
    empty_df = pd.DataFrame(columns=['player_id', 'gw', 'y_true', 'feature'])
    
    with pytest.raises(ValueError):
        calculate_squared_error_memorization_score(trained_model, empty_df)

def test_single_sample(trained_model, test_data):
    """Test with a single sample.
    
    Some metrics like variance can't be meaningfully calculated with a single sample,
    so we just verify that the functions handle this case gracefully without errors.
    """
    single_sample = test_data.iloc[[0]]
    
    # Test squared error memorization score
    df, aggs = calculate_squared_error_memorization_score(
        model=trained_model,
        train_df=single_sample,
        feature_cols=['feature1', 'feature2'],
        return_df=True
    )
    
    # Basic output validation
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert 'memorization_score' in df.columns
    
    # Test bits of memorization with single sample
    with warnings.catch_warnings():
        # Ignore expected warnings about insufficient data
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        
        df_bits, aggs_bits = calculate_bits_of_memorization(
            model=trained_model,
            train_df=single_sample,
            feature_cols=['feature1', 'feature2'],
            variance_estimator="homoscedastic"
        )
    
    # Basic output validation
    assert isinstance(df_bits, pd.DataFrame)
    assert len(df_bits) == 1
    assert 'bits_memorized' in df_bits.columns
    
    # For a single sample, we can't calculate variance-based metrics meaningfully
    # So we'll just check that the required columns exist and the values are as expected
    required_columns = ['y_true', 'y_pred_model', 'y_pred_baseline', 'bits_memorized']
    for col in required_columns:
        assert col in df_bits.columns
        
        # For bits_memorized, it's expected to be NaN for a single sample
        if col == 'bits_memorized':
            assert df_bits[col].isna().all(), "bits_memorized should be NaN for a single sample"
        else:
            # Other columns should have finite values
            assert np.isfinite(df_bits[col]).all()

# Run tests
if __name__ == "__main__":
    pytest.main(["-v", "test_memorization_diagnostics.py"])
