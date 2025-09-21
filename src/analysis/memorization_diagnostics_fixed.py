"""
Memorization diagnostics for time-series regression models.

This module provides functions to quantify and analyze model memorization vs. generalization
in time-series regression tasks, specifically designed for FPL prediction models.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Type aliases
ModelType = Union[xgb.XGBRegressor, BaseEstimator]
BaselinePredictor = Union[str, Callable[[pd.DataFrame], np.ndarray]]
VarianceEstimator = Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]

def rolling_mean_predictor(df: pd.DataFrame, target_col: str = 'y_true', 
                         window: int = 5, min_periods: int = 1,
                         group_col: str = 'player_id',
                         time_col: str = 'gw',
                         return_fallback_info: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Baseline predictor using rolling mean of target values within player groups.
    
    Args:
        df: DataFrame containing the time series data
        target_col: Name of the target column
        window: Size of the rolling window
        min_periods: Minimum number of observations in window required
        group_col: Column name for player/group identifier
        time_col: Column name for time ordering
        return_fallback_info: If True, returns a tuple of (predictions, fallback_info)
        
    Returns:
        If return_fallback_info is False, returns array of predictions.
        If return_fallback_info is True, returns tuple of (predictions, fallback_info)
        where fallback_info is a dict with the following keys:
        - 'global_mean_fallback_used': array of bool indicating if global mean was used
        - 'global_mean': the global mean value used as fallback
        - 'fallback_pct': percentage of predictions that used the global mean
        
    Note:
        - For players with insufficient history, falls back to global mean
        - Handles missing values by forward filling within groups
    """
    # Input validation
    required_cols = [group_col, time_col, target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure the data is sorted by time within each group
    df = df.sort_values(by=[group_col, time_col])
    
    # Calculate global mean as fallback
    global_mean = df[target_col].mean()
    
    # Initialize a Series to track fallback usage
    fallback_used = pd.Series(False, index=df.index)
    
    # Define a function to calculate rolling mean with fallback tracking
    def safe_rolling_mean(series, window_size, min_periods):
        nonlocal fallback_used
        if window_size < 1:
            window_size = 1
        min_periods = max(1, min(min_periods, window_size))
        
        try:
            rolling = series.rolling(window=window_size, min_periods=min_periods)
            result = rolling.mean().shift(1)
            
            # Track where we're using the fallback (NaN values in the result)
            fallback_mask = result.isna() & (series.notna().cumsum() > 0)
            fallback_used.loc[fallback_mask.index[fallback_mask]] = True
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in rolling mean calculation: {str(e)}. Using global mean as fallback.")
            fallback_used[series.index] = True
            return pd.Series(np.nan, index=series.index)
    
    # Calculate rolling means within each group
    rolling_means = df.groupby(group_col, group_keys=False)[target_col].apply(
        lambda x: safe_rolling_mean(x, window, min_periods)
    )
    
    # Fill missing values with global mean
    filled_means = rolling_means.fillna(global_mean)
    
    # Calculate fallback statistics
    fallback_pct = fallback_used.mean() * 100
    if fallback_pct > 0:
        logger.warning(f"Used global mean fallback for {fallback_pct:.1f}% of predictions")
    
    # Ensure the output is in the same order as the input
    filled_means = filled_means.reindex(df.index)
    fallback_used = fallback_used.reindex(df.index)
    
    if return_fallback_info:
        fallback_info = {
            'global_mean_fallback_used': fallback_used.values,
            'global_mean': global_mean,
            'fallback_pct': fallback_pct
        }
        return filled_means.values, fallback_info
    
    return filled_means.values

def calculate_squared_error_memorization_score(
    model: ModelType,
    train_df: pd.DataFrame,
    baseline_predictor: Union[BaselinePredictor, str] = "rolling_mean",
    target_col: str = 'y_true',
    group_col: str = 'player_id',
    time_col: str = 'gw',
    position_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    return_df: bool = True,
    fold_id: Optional[Any] = None
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Calculate memorization scores based on squared error differences.
    
    Args:
        model: Trained regression model (XGBoost or scikit-learn compatible)
        train_df: DataFrame containing training data with features and target
        baseline_predictor: Either a callable or string identifier for baseline prediction method.
            Built-in options: "rolling_mean" (default)
        target_col: Name of the target column
        group_col: Name of the column identifying different time series (e.g., player_id)
        time_col: Name of the time column (must be sortable)
        position_col: Optional column name for player position (used in reporting)
        feature_cols: List of feature column names. If None, all numeric columns except 
                     target/group/time/position are used
        return_df: Whether to return per-sample results
        fold_id: Optional identifier for the current fold/iteration (for logging)
        
    Returns:
        Tuple of (per_sample_df, aggregates_dict) if return_df is True,
        otherwise (None, aggregates_dict)
        
    Raises:
        ValueError: If required columns are missing or data validation fails
        
    Notes:
        - Handles missing values by falling back to global statistics
        - Logs warnings for potential issues during processing
        - Negative memorization scores indicate where the baseline outperforms the model
    """
    # Input validation
    required_cols = [group_col, time_col, target_col]
    if position_col:
        required_cols.append(position_col)
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying the original
    df = train_df.copy()
    
    # Set default feature columns if not provided
    if feature_cols is None:
        # Exclude non-feature columns and non-numeric columns
        non_feature_cols = {group_col, time_col, target_col}
        if position_col:
            non_feature_cols.add(position_col)
        
        # Select only numeric columns for features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
        
        logger.info(f"Using {len(feature_cols)} numeric features: {', '.join(feature_cols[:5])}...")
    
    # Ensure all feature columns exist in the dataframe
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")
    
    # Ensure feature columns are numeric
    non_numeric_features = [col for col in feature_cols if not is_numeric_dtype(df[col])]
    if non_numeric_features:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric_features}")
    
    # Sort data by group and time
    df = df.sort_values(by=[group_col, time_col])
    
    # Get model predictions
    X = df[feature_cols]
    y_true = df[target_col].values
    
    try:
        y_pred_model = model.predict(X)
    except Exception as e:
        logger.error(f"Error generating model predictions: {str(e)}")
        raise
    
    # Get baseline predictions
    if isinstance(baseline_predictor, str):
        if baseline_predictor == "rolling_mean":
            # Get baseline predictions with fallback info
            y_pred_baseline, fallback_info = rolling_mean_predictor(
                df, 
                target_col=target_col,
                group_col=group_col,
                time_col=time_col,
                return_fallback_info=True
            )
            
            # Log fallback statistics
            fallback_pct = fallback_info['fallback_pct']
            if fallback_pct > 0:
                logger.warning(f"Used global mean fallback for {fallback_pct:.1f}% of baseline predictions")
        else:
            raise ValueError(f"Unknown baseline predictor: {baseline_predictor}")
    else:
        # Custom baseline predictor function
        y_pred_baseline = baseline_predictor(df)
    
    # Calculate squared errors
    se_model = (y_true - y_pred_model) ** 2
    se_baseline = (y_true - y_pred_baseline) ** 2
    
    # Calculate memorization scores (can be negative if model is worse than baseline)
    mem_scores = se_baseline - se_model
    
    # Calculate aggregates
    mean_mem_score = np.mean(mem_scores)
    sum_mem_score = np.sum(mem_scores)
    p95_mem_score = np.percentile(mem_scores, 95)
    
    # Calculate positive and negative memorization statistics
    pos_mask = mem_scores > 0
    neg_mask = mem_scores < 0
    
    frac_positive = np.mean(pos_mask) * 100
    frac_negative = np.mean(neg_mask) * 100
    
    mean_positive_mem = np.mean(mem_scores[pos_mask]) if np.any(pos_mask) else 0
    mean_negative_mem = np.mean(mem_scores[neg_mask]) if np.any(neg_mask) else 0
    
    # Prepare aggregates dictionary
    aggregates = {
        'mean_mem_score': float(mean_mem_score),
        'sum_mem_score': float(sum_mem_score),
        'p95_mem_score': float(p95_mem_score),
        'frac_positive': float(frac_positive) / 100,  # Convert back to fraction
        'frac_negative': float(frac_negative) / 100,  # Convert back to fraction
        'mean_positive_mem': float(mean_positive_mem),
        'mean_negative_mem': float(mean_negative_mem),
        'n_samples': len(df)
    }
    
    # Add position-specific aggregates if position_col is provided
    if position_col and position_col in df.columns:
        for pos in df[position_col].unique():
            pos_mask = df[position_col] == pos
            if pos_mask.sum() > 0:
                pos_scores = mem_scores[pos_mask]
                pos_pos_mask = pos_scores > 0
                pos_neg_mask = pos_scores < 0
                
                aggregates[f'frac_positive_{pos}'] = float(np.mean(pos_pos_mask)) if len(pos_scores) > 0 else 0.0
                aggregates[f'frac_negative_{pos}'] = float(np.mean(pos_neg_mask)) if len(pos_scores) > 0 else 0.0
                aggregates[f'mean_positive_mem_{pos}'] = float(np.mean(pos_scores[pos_pos_mask])) if np.any(pos_pos_mask) else 0.0
                aggregates[f'mean_negative_mem_{pos}'] = float(np.mean(pos_scores[pos_neg_mask])) if np.any(pos_neg_mask) else 0.0
                aggregates[f'n_samples_{pos}'] = int(pos_mask.sum())
    
    # Add fold ID to aggregates if provided
    if fold_id is not None:
        aggregates['fold_id'] = fold_id
    
    # Prepare per-sample results if requested
    if return_df:
        result_data = {
            group_col: df[group_col].values,
            time_col: df[time_col].values,
            'y_true': y_true,
            'y_pred_model': y_pred_model,
            'y_pred_baseline': y_pred_baseline,
            'squared_error_model': se_model,
            'squared_error_baseline': se_baseline,
            'memorization_score': mem_scores,
            'is_positive_memorization': pos_mask,
            'is_negative_memorization': neg_mask
        }
        
        # Add position if available
        if position_col and position_col in df.columns:
            result_data[position_col] = df[position_col].values
        
        # Add fallback info if available
        if 'fallback_info' in locals():
            result_data['used_global_mean_fallback'] = fallback_info['global_mean_fallback_used']
        
        # Add timestamp
        result_data['timestamp'] = pd.Timestamp.now().isoformat()
        
        # Create the results DataFrame
        results_df = pd.DataFrame(result_data)
        
        return results_df, aggregates
    
    return None, aggregates

def calculate_bits_of_memorization(
    model: ModelType,
    train_df: pd.DataFrame,
    baseline_predictor: Union[BaselinePredictor, str] = "rolling_mean",
    variance_estimator: Union[str, Callable] = "heteroscedastic_rolling",
    target_col: str = 'y_true',
    group_col: str = 'player_id',
    time_col: str = 'gw',
    position_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    return_df: bool = True,
    fold_id: Optional[Any] = None,
    min_std_dev: float = 0.5,  # Increased from 1e-6 to prevent numerical instability
    max_std_dev: float = 10.0,  # New: Cap on standard deviation
    min_samples_for_std: int = 3,  # Minimum samples to estimate variance
    window_size: int = 10,
    clip_residuals: float = 10.0  # Clip residuals to prevent extreme values
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Calculate memorization in bits using negative log likelihood differences.
    
    Args:
        model: Trained regression model (XGBoost or scikit-learn compatible)
        train_df: DataFrame containing training data with features and target
        baseline_predictor: Either a callable or string identifier for baseline prediction
        variance_estimator: Method for estimating variance. Options:
            - 'homoscedastic': Single variance for all samples
            - 'rolling': Rolling window variance within each player's time series
            - 'heteroscedastic_rolling': Rolling window variance with fallback to global
            - Callable: Custom function that takes (residuals, predictions) and returns variance estimates
        target_col: Name of the target column
        group_col: Column name for player/group identifier
        time_col: Column name for time ordering
        position_col: Optional column name for player position (used in reporting)
        feature_cols: List of feature column names. If None, all numeric columns are used
        return_df: Whether to return per-sample results
        fold_id: Optional identifier for the current fold/iteration (for logging)
        min_std_dev: Minimum standard deviation to avoid division by zero
        window_size: Size of rolling window for variance estimation
        
    Returns:
        Tuple of (per_sample_df, aggregates_dict) if return_df is True,
        otherwise (None, aggregates_dict)
        
    Raises:
        ValueError: If required columns are missing or data validation fails
        
    Notes:
        - Handles missing values by falling back to global statistics
        - Logs warnings for potential issues during processing
        - Negative bits indicate where the baseline outperforms the model
    """
    # Input validation
    required_cols = [group_col, time_col, target_col]
    if position_col:
        required_cols.append(position_col)
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying the original
    df = train_df.copy()
    
    # Set default feature columns if not provided
    if feature_cols is None:
        # Exclude non-feature columns and non-numeric columns
        non_feature_cols = {group_col, time_col, target_col}
        if position_col:
            non_feature_cols.add(position_col)
        
        # Select only numeric columns for features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
        
        logger.info(f"Using {len(feature_cols)} numeric features: {', '.join(feature_cols[:5])}...")
    
    # Ensure all feature columns exist in the dataframe and are numeric
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")
    
    non_numeric_features = [col for col in feature_cols if not is_numeric_dtype(df[col])]
    if non_numeric_features:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric_features}")
    
    # Sort data by group and time
    df = df.sort_values(by=[group_col, time_col])
    
    # Get model predictions and residuals
    X = df[feature_cols]
    y_true = df[target_col].values
    
    try:
        y_pred_model = model.predict(X)
        residuals = y_true - y_pred_model
    except Exception as e:
        logger.error(f"Error generating model predictions: {str(e)}")
        raise
    
    # Get baseline predictions
    if isinstance(baseline_predictor, str):
        if baseline_predictor == "rolling_mean":
            # Get baseline predictions with fallback info
            y_pred_baseline, fallback_info = rolling_mean_predictor(
                df, 
                target_col=target_col,
                group_col=group_col,
                time_col=time_col,
                return_fallback_info=True
            )
            
            # Log fallback statistics
            fallback_pct = fallback_info['fallback_pct']
            if fallback_pct > 0:
                logger.warning(f"Used global mean fallback for {fallback_pct:.1f}% of baseline predictions")
        else:
            raise ValueError(f"Unknown baseline predictor: {baseline_predictor}")
    else:
        # Custom baseline predictor function
        y_pred_baseline = baseline_predictor(df)
    
    # Calculate baseline residuals
    baseline_residuals = y_true - y_pred_baseline
    
    # Clip extreme residuals to prevent numerical instability
    residuals = np.clip(residuals, -clip_residuals, clip_residuals)
    baseline_residuals = np.clip(baseline_residuals, -clip_residuals, clip_residuals)
    
    # Calculate variance estimates with enhanced stability
    sigma_model, sigma_baseline, var_info = _estimate_variances(
        residuals, 
        baseline_residuals, 
        y_pred_model, 
        y_pred_baseline, 
        df, 
        group_col, 
        time_col, 
        variance_estimator, 
        window_size, 
        min_std_dev,
        min_samples=min_samples_for_std
    )
    
    # Apply maximum standard deviation cap
    sigma_model = np.minimum(sigma_model, max_std_dev)
    sigma_baseline = np.minimum(sigma_baseline, max_std_dev)
    
    # Calculate negative log likelihoods with improved numerical stability
    def safe_nll(y_true, y_pred, sigma):
        # Clip values to prevent numerical issues
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        
        # Ensure sigma is within bounds
        sigma = np.clip(sigma, min_std_dev, max_std_dev)
        
        # Calculate squared error with clipping
        squared_error = np.clip((y_true - y_pred) ** 2, 0, 1e6)
        
        # Calculate log(2πσ²) = log(2π) + 2*log(σ)
        log_2pi = np.log(2 * np.pi)
        log_var = 2 * np.log(sigma)
        
        # Calculate NLL = 0.5 * (log(2πσ²) + (y-ŷ)²/σ²)
        nll = 0.5 * (log_2pi + log_var + squared_error / (sigma ** 2 + 1e-16))
        
        # Final clipping to prevent extreme values
        return np.clip(nll, -1e6, 1e6)
    
    # Calculate NLLs with error handling
    try:
        nll_model = safe_nll(y_true, y_pred_model, sigma_model)
        nll_baseline = safe_nll(y_true, y_pred_baseline, sigma_baseline)
        
        # Calculate bits of memorization with clipping
        nll_diff = np.clip(nll_baseline - nll_model, -100, 100)
        bits_of_memorization = nll_diff / np.log(2)
        
    except Exception as e:
        logger.error(f"Error in NLL calculation: {str(e)}")
        raise
    
    # Calculate robust statistics that handle edge cases
    def safe_stats(values):
        values = np.asarray(values)
        finite_mask = np.isfinite(values)
        if np.any(finite_mask):
            valid_values = values[finite_mask]
            return {
                'mean': float(np.mean(valid_values)),
                'median': float(np.median(valid_values)),
                'std': float(np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'count': int(len(valid_values)),
                'finite_count': int(np.sum(finite_mask)),
                'finite_frac': float(np.mean(finite_mask))
            }
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 
                'count': 0, 'finite_count': 0, 'finite_frac': 0.0}
    
    # Calculate statistics for bits of memorization
    bits_stats = safe_stats(bits_of_memorization)
    
    # Calculate positive and negative memorization statistics
    pos_mask = bits_of_memorization > 0
    neg_mask = bits_of_memorization < 0
    
    # Calculate position-specific statistics if position_col is available
    pos_stats = {}
    if position_col and position_col in df.columns:
        for pos in df[position_col].unique():
            pos_mask = df[position_col] == pos
            if pos_mask.sum() > 0:
                pos_bits = bits_of_memorization[pos_mask]
                pos_stats[f'frac_positive_{pos}'] = float(np.mean(pos_bits > 0)) if len(pos_bits) > 0 else 0.0
                pos_stats[f'frac_negative_{pos}'] = float(np.mean(pos_bits < 0)) if len(pos_bits) > 0 else 0.0
                pos_stats[f'mean_positive_bits_{pos}'] = float(np.mean(pos_bits[pos_bits > 0])) if np.any(pos_bits > 0) else 0.0
                pos_stats[f'mean_negative_bits_{pos}'] = float(np.mean(pos_bits[pos_bits < 0])) if np.any(pos_bits < 0) else 0.0
                pos_stats[f'n_samples_{pos}'] = int(pos_mask.sum())
    
    # Calculate overall fractions
    frac_positive = float(np.mean(pos_mask)) if len(bits_of_memorization) > 0 else 0.0
    frac_negative = float(np.mean(neg_mask)) if len(bits_of_memorization) > 0 else 0.0
    
    # Calculate mean positive/negative bits
    mean_positive_bits = float(np.mean(bits_of_memorization[pos_mask])) if np.any(pos_mask) else 0.0
    mean_negative_bits = float(np.mean(bits_of_memorization[neg_mask])) if np.any(neg_mask) else 0.0
    
    # Prepare aggregates dictionary with enhanced diagnostics
    aggregates = {
        # Basic bits statistics
        'mean_bits_per_sample': bits_stats['mean'],
        'median_bits_per_sample': bits_stats['median'],
        'std_bits': bits_stats['std'],
        'min_bits': bits_stats['min'],
        'max_bits': bits_stats['max'],
        'finite_bits_frac': bits_stats['finite_frac'],
        'total_bits': bits_stats['mean'] * bits_stats['count'],  # Preserve scale
        
        # Fractions
        'frac_positive': frac_positive,
        'frac_negative': frac_negative,
        'frac_zero': 1.0 - (frac_positive + frac_negative),
        
        # Mean bits by sign
        'mean_positive_bits': mean_positive_bits,
        'mean_negative_bits': mean_negative_bits,
        
        # Sample counts
        'n_samples': len(df),
        'n_finite_bits': int(bits_stats['finite_count']),
        
        # Configuration
        'variance_method': str(variance_estimator) if not callable(variance_estimator) else 'custom',
        'min_std_dev': float(min_std_dev),
        'max_std_dev': float(max_std_dev),
        'window_size': int(window_size),
        'min_samples_for_std': int(min_samples_for_std),
        'clip_residuals': float(clip_residuals),
        
        # Add position-specific stats
        **pos_stats,
        
        # Add warning flags
        'warning_high_std_frac': float(
            (np.mean(sigma_model >= max_std_dev * 0.9) > 0.1) or 
            (np.mean(sigma_baseline >= max_std_dev * 0.9) > 0.1)
        ),
        'warning_low_std_frac': float(
            (np.mean(sigma_model <= min_std_dev * 1.1) > 0.5) or 
            (np.mean(sigma_baseline <= min_std_dev * 1.1) > 0.5)
        ),
        'warning_extreme_bits': float(
            (abs(bits_stats['mean']) > 10) or 
            (abs(bits_stats['median']) > 10) or
            (not np.isfinite(bits_stats['mean'])) or
            (not np.isfinite(bits_stats['median']))
        )
    }
    
    # Add variance estimation info with additional diagnostics
    aggregates.update({
        'model_std_mean': float(np.mean(sigma_model)),
        'model_std_median': float(np.median(sigma_model)),
        'model_std_min': float(np.min(sigma_model)),
        'model_std_max': float(np.max(sigma_model)),
        'model_std_frac_at_min': float(np.mean(sigma_model <= min_std_dev * 1.1)),
        'model_std_frac_at_max': float(np.mean(sigma_model >= max_std_dev * 0.9)),
        'baseline_std_mean': float(np.mean(sigma_baseline)),
        'baseline_std_median': float(np.median(sigma_baseline)),
        'baseline_std_min': float(np.min(sigma_baseline)),
        'baseline_std_max': float(np.max(sigma_baseline)),
        'baseline_std_frac_at_min': float(np.mean(sigma_baseline <= min_std_dev * 1.1)),
        'baseline_std_frac_at_max': float(np.mean(sigma_baseline >= max_std_dev * 0.9)),
        **var_info  # Include any additional variance info from the estimator
    })
    
    # Add position-specific aggregates if position_col is provided
    if position_col and position_col in df.columns:
        for pos in df[position_col].unique():
            pos_mask = df[position_col] == pos
            if pos_mask.sum() > 0:
                pos_bits = bits_of_memorization[pos_mask]
                pos_pos_mask = pos_bits > 0
                pos_neg_mask = pos_bits < 0
                
                aggregates[f'frac_positive_{pos}'] = float(np.mean(pos_pos_mask)) if len(pos_bits) > 0 else 0.0
                aggregates[f'frac_negative_{pos}'] = float(np.mean(pos_neg_mask)) if len(pos_bits) > 0 else 0.0
                aggregates[f'mean_positive_bits_{pos}'] = float(np.mean(pos_bits[pos_pos_mask])) if np.any(pos_pos_mask) else 0.0
                aggregates[f'mean_negative_bits_{pos}'] = float(np.mean(pos_bits[pos_neg_mask])) if np.any(pos_neg_mask) else 0.0
                aggregates[f'n_samples_{pos}'] = int(pos_mask.sum())
    
    # Add fold ID to aggregates if provided
    if fold_id is not None:
        aggregates['fold_id'] = fold_id
    
    # Prepare per-sample results if requested
    if return_df:
        result_data = {
            group_col: df[group_col].values,
            time_col: df[time_col].values,
            'y_true': y_true,
            'y_pred_model': y_pred_model,
            'y_pred_baseline': y_pred_baseline,
            'residual_model': residuals,
            'residual_baseline': baseline_residuals,
            'sigma_model': sigma_model,
            'sigma_baseline': sigma_baseline,
            'nll_model': nll_model,
            'nll_baseline': nll_baseline,
            'bits_of_memorization': bits_of_memorization,
            'is_positive_memorization': pos_mask,
            'is_negative_memorization': neg_mask
        }
        
        # Add position if available
        if position_col and position_col in df.columns:
            result_data[position_col] = df[position_col].values
        
        # Add fallback info if available
        if 'fallback_info' in locals():
            result_data['used_global_mean_fallback'] = fallback_info['global_mean_fallback_used']
        
        # Add variance estimation flags and diagnostics
        result_data.update({
            'var_fallback_used': var_info.get('var_fallback_used', False),
            'model_std': sigma_model,
            'baseline_std': sigma_baseline,
            'nll_model': nll_model,
            'nll_baseline': nll_baseline,
            'timestamp': pd.Timestamp.now().isoformat(),
            'min_std_dev_used': min_std_dev,
            'max_std_dev_used': max_std_dev,
            'clip_residuals_used': clip_residuals
        })
        
        # Create the results DataFrame
        results_df = pd.DataFrame(result_data)
        
        return results_df, aggregates
    
    return None, aggregates

def _estimate_variances(
    residuals: np.ndarray,
    baseline_residuals: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    variance_estimator: Union[str, Callable],
    window_size: int = 10,
    min_std_dev: float = 1e-6,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Estimate variances for model and baseline predictions.
    
    Args:
        residuals: Model residuals (y_true - y_pred_model)
        baseline_residuals: Baseline model residuals (y_true - y_pred_baseline)
        y_pred_model: Model predictions
        y_pred_baseline: Baseline model predictions
        df: DataFrame containing the data
        group_col: Column name for grouping (e.g., player_id)
        time_col: Column name for time ordering
        variance_estimator: Method for estimating variance
        window_size: Size of rolling window for variance estimation
        min_std_dev: Minimum standard deviation to avoid division by zero
        
    Returns:
        Tuple of (sigma_model, sigma_baseline, var_info)
    """
    var_info = {}
    
    def safe_rolling_std(series, window, min_periods=3):
        """Calculate rolling standard deviation with fallback to global std."""
        try:
            rolling_std = series.rolling(window=window, min_periods=min_periods).std()
            
            # Check if we have enough data for rolling std
            if rolling_std.isna().all():
                global_std = series.std()
                logger.warning(f"Rolling std failed with window={window}, using global std {global_std:.4f}")
                return pd.Series(global_std, index=series.index), True
                
            # Forward fill NaN values that occur at the start of each group
            rolling_std = rolling_std.groupby(df[group_col]).ffill()
            
            # For any remaining NaNs, use global std
            if rolling_std.isna().any():
                global_std = series.std()
                rolling_std = rolling_std.fillna(global_std)
                return rolling_std, True
                
            return rolling_std, False
            
        except Exception as e:
            logger.warning(f"Error in rolling std calculation: {str(e)}. Using global std.")
            global_std = series.std()
            return pd.Series(global_std, index=series.index), True
    
    if isinstance(variance_estimator, str):
        if variance_estimator == 'homoscedastic':
            # Single variance estimate for all samples
            sigma_model = np.full_like(residuals, np.std(residuals, ddof=1))
            sigma_baseline = np.full_like(baseline_residuals, np.std(baseline_residuals, ddof=1))
            var_info['variance_method'] = 'homoscedastic'
            
        elif variance_estimator == 'rolling':
            # Rolling window variance within each group
            df_resid = df[[group_col, time_col]].copy()
            df_resid['residual'] = residuals
            df_resid['baseline_residual'] = baseline_residuals
            
            # Sort by group and time
            df_resid = df_resid.sort_values(by=[group_col, time_col])
            
            # Calculate rolling std within each group
            rolling_std_model = df_resid.groupby(group_col)['residual'].apply(
                lambda x: x.rolling(window=window_size, min_periods=min_samples).std()
            )
            
            rolling_std_baseline = df_resid.groupby(group_col)['baseline_residual'].apply(
                lambda x: x.rolling(window=window_size, min_periods=min_samples).std()
            )
            
            # Forward fill within each group to handle initial NaNs
            rolling_std_model = rolling_std_model.groupby(level=0).ffill()
            rolling_std_baseline = rolling_std_baseline.groupby(level=0).ffill()
            
            # For any remaining NaNs, use global std
            global_std_model = np.std(residuals, ddof=1)
            global_std_baseline = np.std(baseline_residuals, ddof=1)
            
            sigma_model = rolling_std_model.fillna(global_std_model).values
            sigma_baseline = rolling_std_baseline.fillna(global_std_baseline).values
            
            # Log fallback usage
            model_fallback_frac = np.isnan(rolling_std_model).mean() * 100
            baseline_fallback_frac = np.isnan(rolling_std_baseline).mean() * 100
            
            if model_fallback_frac > 0 or baseline_fallback_frac > 0:
                logger.warning(
                    f"Used global std fallback for {model_fallback_frac:.1f}% of model "
                    f"and {baseline_fallback_frac:.1f}% of baseline variance estimates"
                )
            
            var_info.update({
                'variance_method': 'rolling',
                'window_size': window_size,
                'model_fallback_frac': float(model_fallback_frac),
                'baseline_fallback_frac': float(baseline_fallback_frac)
            })
            
        elif variance_estimator == 'heteroscedastic_rolling':
            # More robust rolling variance with fallback to global statistics
            df_resid = df[[group_col, time_col]].copy()
            df_resid['residual'] = residuals
            df_resid['baseline_residual'] = baseline_residuals
            
            # Sort by group and time
            df_resid = df_resid.sort_values(by=[group_col, time_col])
            
            # Calculate rolling std with fallback
            rolling_std_model, model_fallback = safe_rolling_std(
                df_resid.groupby(group_col)['residual'].transform('mean'),
                window=window_size
            )
            
            rolling_std_baseline, baseline_fallback = safe_rolling_std(
                df_resid.groupby(group_col)['baseline_residual'].transform('mean'),
                window=window_size
            )
            
            # Ensure minimum standard deviation
            sigma_model = np.maximum(rolling_std_model, min_std_dev).values
            sigma_baseline = np.maximum(rolling_std_baseline, min_std_dev).values
            
            # Track fallback usage
            var_info.update({
                'variance_method': 'heteroscedastic_rolling',
                'window_size': window_size,
                'model_used_fallback': bool(model_fallback),
                'baseline_used_fallback': bool(baseline_fallback),
                'min_std_dev': float(min_std_dev)
            })
            
            if model_fallback or baseline_fallback:
                logger.warning(
                    f"Used fallback variance estimation (model: {model_fallback}, "
                    f"baseline: {baseline_fallback})"
                )
                
        else:
            raise ValueError(f"Unknown variance estimator: {variance_estimator}")
            
    elif callable(variance_estimator):
        # Custom variance estimator function
        try:
            sigma_model, sigma_baseline = variance_estimator(
                residuals, baseline_residuals, y_pred_model, y_pred_baseline, df
            )
            var_info['variance_method'] = 'custom_function'
            
            # Validate outputs
            if not (isinstance(sigma_model, np.ndarray) and isinstance(sigma_baseline, np.ndarray)):
                raise ValueError("Custom variance estimator must return two numpy arrays")
                
            if len(sigma_model) != len(residuals) or len(sigma_baseline) != len(residuals):
                raise ValueError("Variance estimates must have same length as input data")
                
        except Exception as e:
            logger.error(f"Error in custom variance estimator: {str(e)}")
            logger.warning("Falling back to homoscedastic variance estimation")
            
            # Fall back to homoscedastic
            sigma_model = np.full_like(residuals, np.std(residuals, ddof=1))
            sigma_baseline = np.full_like(baseline_residuals, np.std(baseline_residuals, ddof=1))
            
            var_info.update({
                'variance_method': 'homoscedastic_fallback',
                'error': str(e)
            })
    
    else:
        raise TypeError("variance_estimator must be a string or callable")
    
    # Ensure minimum standard deviation
    sigma_model = np.maximum(sigma_model, min_std_dev)
    sigma_baseline = np.maximum(sigma_baseline, min_std_dev)
    
    return sigma_model, sigma_baseline, var_info

def format_sample_row(row: pd.Series, include_fallback: bool = False) -> str:
    """Format a single row of sample data for the report."""
    # Base information
    player_id = row.get('player_id', 'unknown')
    gw = row.get('gw', 'unknown')
    score = row.get('memorization_score', row.get('bits_of_memorization', 0))
    y_true = row.get('y_true', 0)
    y_pred_model = row.get('y_pred_model', 0)
    y_pred_baseline = row.get('y_pred_baseline', 0)
    
    # Format the base string
    parts = [
        f"- Player {player_id}, GW {gw}:",
        f"Score={score:.2f}",
        f"(True={y_true:.1f}, Model={y_pred_model:.1f}, Baseline={y_pred_baseline:.1f})"
    ]
    
    # Add position if available
    position = row.get('position') or row.get('pos')
    if position:
        parts.append(f"Pos={position}")
    
    # Add fallback info if available
    if include_fallback and 'used_global_mean_fallback' in row:
        parts.append("[Used fallback]" if row['used_global_mean_fallback'] else "")
    
    # Add variance fallback info if available
    if 'var_fallback_used' in row and row['var_fallback_used']:
        parts.append("[Var fallback]")
    
    return " ".join(part for part in parts if part)

def generate_memorization_report(
    results_list: List[Tuple[Optional[pd.DataFrame], Dict]],
    metric_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    n_samples: int = 5
) -> str:
    """
    Generate a human-readable report from memorization diagnostics.
    
    Args:
        results_list: List of (df, aggregates) tuples from the diagnostic functions
        metric_names: Optional list of names for each metric in results_list
        output_path: Optional path to save the report as markdown
        n_samples: Number of top and bottom samples to show in the report
        
    Returns:
        Formatted report as a string
    """
    if metric_names is None:
        metric_names = [f"Metric_{i+1}" for i in range(len(results_list))]
    
    report = ["# Model Memorization Analysis Report\n"]
    
    # Add summary section
    report.append("## Executive Summary\n")
    report.append("This report analyzes the model's tendency to memorize training data "
                "versus learning generalizable patterns.\n")
    
    # Add results for each metric
    for (df, aggs), name in zip(results_list, metric_names):
        if 'mean_mem_score' in aggs:  # Squared error metric
            report.append(f"### {name} (Squared Error)")
            report.append(f"- **Average memorization score**: {aggs['mean_mem_score']:.4f}")
            report.append(f"- **Total memorization**: {aggs['sum_mem_score']:.2f}")
            report.append(f"- **95th percentile**: {aggs['p95_mem_score']:.4f}")
            report.append(f"- **Samples with positive memorization**: {aggs['frac_positive']*100:.1f}%")
            report.append(f"- **Samples with negative memorization**: {aggs['frac_negative']*100:.1f}%")
            
            # Add mean positive and negative memorization
            if 'mean_positive_mem' in aggs and aggs['mean_positive_mem'] > 0:
                report.append(f"- **Mean positive memorization**: {aggs['mean_positive_mem']:.4f}")
            if 'mean_negative_mem' in aggs and aggs['mean_negative_mem'] < 0:
                report.append(f"- **Mean negative memorization**: {aggs['mean_negative_mem']:.4f}")
                
        elif 'mean_bits_per_sample' in aggs:  # Bits metric
            report.append(f"### {name} (Bits of Memorization)")
            report.append(f"- **Average bits per sample**: {aggs['mean_bits_per_sample']:.4f}")
            report.append(f"- **Total bits of memorization**: {aggs['total_bits']:.2f}")
            report.append(f"- **95th percentile (bits)**: {aggs['p95_bits']:.4f}")
            report.append(f"- **Samples with positive memorization**: {aggs['frac_positive']*100:.1f}%")
            report.append(f"- **Samples with negative memorization**: {aggs.get('frac_negative', 0)*100:.1f}%")
        
        # Add top and bottom memorized samples if available
        if df is not None and not df.empty and 'memorization_score' in df.columns:
            # Top memorized samples (model much better than baseline)
            top_mem = df.nlargest(n_samples, 'memorization_score')
            if len(top_mem) > 0:
                report.append("\n**Top memorized samples (model much better than baseline):**")
                for _, row in top_mem.iterrows():
                    report.append(format_sample_row(row, include_fallback='used_global_mean_fallback' in df.columns))
            
            # Bottom memorized samples (baseline much better than model)
            bottom_mem = df.nsmallest(n_samples, 'memorization_score')
            if len(bottom_mem) > 0 and (bottom_mem['memorization_score'] < 0).any():
                report.append("\n**Worst performing samples (baseline better than model):**")
                for _, row in bottom_mem.iterrows():
                    if row['memorization_score'] < 0:  # Only show if actually worse than baseline
                        report.append(format_sample_row(row, include_fallback='used_global_mean_fallback' in df.columns))
            
            # Add position-specific summaries if available
            position_col = next((col for col in ['position', 'pos'] if col in df.columns), None)
            if position_col:
                report.append("\n**By position:**")
                pos_summary = []
                
                for pos in sorted(df[position_col].unique()):
                    pos_df = df[df[position_col] == pos]
                    if len(pos_df) > 0:
                        pos_mean = pos_df['memorization_score'].mean()
                        pos_frac_pos = pos_df['memorization_score'].gt(0).mean() * 100
                        pos_frac_neg = pos_df['memorization_score'].lt(0).mean() * 100
                        
                        pos_summary.append({
                            'position': pos,
                            'n_samples': len(pos_df),
                            'mean_score': pos_mean,
                            'frac_positive': pos_frac_pos,
                            'frac_negative': pos_frac_neg
                        })
                
                # Sort by sample size or mean score
                pos_summary.sort(key=lambda x: -x['n_samples'])
                
                for pos_info in pos_summary:
                    report.append(
                        f"- {pos_info['position']}: "
                        f"n={pos_info['n_samples']}, "
                        f"Mean={pos_info['mean_score']:.2f}, "
                        f"Positive={pos_info['frac_positive']:.1f}%, "
                        f"Negative={pos_info['frac_negative']:.1f}%"
                    )
        
        report.append("\n---\n")
    
    # Add interpretation section
    report.append("## Interpretation and Recommendations\n")
    report.append("### Key Findings")
    report.append("- **High positive memorization scores** suggest the model is learning patterns that the baseline cannot capture, which could be either valuable signal or overfitting.")
    report.append("- **Negative scores** indicate cases where the baseline outperforms the model, which may suggest areas where the model could be improved.")
    report.append("- **Position-based analysis** helps identify if memorization patterns vary by player position.")
    report.append("- **Fallback usage** (when shown) indicates how often the baseline had to use global statistics due to insufficient history.")
    
    report.append("\n### Recommendations")
    report.append("- Investigate samples with high positive memorization to determine if they represent valuable patterns or overfitting.")
    report.append("- Examine samples with negative memorization to understand why the baseline performs better and how the model can be improved.")
    report.append("- Consider position-specific modeling if memorization patterns vary significantly by position.")
    report.append("- For time series data, check if the model's performance changes over time or with different amounts of historical data.")
    
    # Convert to string
    report_str = "\n".join(report)
    
    # Save to file if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        logger.info(f"Report saved to {os.path.abspath(output_path)}")
    
    return report_str
