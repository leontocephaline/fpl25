"""
Historical Backtesting Engine for FPL using SQLite Database

Runs week-by-week backtests using real historical data with:
- Progress bars and live stats
- Multiple model benchmarks (XGBoost, Decision Tree, Random, Perfect)
- Intermediate checkpointing
- Comprehensive reporting
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import pickle
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import uuid
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time

from ..data.historical_db import FPLHistoricalDB
from ..data.data_processor import DataProcessor
from ..data.data_importer import FPLDataImporter
from ..models.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# Sanitize log messages for Windows consoles that can't encode emojis (cp1252)
class AsciiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = str(record.msg).encode('ascii', 'ignore').decode()
        except Exception:
            pass
        return True

logger.addFilter(AsciiFilter())

class HistoricalBacktester:
    """Historical backtest engine using real FPL data with performance optimizations"""
    
    def __init__(self, db_path: str = None, config: Dict = None):
        """Initialize the backtesting engine
        
        Args:
            db_path: Path to SQLite database file
            config: Configuration dictionary with parameters
        """
        self.db = FPLHistoricalDB(db_path)
        self.config = config or {}
        self.data_processor = DataProcessor(config=self.config)
        self.importer = FPLDataImporter(self.db)
        
        # Performance optimization settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', min(4, mp.cpu_count()))
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model hyperparameters configuration
        self.model_hyperparams = self._get_model_hyperparameters()
        
        # Initialize caches
        self._feature_cache = {}
        self._model_cache = {}
        
        # Initialize benchmarks
        self.benchmarks = {
            'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8)
        }
        
        logger.info(f"HistoricalBacktester initialized - DB: {self.db.db_path}, Caching: {self.enable_caching}, Parallel: {self.enable_parallel}, Workers: {self.max_workers}")
        
        # Model directory
        if 'models_dir' not in self.config:
            project_root = Path(__file__).parent.parent.parent
            self.config['models_dir'] = str(project_root / "models")
        self.models_dir = Path(self.config['models_dir'])
        
        # Results storage
        self.run_id = str(uuid.uuid4())[:8]
        self.checkpoint_frequency = 5  # Save every N gameweeks
    
    def _generate_cache_key(self, season: str, gw: int, data_type: str, **kwargs) -> str:
        """Generate a unique cache key for data caching"""
        key_data = f"{season}_{gw}_{data_type}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached feature data"""
        if not self.enable_caching:
            return None
            
        # Check memory cache first
        if cache_key in self._feature_cache:
            logger.debug(f"📋 Cache hit (memory): {cache_key}")
            return self._feature_cache[cache_key].copy()
        
        # Check disk cache
        cache_file = self.cache_dir / f"features_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._feature_cache[cache_key] = data  # Store in memory cache
                    logger.debug(f"📋 Cache hit (disk): {cache_key}")
                    return data.copy()
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _cache_features(self, cache_key: str, data: pd.DataFrame):
        """Cache feature data to memory and disk"""
        if not self.enable_caching or data is None or data.empty:
            return
            
        try:
            # Store in memory cache (limit size)
            if len(self._feature_cache) > 50:  # Limit memory cache size
                # Remove oldest entries
                oldest_key = list(self._feature_cache.keys())[0]
                del self._feature_cache[oldest_key]
            
            self._feature_cache[cache_key] = data.copy()
            
            # Store in disk cache
            cache_file = self.cache_dir / f"features_{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug(f"💾 Cached features: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache features {cache_key}: {e}")
    
    def _train_predict_position(self, task: Dict, model) -> List[Dict]:
        """Train and predict for a single position (used for parallel processing)"""
        position = task['position']
        pos_train = task['pos_train']
        pos_test = task['pos_test']
        context = task['context']
        current_gw = task['current_gw']
        should_retrain = task.get('should_retrain', False)
        predictions = []
        
        logger.info(f"🏋️  {position} {context} - candidates: train={len(pos_train)}, test={len(pos_test)}, should_retrain={should_retrain}")
        
        try:
            # Decide whether to train or reuse prior model for this position
            last_trained_gw = getattr(self, '_last_trained_gw', {}).get(position)
            train_now = should_retrain or (last_trained_gw is None)
            horizon_for_pred = current_gw
            
            if train_now:
                logger.info(f"🧠 Training {position} model {context} @GW{current_gw} (last_trained={last_trained_gw})")
                model.fit(pos_train, position, {}, current_gw)
                # Record last trained GW for this position
                if not hasattr(self, '_last_trained_gw'):
                    self._last_trained_gw = {}
                self._last_trained_gw[position] = current_gw
                horizon_for_pred = current_gw
            else:
                # Reuse prior trained horizon
                horizon_for_pred = last_trained_gw
                logger.info(f"♻️  Reusing {position} model {context} trained at GW{last_trained_gw} (skip retrain)")
            
            # Get predictions for current gameweek
            try:
                pred_result = model.predict(pos_test, position, horizon_for_pred)
            except RuntimeError as e:
                # If model not available for reused horizon, train now then retry
                if ("No model available" in str(e)) or ("No valid models available" in str(e)):
                    logger.info(f"⚠️  Missing in-memory model for {position} GW{horizon_for_pred}. Training now and retrying predict().")
                    model.fit(pos_train, position, {}, current_gw)
                    if not hasattr(self, '_last_trained_gw'):
                        self._last_trained_gw = {}
                    self._last_trained_gw[position] = current_gw
                    horizon_for_pred = current_gw
                    pred_result = model.predict(pos_test, position, horizon_for_pred)
                else:
                    raise
            
            # Format predictions with actual points for evaluation  
            pos_test_reset = pos_test.reset_index(drop=True)
            for i, player in pos_test_reset.iterrows():
                player_id = player['player_id']
                # Align prediction by position rather than original index
                try:
                    if hasattr(pred_result, 'iloc'):
                        predicted = float(pred_result.iloc[i])
                    elif isinstance(pred_result, (list, np.ndarray)):
                        predicted = float(pred_result[i])
                    else:
                        # Fallback for dict-like results
                        predicted = float(pred_result.get(i, 0.0))
                except Exception:
                    predicted = 0.0
                predictions.append({
                    'player_id': player_id,
                    'name': player.get('name', ''),
                    'position': position,
                    'predicted_points': predicted,
                    'actual_points': player.get('total_points', 0)
                })
                
            logger.info(f"✅ {position} model {context}: trained on {len(pos_train)} historical, predicted {len(pos_test)} current")
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {position} {context}: {e}")
            # Fallback to position average if prediction fails
            pos_avg = pos_train['total_points'].mean() if not pos_train.empty else 2.0
            for _, player in pos_test.iterrows():
                predictions.append({
                    'player_id': player['player_id'],
                    'name': player.get('name', ''),
                    'position': position,
                    'predicted_points': pos_avg,
                    'actual_points': player.get('total_points', 0)
                })
        
        return predictions
    
    def _get_model_hyperparameters(self) -> Dict:
        """Get configurable model hyperparameters from config"""
        default_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            },
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 8,
                'min_samples_split': 20,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Override with user-provided hyperparameters
        config_params = self.config.get('model_hyperparams', {})
        for model_type, params in config_params.items():
            if model_type in default_params:
                default_params[model_type].update(params)
                logger.info(f"Updated {model_type} hyperparameters: {params}")
        
        return default_params
        
    def _setup_benchmarks(self):
        """Setup benchmark models for comparison using configurable hyperparameters"""
        dt_params = self.model_hyperparams.get('decision_tree', {})
        rf_params = self.model_hyperparams.get('random_forest', {})
        
        self.benchmarks = {
            'decision_tree': DecisionTreeRegressor(**dt_params),
            'random_forest': RandomForestRegressor(**rf_params)
        }

    def _load_model(self, model_type: str, position: str, retrain: bool = False) -> Optional[MLPredictor]:
        """Load the specified model with configurable hyperparameters"""
        try:
            # Use configured hyperparameters instead of hardcoded ones
            xgb_params = self.model_hyperparams.get('xgboost', {})
            lgb_params = self.model_hyperparams.get('lightgbm', {})
            
            dummy_config = {
                'ml': {
                    # Map shorthands to full names expected by MLPredictor
                    'model_type': {'xgb': 'xgboost', 'lgbm': 'lightgbm', 'dt': 'xgboost'}.get(model_type, model_type),
                    'xgboost': xgb_params,
                    'lightgbm': lgb_params,
                    'save_feature_importance':False,
                }
            }

            # Check if a model file exists before creating the predictor
            model_files_exist = False
            search_token = {'xgboost': 'xgb', 'lightgbm': 'lgbm'}.get(model_type, model_type)
            
            if position == 'ALL':
                # For ALL, check for any position-specific model
                if any(self.models_dir.glob(f"*_*_{search_token}*.joblib")):
                    model_files_exist = True
                elif any(self.models_dir.glob(f"*_{search_token}*.joblib")): # Legacy check
                    model_files_exist = True
            else:
                # For specific positions, check for that position's model
                if any(self.models_dir.glob(f"{position}_*_{search_token}*.joblib")):
                    model_files_exist = True
                elif any(self.models_dir.glob(f"*_{search_token}*.joblib")): # Fallback
                    model_files_exist = True

            if not model_files_exist and not retrain:
                # Proceed without preloaded files; models will be trained in-session at first GW
                logger.warning(f"No saved model found for {model_type}_{position}; proceeding to train in-session.")

            ml_predictor = MLPredictor(dummy_config)
            return ml_predictor

        except Exception as e:
            logger.error(f"Failed to load model {model_type}_{position}: {e}")
            return None

    def _get_data_processor(self):
        dummy_config = {}
        dp = DataProcessor(dummy_config)
        return dp

    
    def run_historical_backtest(
        self,
        seasons: List[str] = None,
        gw_range: Tuple[int, int] = None,
        model_type: str = 'xgb',
        position: str = 'ALL',
        use_live_mode: bool = True,
        recalibration_window: int = None,
        checkpoint_dir: str = None,
        retrain_frequency: int = None,
        retrain: bool = False
    ) -> Dict:
        """
        Run comprehensive historical backtest
        
        Args:
            seasons: List of seasons to test (default: all available)
            gw_range: Tuple of (start_gw, end_gw) per season
            model_type: Primary model type to use
            position: Position to test ('ALL', 'GK', 'DEF', 'MID', 'FWD')
            use_live_mode: Whether to simulate live conditions
            retrain: If True, force retraining of models even if they exist
            
        Returns:
            Complete backtest results dictionary
        """
        logger.info(f"Starting historical backtest (Run ID: {self.run_id})")
        
        # Setup
        if seasons is None:
            seasons = self.db.get_available_seasons()
            if not seasons:
                logger.error("No seasons available in database. Run data import first.")
                return {}
        
        if gw_range is None:
            gw_range = (1, 38)
        
        # Ensure data is up to date
        logger.info("Checking for latest data updates...")
        for season in seasons:
            self.importer.auto_update_latest_data(season)

        # Generate or load feature superset for consistent model training/prediction
        superset_path = Path('models') / 'fpl_safe_superset_features.txt'
        if not superset_path.exists() or retrain:
            logger.info("Generating global feature superset...")
            self._generate_feature_superset(seasons, gw_range)
        else:
            logger.info(f"Using existing feature superset from {superset_path}")
        
        # Initialize results structure
        results = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'seasons': seasons,
                'gw_range': gw_range,
                'recalibration_window': recalibration_window,
                'model_type': model_type,
                'retrain_frequency': retrain_frequency,
                'position': position,
                'use_live_mode': use_live_mode
            },
            'results': {
                'by_season': {},
                'overall': {},
                'benchmarks': {}
            },
            'rolling_stats': []
        }
        
        # Setup checkpoint directory

        if checkpoint_dir is None:
            project_root = Path(__file__).parent.parent.parent
            checkpoint_dir = project_root / "data" / "checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"backtest_{self.run_id}.pkl"
        
        try:
            # Run backtest for each season
            total_gws = len(seasons) * (gw_range[1] - gw_range[0] + 1)
            progress_bar = tqdm(total=total_gws, desc="Historical Backtest")

            
            gw_counter = 0
            
            for season in seasons:
                logger.info(f"Processing season: {season}")
                season_results = self._run_season_backtest(
                    season, gw_range, model_type, position, 
                    use_live_mode, recalibration_window, progress_bar,
                    retrain_frequency, retrain
                )
                
                results['results']['by_season'][season] = season_results
                
                # Update rolling stats and checkpoint periodically
                for gw_result in season_results.get('by_gameweek', {}).values():
                    gw_counter += 1
                    self._update_rolling_stats(results, gw_result)
                    
                    if gw_counter % self.checkpoint_frequency == 0:
                        self._save_checkpoint(results, checkpoint_file)
            
            progress_bar.close()
            
            # Calculate overall statistics
            results['results']['overall'] = self._calculate_overall_stats(results)
            
            # Run benchmark comparisons
            results['results']['benchmarks'] = self._run_benchmarks(
                results, seasons, gw_range
            )
            
            # Final save
            self._save_results(results)
            
            logger.info(f"Historical backtest completed (Run ID: {self.run_id})")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            # Save partial results
            self._save_checkpoint(results, checkpoint_file)
            raise
    
    def _run_season_backtest(
        self,
        season: str,
        gw_range: Tuple[int, int],
        model_type: str,
        position: str,
        use_live_mode: bool,
        recalibration_window: int,
        progress_bar: tqdm,
        retrain_frequency: int,
        retrain: bool = False
    ) -> Dict:
        """Run backtest for a single season
        
        Args:
            season: Season to run backtest for (e.g., '2023-24')
            gw_range: Tuple of (start_gw, end_gw) to test
            model_type: Type of model to use ('xgb' or 'lgbm')
            position: Position to test ('GK', 'DEF', 'MID', 'FWD', or 'ALL')
            recalibration_window: The size of the rolling window, if enabled
            use_live_mode: Whether to simulate live prediction conditions
            progress_bar: tqdm progress bar for tracking progress
            retrain: If True, force retraining of the model even if it exists
            
        Returns:
            Dictionary containing backtest results for the season
        """
        
        season_results = {
            'season': season,
            'by_gameweek': {},
            'summary': {}
        }
        ml_predictor = None
        
        # Load primary model with retrain flag (ensure model_type is correctly passed)
        ml_predictor = self._load_model(model_type, position, retrain=retrain)
        logger.info(f"🔧 Model loading result: ml_predictor={'LOADED' if ml_predictor else 'NULL'}, model_type={model_type}, position={position}, retrain={retrain}")
        
        start_gw, end_gw = gw_range
        
        # Track last trained GW per position for reuse when retraining is skipped
        # Reset per season run
        self._last_trained_gw = {}
        
        for gw in range(start_gw, end_gw + 1):
            try:
                # Get available data for this gameweek
                gw_data = self.db.get_player_data(season, gw, exclude_outliers=True)
                
                if gw_data.empty:
                    progress_bar.update(1)
                    continue
                
                # Check if we should retrain based on retrain frequency
                should_retrain = retrain
                if retrain_frequency and gw > start_gw:
                    # Only retrain on gameweeks divisible by retrain_frequency
                    should_retrain = should_retrain or (gw % retrain_frequency == 0)
                    if should_retrain and gw % retrain_frequency == 0:
                        logger.info(f"Retraining triggered at GW {gw} (frequency: every {retrain_frequency} GWs)")
                elif not retrain_frequency:
                    # Default: retrain every GW if no frequency specified
                    should_retrain = retrain or gw > start_gw
                
                # Simulate live conditions - only use data available up to this point
                if use_live_mode and gw > 1:
                    historical_data = self._get_historical_features(season, gw, recalibration_window)
                else:
                    historical_data = gw_data
                
                # Run predictions for different approaches
                gw_result = {
                    'season': season,
                    'gameweek': gw,
                    'predictions': {},
                    'actuals': {},
                    'metrics': {}
                }
                
                # Actual results (compute first so Perfect XI can use them)
                # Use NON-filtered GW data to preserve true scorers for metrics and Perfect XI
                gw_data_raw = self.db.get_player_data(season, gw, exclude_outliers=False)
                actuals_df = gw_data_raw.copy() if gw_data_raw is not None and not gw_data_raw.empty else gw_data.copy()
                event_ok = False
                if 'event_points' in actuals_df.columns:
                    ev_raw = pd.to_numeric(actuals_df['event_points'], errors='coerce')
                    # Consider event_points usable only if we actually have non-null values and non-zero total
                    event_ok = bool((ev_raw.notna().sum() > 0) and (ev_raw.fillna(0).abs().sum() > 0))
                if event_ok:
                    actuals_small = actuals_df[['player_id', 'event_points']].rename(columns={'event_points': 'actual_points'})
                elif 'total_points' in actuals_df.columns:
                    tp = pd.to_numeric(actuals_df['total_points'], errors='coerce').fillna(0)
                    if tp.abs().sum() == 0 and 'event_points' in actuals_df.columns:
                        # both are zero-ish; still map from total_points to keep type
                        pass
                    actuals_small = actuals_df[['player_id', 'total_points']].rename(columns={'total_points': 'actual_points'})
                else:
                    logger.warning("No 'event_points' or 'total_points' found in gw_data; actuals will be zeros for metrics")
                    actuals_small = actuals_df[['player_id']].copy()
                    actuals_small['actual_points'] = 0.0
                actuals_small['actual_points'] = pd.to_numeric(actuals_small['actual_points'], errors='coerce').fillna(0)
                gw_result['actuals'] = actuals_small.to_dict('records')

                # Primary model prediction - run whenever a model is loaded (ensures primary model participates)
                should_run_primary = ml_predictor is not None
                logger.info(f"🔍 Primary model check [{season} GW{gw}]: ml_predictor={'OK' if ml_predictor else 'NULL'}, retrain={retrain}, retrain_freq={retrain_frequency}, should_run={should_run_primary}")
                
                if should_run_primary:
                    logger.info(f"🎯 Running primary model prediction [{season} GW{gw}] - Data: {len(historical_data)} historical rows, {len(gw_data)} current players")
                    predictions = self._run_model_prediction(
                        ml_predictor, historical_data, gw_data, should_retrain, season, gw
                    )
                    gw_result['predictions']['primary'] = predictions
                    logger.info(f"✅ Primary model completed [{season} GW{gw}] - Generated {len(predictions) if predictions else 0} predictions")
                else:
                    logger.warning(f"❌ Primary model SKIPPED [{season} GW{gw}] - Condition not met")
                
                # Benchmark predictions - need proper historical training data
                for benchmark_name, benchmark_model in self.benchmarks.items():
                    # Get training data: all data from previous GWs in this season
                    if gw > 1:
                        train_data = self.db.get_historical_data_for_training(season, gw)
                    else:
                        # First GW: use previous season or fallback
                        train_data = self.db.get_previous_season_data(season) if hasattr(self.db, 'get_previous_season_data') else pd.DataFrame()
                    
                    benchmark_pred = self._run_benchmark_prediction(
                        benchmark_model, train_data, gw_data
                    )
                    gw_result['predictions'][benchmark_name] = benchmark_pred
                
                # Perfect XI benchmark (top 11 actual scorers under FPL constraints)
                perfect_xi_pred = self._run_perfect_xi_benchmark(gw_data, pd.DataFrame(gw_result['actuals']))
                gw_result['predictions']['perfect_xi'] = perfect_xi_pred
                
                # Random baseline
                random_pred = self._run_random_benchmark(gw_data)
                gw_result['predictions']['random'] = random_pred

                # Add last-week stars baseline (previous GW top scorers by position)
                lastweek_xi = self._run_last_week_xi_benchmark(season, gw)
                if lastweek_xi:
                    gw_result['predictions']['lastweek_xi'] = lastweek_xi

                # Ensure all predictions have 'actual_points' for consistent metric calculation
                for pred_type, predictions in gw_result['predictions'].items():
                    if predictions:
                        pred_df = pd.DataFrame(predictions).copy()
                        # Reduce actuals to just player_id and a single actual_points column to avoid conflicts
                        actuals_full = pd.DataFrame(gw_result['actuals']).copy()
                        if 'actual_points' not in actuals_full.columns:
                            if 'event_points' in actuals_full.columns:
                                actuals_full['actual_points'] = actuals_full['event_points']
                            elif 'total_points' in actuals_full.columns:
                                actuals_full['actual_points'] = actuals_full['total_points']
                            else:
                                actuals_full['actual_points'] = 0.0
                        actuals_full['actual_points'] = pd.to_numeric(actuals_full['actual_points'], errors='coerce').fillna(0)
                        actuals_small = actuals_full[['player_id', 'actual_points']].copy()
                        # Drop any pre-existing actual_points variants in predictions
                        cols_to_drop = [c for c in pred_df.columns if c.startswith('actual_points')]
                        if cols_to_drop:
                            pred_df = pred_df.drop(columns=cols_to_drop)
                        merged_df = pred_df.merge(actuals_small, on='player_id', how='left')
                        merged_df['actual_points'] = pd.to_numeric(merged_df['actual_points'], errors='coerce').fillna(0)
                        gw_result['predictions'][pred_type] = merged_df.to_dict('records')
                
                # Calculate metrics for each approach with evaluation transparency labels
                for pred_type, predictions in gw_result['predictions'].items():
                    metrics = self._calculate_gw_metrics(predictions, gw_result['actuals'])
                    
                    # Add evaluation transparency labels
                    if pred_type in ['decision_tree', 'random_forest']:
                        metrics['evaluation_type'] = 'out-of-sample'
                        metrics['description'] = f'{pred_type.replace("_", " ").title()} trained on historical GWs <{gw}, predicted on GW{gw}'
                    elif pred_type == 'primary':
                        metrics['evaluation_type'] = 'out-of-sample' 
                        metrics['description'] = f'XGBoost/LightGBM trained on historical GWs <{gw}, predicted on GW{gw}'
                    elif pred_type == 'perfect_xi':
                        metrics['evaluation_type'] = 'oracle'
                        metrics['description'] = 'Perfect XI benchmark (hindsight oracle, FPL constraints)'
                    elif pred_type == 'random':
                        metrics['evaluation_type'] = 'baseline'
                        metrics['description'] = 'Random prediction baseline'
                    else:
                        metrics['evaluation_type'] = 'unknown'
                        metrics['description'] = f'{pred_type} prediction'
                    
                    gw_result['metrics'][pred_type] = metrics

                    # Team-level XI selection and Best XI comparison (skip perfect_xi which is already XI)
                    if pred_type != 'perfect_xi':
                        # Unconstrained XI
                        selected_xi = self._select_xi_from_predictions(gw_data, predictions)
                        if selected_xi:
                            xi_metrics = self._calculate_team_metrics(selected_xi, gw_result['actuals'])
                            # Hit-rate vs Best XI
                            perfect_xi = gw_result.get('predictions', {}).get('perfect_xi', [])
                            try:
                                xi_hit = self._compute_xi_hit_metrics(perfect_xi, selected_xi, gw_result['actuals'])
                                xi_metrics.update({
                                    'xi_overlap_count': xi_hit.get('overlap_count'),
                                    'xi_overlap_ratio': xi_hit.get('overlap_ratio'),
                                    'xi_model_total': xi_hit.get('model_total'),
                                    'xi_best_total': xi_hit.get('best_total'),
                                    'xi_points_gap': xi_hit.get('points_gap'),
                                    'xi_points_gap_abs': xi_hit.get('points_gap_abs'),
                                })
                            except Exception as e:
                                logger.debug(f"Best XI hit-rate metrics skipped: {e}")
                            xi_metrics['evaluation_type'] = metrics.get('evaluation_type', 'unknown')
                            xi_metrics['description'] = metrics.get('description', '') + ' | Team XI (1-4-4-2)'
                            gw_result['metrics'][f"{pred_type}_xi"] = xi_metrics
                        # (Rolled back) Budget-aware XI and cost reporting disabled

                # After collecting metrics for all types, compute normalized entropy/skill scores
                try:
                    self._attach_entropy_scores(gw_result['metrics'])
                except Exception as e:
                    logger.debug(f"Entropy score attachment skipped due to error: {e}")
                
                # Use season-qualified gameweek key to avoid cross-season confusion (e.g., '2023-13')
                gw_key = f"{season}-{gw:02d}"
                season_results['by_gameweek'][gw_key] = gw_result
                
                # Update progress with current stats
                if gw_result['metrics'].get('primary'):
                    primary_metrics = gw_result['metrics']['primary']
                    progress_bar.set_postfix({
                        'Season': season,
                        'GW': gw,
                        'RMSE': f"{primary_metrics.get('rmse', 0):.2f}",
                        'R²': f"{primary_metrics.get('r2', 0):.3f}"
                    })
                
                progress_bar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing {season} GW{gw}: {e}")
                progress_bar.update(1)
                continue
        

        # Calculate season summary
        season_results['summary'] = self._calculate_season_summary(season_results)
        
        return season_results
    
    def _generate_evaluation_summary(self, aggregated_results: Dict) -> str:
        """Generate a summary table showing evaluation transparency for all models"""
        summary_lines = []
        summary_lines.append("\n" + "="*80)
        summary_lines.append("FANTASY FOOTBALL MODEL EVALUATION SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append(f"{'Model':<20} {'Evaluation Type':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
        summary_lines.append("-"*80)
        
        # Sort models by evaluation type and performance
        model_order = ['primary', 'decision_tree', 'random_forest', 'random', 'perfect_xi']
        
        for model_name in model_order:
            if model_name not in aggregated_results:
                continue
                
            metrics = aggregated_results[model_name]
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0) 
            r2 = metrics.get('r2', 0)
            eval_type = metrics.get('evaluation_type', 'unknown')
            
            # Format model name for display
            display_name = {
                'primary': 'XGBoost/LightGBM',
                'decision_tree': 'Decision Tree',
                'random_forest': 'Random Forest', 
                'random': 'Random Baseline',
                'perfect_xi': 'Perfect XI'
            }.get(model_name, model_name)
            
            # Color code evaluation types
            eval_display = {
                'out-of-sample': '🟢 Out-of-sample',
                'oracle': '🔮 Oracle',
                'baseline': '🎲 Baseline',
                'unknown': '❓ Unknown'
            }.get(eval_type, eval_type)
            
            summary_lines.append(
                f"{display_name:<20} {eval_display:<15} {rmse:<8.3f} {mae:<8.3f} {r2:<8.3f}"
            )
        
        summary_lines.append("-"*80)
        summary_lines.append("🟢 Out-of-sample: Fair comparison - trained on historical data, predicted on future")
        summary_lines.append("🔮 Oracle: Perfect hindsight benchmark (uses actual results)")
        summary_lines.append("🎲 Baseline: Random/statistical baseline for comparison")
        summary_lines.append("="*80)
        
        return "\n".join(summary_lines)
    
    def _get_historical_features(self, season: str, current_gw: int, recalibration_window = None) -> pd.DataFrame:
        """Get combined historical data for training from all previous gameweeks in season
        
        Args:
            season: Season to get data for (e.g. '2023-24')
            current_gw: Current gameweek (exclusive, gets data for GWs 1 through current_gw-1)
            recalibration_window: Optional limit to only include N most recent gameweeks
            
        Returns:
            Combined historical DataFrame with lagged features and targets
        """
        # Generate cache key for this feature request
        cache_key = self._generate_cache_key(
            season, current_gw, 'historical_features', 
            recalibration_window=recalibration_window
        )
        
        # Check cache first
        cached_data = self._get_cached_features(cache_key)
        if cached_data is not None:
            return cached_data
        
        start_time = time.time()
        previous_gws = []
        
        # Calculate the starting GW based on recalibration window
        start_gw = max(1, current_gw - recalibration_window) if recalibration_window else 1
        
        logger.debug(f"Getting historical features [{season}] GW {start_gw} to {current_gw-1} (recalibration_window={recalibration_window})")
        
        for gw in range(start_gw, current_gw):
            gw_data = self.db.get_gameweek_data(season, gw)
            if not gw_data.empty:
                previous_gws.append(gw_data)
        
        if not previous_gws:
            return pd.DataFrame()
        
        # Combine all prior gameweeks as raw rows to preserve target and position columns
        historical_df = pd.concat(previous_gws, ignore_index=True)
        
        # Target robustness: alias missing total_points column
        if 'total_points' not in historical_df.columns:
            for alt in ['event_points', 'points', 'gw_points']:
                if alt in historical_df.columns:
                    historical_df['total_points'] = historical_df[alt]
                    logger.info(f"Aliased {alt} to total_points for target consistency in historical features.")
                    break
            else:
                logger.debug("No suitable target column found for total_points alias in historical data.")
        
        # Ensure a unified 'position' column exists
        pos_map_num_to_str = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        if 'position' in historical_df.columns:
            historical_df['position'] = historical_df['position'].apply(
                lambda v: pos_map_num_to_str.get(v, v) if not isinstance(v, str) else v
            )
        elif 'element_type' in historical_df.columns:
            historical_df['position'] = historical_df['element_type'].map(pos_map_num_to_str).fillna('FWD')
        else:
            # As a last resort, set a default to avoid key errors downstream
            historical_df['position'] = 'FWD'
            logger.warning("Historical data lacks position/element_type; defaulting to FWD for all rows")
        
        # Create lagged features by player
        historical_df = self._create_lagged_features(historical_df)
        
        # Cache the results for future use
        processing_time = time.time() - start_time
        logger.debug(f"⚡ Historical features processed in {processing_time:.2f}s - {len(historical_df)} rows")
        self._cache_features(cache_key, historical_df)
        
        # Return enhanced historical data with lagged features
        return historical_df

    def _create_lagged_features(self, df: pd.DataFrame, recalibration_window = None) -> pd.DataFrame:
        """Create lagged and rolling features for each player with caching"""
        if df.empty:
            return df
        
        # Generate cache key for lagged features
        cache_key = self._generate_cache_key(
            'lagged', len(df), 'features',
            recalibration_window=recalibration_window,
            columns=sorted(df.columns.tolist())[:10]  # First 10 columns for key
        )
        
        # Check cache first
        cached_data = self._get_cached_features(cache_key)
        if cached_data is not None and len(cached_data) == len(df):
            return cached_data
        
        start_time = time.time()
        
        # Ensure we have the required columns
        if 'player_id' not in df.columns or 'gw' not in df.columns:
            logger.warning("Missing player_id or gw columns for lagged features")
            return df
        
        # Sort by player and gameweek for proper lag calculation
        df = df.sort_values(['player_id', 'gw']).copy()
        
        # Features to create lags for
        lag_features = {
            'total_points': [1, 3, 5],
            'minutes': [1],
            'ict_index': [1, 3, 5],
            'transfers_in': [1, 3, 5],
            'transfers_out': [1, 3, 5],
            'transfers_balance': [1, 3, 5],
            'influence': [1, 3],
            'creativity': [1, 3],
            'threat': [1, 3],
            'selected_by_percent': [1, 3],
            'value': [1],
            'bps': [1, 3]
        }
        
        # Create lagged features
        for feature, lags in lag_features.items():
            if feature in df.columns:
                # Convert to numeric if needed
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df.groupby('player_id')[feature].shift(lag)
        
        # Create rolling mean features
        rolling_features = {
            'total_points': [3, 5],
            'minutes': [3, 5],
            'ict_index': [3, 5]
        }
        
        for feature, windows in rolling_features.items():
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_mean_{window}'] = df.groupby('player_id')[feature].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
        
        # Create momentum features (delta from previous gameweek)
        momentum_features = ['total_points', 'ict_index', 'selected_by_percent']
        for feature in momentum_features:
            if feature in df.columns:
                df[f'{feature}_delta'] = df.groupby('player_id')[feature].diff()
        
        # Starter prediction features (based on minutes)
        if 'minutes' in df.columns:
            # Previous GW starter status (>60 minutes)
            df['was_starter_last_gw'] = (df.groupby('player_id')['minutes'].shift(1) >= 60).astype(int)
            
            # Rolling starter rate over last 3 gameweeks
            df['starter_rate_3gw'] = (
                df.groupby('player_id')['minutes'].rolling(
                    window=3, min_periods=1
                ).apply(lambda x: (x >= 60).mean()).reset_index(0, drop=True)
            )
        
        logger.debug(f"Created lagged features. Data shape: {df.shape}")
        return df

    def _filter_safe_features(self, df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        """Filter a DataFrame to pre-match/lagged features to avoid leakage.

        Keeps:
        - Identifier/context: player_id, position, gw, element_type, team_id, name
        - Target column 'total_points' only if include_target=True (for training)
        - Lagged/rolling/historical features: columns containing patterns
        - Safe static/pre-match features: cost, form, season stats, etc.
        """
        if df is None or df.empty:
            return df

        keep_cols = set()
        base_keep = {'player_id', 'position', 'gw', 'element_type', 'team_id', 'name', 'web_name', 'short_name'}
        
        # Expanded safe pre-match features
        static_safe = {
            # Cost and value
            'now_cost', 'value', 'cost_change_start', 'cost_change_event', 'cost_change_start_fall',
            'cost_change_event_fall', 'value_form', 'value_season', 'in_dreamteam', 'dreamteam_count',
            
            # Form and points
            'form', 'points_per_game', 'total_points_season', 'points_per_game_rank', 'form_rank',
            'ep_this', 'ep_next', 'ep_this_rank', 'ep_next_rank',
            
            # ICT (Influence, Creativity, Threat) metrics
            'influence', 'creativity', 'threat', 'ict_index',
            'influence_rank', 'creativity_rank', 'threat_rank', 'ict_index_rank',
            'influence_rank_type', 'creativity_rank_type', 'threat_rank_type', 'ict_index_rank_type',
            
            # Set piece and penalty information
            'corners_and_indirect_freekicks_order', 'direct_freekicks_order', 'penalties_order',
            'corners_and_indirect_freekicks_text', 'direct_freekicks_text', 'penalties_text',
            
            # Fixture difficulty
            'chance_of_playing_next_round', 'chance_of_playing_this_round',
            'difficulty', 'difficulty_rank', 'fixture_difficulty',
            
            # Team strength and form
            'strength', 'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away', 'strength_overall_home',
            'strength_overall_away', 'team_strength', 'team_strength_overall',
            'team_strength_attack_home', 'team_strength_attack_away',
            'team_strength_defence_home', 'team_strength_defence_away',
            
            # Player stats
            'selected_by_percent', 'transfers_in', 'transfers_out',
            'transfers_in_event', 'transfers_out_event', 'transfers_balance',
            'loaned_in', 'loaned_out', 'loans_in', 'loans_out', 'loans'
        }
        
        # Patterns for historical/lagged features
        lag_patterns = (
            # Lagged features
            '_lag', '_lag1', '_lag2', '_lag3', '_lag4', '_lag5',
            
            # Rolling statistics
            'rolling_', '_roll_', '_rolling_',
            
            # Moving averages and exponential moving averages
            '_ma', '_ewm', '_ema', '_sma',
            
            # Statistical aggregations
            '_mean', '_avg', '_std', '_sum', '_min', '_max', '_median',
            '_q10', '_q25', '_q50', '_q75', '_q90',  # Quantiles
            
            # Time-based features
            'season_', 'career_', 'prev_', 'last_', 'recent_', 'last',
            
            # Derived features
            '_diff', '_ratio', '_per_90', '_per_96', '_per_game', '_rate', '_trend',
            '_pct', '_percent', '_pctg',
            
            # Player-specific
            'vs_', 'h2h_', 'against_', 'for_', 'at_home', 'away_', 'home_'
        )

        # EXCLUDE current-GW outcome columns (these cause leakage)
        current_gw_outcomes = {
            # Match events
            'minutes', 'was_home', 'starts', 'on_for_goal', 'on_for_clean_sheet',
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'own_goals',
            'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'expected_goals_conceded', 'expected_goals_per_90', 'expected_assists_per_90',
            'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90',
            'expected_goals_per_96', 'expected_assists_per_96',
            'expected_goal_involvements_per_96', 'expected_goals_conceded_per_96',
            
            # Post-match stats
            'selected', 'in_dreamteam', 'dreamteam_count', 'bps', 'influence',
            'creativity', 'threat', 'ict_index', 'value', 'transfers_balance',
            'transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event',
            'loaned_in', 'loaned_out', 'loans_in', 'loans_out', 'loans'
        }

        # First pass: include all base and static safe features
        for col in df.columns:
            # Always include base columns
            if col in base_keep:
                keep_cols.add(col)
                continue
                
            # Include static safe features
            if col in static_safe:
                keep_cols.add(col)
                continue
                
            # Skip current GW outcomes
            if col in current_gw_outcomes:
                continue
                
            # Include features matching lag patterns
            if any(pat in col.lower() for pat in lag_patterns):
                keep_cols.add(col)
        
        # Add target if requested
        if include_target and 'total_points' in df.columns:
            keep_cols.add('total_points')

        # Ensure we have some predictive features beyond IDs
        predictive_cols = keep_cols - {'player_id', 'name', 'position', 'gw'}
        if not predictive_cols:
            logger.warning(f"No predictive features found in {df.columns.tolist()}. Keeping all numeric columns.")
            # As a last resort, include all numeric columns that aren't explicitly excluded
            for col in df.select_dtypes(include=['number']).columns:
                if col not in current_gw_outcomes and col != 'total_points' or include_target:
                    keep_cols.add(col)
        
        # Log the features being used
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Selected {len(keep_cols)} safe features: {sorted(keep_cols)}")
            
            # Log which features were excluded
            excluded = set(df.columns) - keep_cols
            if excluded:
                logger.debug(f"Excluded {len(excluded)} features: {sorted(excluded)}")

        return df[list(keep_cols & set(df.columns))].copy()
    
    
    def _run_model_prediction(
        self, 
        model: MLPredictor, 
        historical_data: pd.DataFrame, 
        current_data: pd.DataFrame,
        retrain: bool = False,
        season: str = None,
        gw: int = None
    ) -> List[Dict]:
        """Run prediction using rolling train/predict split for XGBoost/LightGBM model
        
        Args:
            model: MLPredictor instance to use for predictions
            historical_data: DataFrame containing historical data for training
            current_data: DataFrame containing current gameweek data for prediction
            retrain: If True, force retraining of the model
            
        Returns:
            List of prediction dictionaries with player_id and predicted_points
        """
        
        if historical_data.empty or current_data.empty:
            return []
        
        try:
            context = f"[{season} GW{gw}]" if season and gw else ""
            logger.info(f"🤖 Running out-of-sample ML predictions {context} for {len(current_data)} players")
            
            # Normalize positions to consistent format
            pos_map_num_to_str = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            
            # Ensure target column exists for training
            if 'total_points' not in historical_data.columns:
                for alt in ['event_points', 'points', 'gw_points']:
                    if alt in historical_data.columns:
                        historical_data = historical_data.copy()
                        historical_data['total_points'] = historical_data[alt]
                        logger.debug(f"Aliased {alt} -> total_points for training consistency {context}")
                        break

            # Prepare training data (historical)
            train_df = self._filter_safe_features(historical_data.copy(), include_target=True)
            if 'position' in train_df.columns:
                train_df['position'] = train_df['position'].apply(
                    lambda v: pos_map_num_to_str.get(v, v) if not isinstance(v, str) else v
                )
            elif 'element_type' in train_df.columns:
                train_df['position'] = train_df['element_type'].map(pos_map_num_to_str).fillna('FWD')
            else:
                # Fallback - assign default positions if neither column exists
                train_df['position'] = 'FWD'
                logger.warning("No position or element_type column found in training data, using FWD default")
            
            # Prepare test data (current GW)
            test_df = self._filter_safe_features(current_data.copy(), include_target=False)
            if 'position' in test_df.columns:
                test_df['position'] = test_df['position'].apply(
                    lambda v: pos_map_num_to_str.get(v, v) if not isinstance(v, str) else v
                )
            elif 'element_type' in test_df.columns:
                test_df['position'] = test_df['element_type'].map(pos_map_num_to_str).fillna('FWD')
            else:
                # Fallback - assign default positions if neither column exists
                test_df['position'] = 'FWD'
                logger.warning("No position or element_type column found in test data, using FWD default")
            
            predictions = []
            feature_lineage = {}  # Empty for now, can be populated later
            
            # Get current gameweek for horizon
            current_gw = test_df['gw'].iloc[0] if 'gw' in test_df.columns else 4

            # Enrich test_df with latest available lag/rolling features from historical_data (up to previous GW)
            try:
                if not historical_data.empty and 'player_id' in historical_data.columns and 'gw' in historical_data.columns:
                    hist_prior = historical_data[historical_data['gw'] < current_gw].copy()
                    if not hist_prior.empty:
                        # Identify lag/rolling-derived columns present in historical data
                        join_patterns = (
                            '_lag_', '_mean_', '_rolling_', '_roll_', '_ma', '_ewm', '_ema', '_sma',
                            '_std', '_sum', '_min', '_max', '_median', '_delta', '_rate'
                        )
                        lag_cols = [
                            c for c in hist_prior.columns
                            if any(p in c for p in join_patterns) or c in ('was_starter_last_gw', 'starter_rate_3gw')
                        ]
                        # Keep only numeric lag columns to avoid merge issues
                        lag_cols = [c for c in lag_cols if pd.api.types.is_numeric_dtype(hist_prior[c])]
                        if lag_cols:
                            # For each player, take the most recent row before current_gw
                            last_hist = (
                                hist_prior.sort_values(['player_id', 'gw'])
                                .groupby('player_id', as_index=False)
                                .tail(1)
                            )
                            last_hist_sel = last_hist[['player_id'] + lag_cols].copy()
                            before_cols = set(test_df.columns)
                            test_df = test_df.merge(last_hist_sel, on='player_id', how='left')
                            # Fill any missing engineered lags with 0
                            new_cols = [c for c in test_df.columns if c not in before_cols]
                            if new_cols:
                                test_df[new_cols] = test_df[new_cols].fillna(0)
                                logger.debug(f"Augmented test_df with {len(new_cols)} lag/rolling cols from history for GW{current_gw}")
            except Exception as e:
                logger.warning(f"Failed to augment test features with lag history: {e}")
            
            # Define a helper function to align features between train and test sets
            def align_features(X_train, X_test):
                """Ensure test set has same columns as training set in same order"""
                # Get common columns and their order from training data
                common_cols = [col for col in X_train.columns if col in X_test.columns]
                
                # Reorder test columns to match training data
                X_test_aligned = X_test[common_cols].copy()
                
                # Add any missing columns with 0s
                missing_cols = set(X_train.columns) - set(common_cols)
                for col in missing_cols:
                    X_test_aligned[col] = 0.0
                
                # Ensure column order matches training data
                X_test_aligned = X_test_aligned[X_train.columns]
                return X_test_aligned
            
            # Prepare position-based training tasks
            position_tasks = []
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                pos_train = train_df[train_df['position'] == position].copy()
                pos_test = test_df[test_df['position'] == position].copy()
                
                if pos_train.empty or pos_test.empty:
                    logger.debug(f"⏭️  Skipping {position} {context} - {'no training data' if pos_train.empty else 'no test data'}")
                    continue
                
                position_tasks.append({
                    'position': position,
                    'pos_train': pos_train,
                    'pos_test': pos_test,
                    'context': context,
                    'current_gw': current_gw,
                    'should_retrain': retrain
                })
            
            # Process positions in parallel if enabled
            if self.enable_parallel and len(position_tasks) > 1:
                logger.info(f"🚀 Processing {len(position_tasks)} positions in parallel with {self.max_workers} workers")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self._train_predict_position, task, model)
                        for task in position_tasks
                    ]
                    
                    for future in futures:
                        try:
                            pos_predictions = future.result()
                            predictions.extend(pos_predictions)
                        except Exception as e:
                            logger.error(f"❌ Parallel position processing failed: {e}")
            else:
                # Sequential processing
                for task in position_tasks:
                    pos_predictions = self._train_predict_position(task, model)
                    predictions.extend(pos_predictions)
            
            logger.info(f"Generated {len(predictions)} out-of-sample XGBoost/LightGBM predictions")
            
            # Debug output for first few predictions
            if predictions and logger.isEnabledFor(logging.DEBUG):
                debug_sample = predictions[:3]
                logger.debug(f"Sample XGBoost predictions: {debug_sample}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"XGBoost/LightGBM prediction failed: {e}")
            logger.error(f"Falling back to historical average heuristic")
            
            # Fallback to simple heuristic if XGBoost fails
            predictions = []
            for _, row in current_data.iterrows():
                player_id = row['player_id']
                
                # Get historical stats for this player
                player_hist = historical_data[
                    historical_data['player_id_'] == player_id
                ] if 'player_id_' in historical_data.columns else pd.DataFrame()
                
                if not player_hist.empty:
                    # Use historical average as prediction
                    predicted_points = player_hist.get('total_points_mean', [row.get('total_points', 0)]).iloc[0]
                else:
                    # Fallback to position average
                    position_avg = current_data[
                        current_data['position'] == row['position']
                    ]['total_points'].mean()
                    predicted_points = position_avg if not pd.isna(position_avg) else 2.0
                
                predictions.append({
                    'player_id': player_id,
                    'predicted_points': float(predicted_points)
                })
            
            return predictions
    
    def _run_benchmark_prediction(
        self,
        model,
        historical_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> List[Dict]:
        """Run benchmark model prediction using proper out-of-sample train/test splits"""
        
        if historical_data.empty or current_data.empty:
            logger.warning(f"Empty data for benchmark {model.__class__.__name__}: hist={len(historical_data)}, curr={len(current_data)}")
            return self._honest_fallback_prediction(current_data, model.__class__.__name__)
        
        try:
            # Ensure target column exists for training
            if 'total_points' not in historical_data.columns:
                for alt in ['event_points', 'points', 'gw_points']:
                    if alt in historical_data.columns:
                        historical_data = historical_data.copy()
                        historical_data['total_points'] = historical_data[alt]
                        logger.debug(f"Aliased {alt} -> total_points for benchmark training consistency")
                        break
            # Filter to safe features (same as main model)
            hist_safe = self._filter_safe_features(historical_data, include_target=True)
            curr_safe = self._filter_safe_features(current_data, include_target=False)
            
            # Identify common features between train and test sets
            common_features = list(set(hist_safe.columns) & set(curr_safe.columns))
            
            # Remove non-feature columns
            non_feature_cols = {'player_id', 'position', 'gw', 'element_type', 
                              'team_id', 'name', 'web_name', 'short_name', 'total_points'}
            feature_cols = [col for col in common_features if col not in non_feature_cols]
            
            if not feature_cols:
                logger.warning(f"No valid features found for {model.__class__.__name__}")
                return self._honest_fallback_prediction(current_data, model.__class__.__name__)
                
            logger.debug(f"Using {len(feature_cols)} features for {model.__class__.__name__}: {feature_cols}")
            
            # Check for sufficient training data (out-of-sample requirement)
            if len(hist_safe) < 20:
                logger.warning(f"Insufficient training data for out-of-sample {model.__class__.__name__}: {len(hist_safe)} records")
                return self._honest_fallback_prediction(current_data, model.__class__.__name__)
            
            # Prepare training/test data (historical vs current GW)
            X_train = hist_safe[feature_cols].fillna(0)
            y_train = hist_safe['total_points'].fillna(0)
            X_test = curr_safe[feature_cols].fillna(0)
            
            # Train baseline model on historical data only
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            logger.info(f"Out-of-sample {model.__class__.__name__}: trained on {len(X_train)} historical samples, predicting {len(X_test)} current GW players")
            
            # Format predictions with actual points for evaluation
            results = []
            for i, (_, player) in enumerate(current_data.iterrows()):
                results.append({
                    'player_id': player['player_id'],
                    'name': player['name'],
                    'position': player['position'],
                    'predicted_points': max(0, predictions[i]),
                    'actual_points': player['total_points']
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Out-of-sample {model.__class__.__name__} failed: {e}, using honest fallback")
            return self._honest_fallback_prediction(current_data, model.__class__.__name__)
    
    def _fallback_benchmark_prediction(self, current_data: pd.DataFrame, model_name: str = "") -> List[Dict]:
        """Legacy fallback - kept for compatibility but redirects to honest version"""
        return self._honest_fallback_prediction(current_data, model_name)
    
    def _honest_fallback_prediction(self, current_data: pd.DataFrame, model_name: str = "") -> List[Dict]:
        """Leakage-free conservative fallback: predict small baseline points by position average."""
        if current_data.empty:
            return []
        preds = []
        df = current_data.copy()
        # Normalize position
        if 'position' in df.columns:
            pos_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4, 1: 1, 2: 2, 3: 3, 4: 4}
            df['position_norm'] = df['position'].map(pos_map).fillna(4)
        else:
            df['position_norm'] = 4
        # Use last known average (if lag exists), else static small defaults per position
        for pos_code in [1, 2, 3, 4]:
            mask = df['position_norm'] == pos_code
            default = {1: 3.0, 2: 3.5, 3: 4.0, 4: 4.0}.get(pos_code, 3.0)
            avg = default
            lag_cols = [c for c in df.columns if c.startswith('total_points_lag')]
            if lag_cols:
                avg = df.loc[mask, lag_cols].mean(axis=1).mean()
                if np.isnan(avg):
                    avg = default
            df.loc[mask, 'pred_baseline'] = avg
        for _, row in df.iterrows():
            preds.append({'player_id': row['player_id'], 'predicted_points': float(max(0.0, row.get('pred_baseline', 3.0)))})
        return preds
    
    # Removed oracle benchmark (perfect hindsight per-player actuals)
    
    def _run_random_benchmark(self, current_data: pd.DataFrame) -> List[Dict]:
        """Random baseline prediction"""
        
        np.random.seed(42)  # Reproducible randomness
        
        predictions = []
        for _, row in current_data.iterrows():
            # Random points based on position
            position = row.get('position', 4)
            # Normalize string positions to numeric codes
            if isinstance(position, str):
                position = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}.get(position, 4)
            
            if position == 1:  # GK
                random_points = np.random.normal(3.0, 2.0)
            elif position == 2:  # DEF
                random_points = np.random.normal(3.5, 2.5)
            elif position == 3:  # MID
                random_points = np.random.normal(4.0, 3.0)
            else:  # FWD
                random_points = np.random.normal(4.5, 3.5)
            
            # Ensure non-negative
            random_points = max(0, random_points)
            
            predictions.append({
                'player_id': row['player_id'],
                'predicted_points': float(random_points)
            })
        
        return predictions
    
    def _run_perfect_xi_benchmark(self, current_data: pd.DataFrame, actuals_df: pd.DataFrame | None = None) -> List[Dict]:
        """Run Perfect XI benchmark - select top 11 actual scorers under FPL constraints"""
        
        if current_data.empty:
            return []
        
        try:
            # FPL formation constraints: 1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD, total 11
            # For simplicity, use 1-4-4-2 formation
            formation = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
            
            # Enrich and normalize
            df = current_data.copy()
            df = self._enrich_with_player_metadata(df)
            df = self._add_position_norm(df)
            if 'position_norm' not in df.columns or df['position_norm'].isna().all():
                logger.warning("Could not infer positions for Perfect XI; falling back to top-11 overall")
                # Fallback to global top 11 by points
                points_col_fb = None
                if 'event_points' in df.columns and pd.to_numeric(df['event_points'], errors='coerce').fillna(0).abs().sum() > 0:
                    points_col_fb = 'event_points'
                elif 'total_points' in df.columns:
                    points_col_fb = 'total_points'
                if points_col_fb is None:
                    return []
                top_any = df.copy()
                top_any['__pts__'] = pd.to_numeric(top_any[points_col_fb], errors='coerce').fillna(0)
                top_any = top_any.sort_values('__pts__', ascending=False).head(11)
                preds = []
                for _, player in top_any.iterrows():
                    name = player.get('web_name') or player.get('name') or (f"{player.get('first_name','')} {player.get('second_name','')}".strip())
                    preds.append({'player_id': player['player_id'], 'name': name, 'position': player.get('position_norm', ''), 'predicted_points': float(player['__pts__'])})
                return preds

            # Prefer provided actuals_df['actual_points'] if available; else fallback to gw columns
            points_col = None
            if actuals_df is not None and not actuals_df.empty and 'actual_points' in actuals_df.columns:
                try:
                    actuals_small = actuals_df[['player_id', 'actual_points']].copy()
                    actuals_small['actual_points'] = pd.to_numeric(actuals_small['actual_points'], errors='coerce').fillna(0)
                    df = df.merge(actuals_small, on='player_id', how='left')
                    points_col = 'actual_points'
                except Exception:
                    points_col = None
            if points_col is None:
                # Prefer per-GW event points only when it has content; else fall back to total_points
                use_event = False
                if 'event_points' in df.columns:
                    try:
                        ev_raw = pd.to_numeric(df['event_points'], errors='coerce')
                        use_event = bool((ev_raw.notna().sum() > 0) and (ev_raw.fillna(0).abs().sum() > 0))
                    except Exception:
                        use_event = False
                if use_event:
                    points_col = 'event_points'
                elif 'total_points' in df.columns:
                    points_col = 'total_points'
                else:
                    points_col = None

            # Select top players by actual points for each position
            selected_players = []
            total_perfect_points = 0
            
            for pos, count in formation.items():
                pos_players = df[df['position_norm'] == pos].copy()
                if pos_players.empty:
                    logger.warning(f"No {pos} players found for Perfect XI")
                    continue
                
                # Prefer players who actually played (minutes > 0) when available
                if 'minutes' in pos_players.columns:
                    played = pos_players['minutes']
                    try:
                        played = pd.to_numeric(played, errors='coerce').fillna(0)
                    except Exception:
                        played = 0
                    pos_players = pos_players[played > 0]
                    if pos_players.empty:
                        # fallback to full set if filter removed all
                        pos_players = df[df['position_norm'] == pos].copy()

                # Sort by actual points descending and take top N
                if points_col is None:
                    pos_players['__pts__'] = 0.0
                else:
                    pos_players['__pts__'] = pd.to_numeric(pos_players[points_col], errors='coerce').fillna(0)
                pos_players = pos_players.sort_values('__pts__', ascending=False)
                top_players = pos_players.head(count)
                
                for _, player in top_players.iterrows():
                    player_name = player.get('web_name') or player.get('name') or (f"{player.get('first_name','')} {player.get('second_name','')}".strip())
                    val = pd.to_numeric(player.get('__pts__', 0), errors='coerce')
                    val = 0.0 if pd.isna(val) else float(val)
                    selected_players.append({
                        'player_id': player['player_id'],
                        'name': player_name,
                        'predicted_points': val,  # Perfect XI "predicts" actual per-GW points when available
                        'position': player['position_norm'],
                        'actual_points': val
                    })
                    total_perfect_points += val

            # If we didn't reach 11 due to position inference issues, fill remaining by global top scorers
            if len(selected_players) < 11:
                remaining = 11 - len(selected_players)
                chosen_ids = {sp['player_id'] for sp in selected_players}
                df['__pts__'] = pd.to_numeric(df[points_col], errors='coerce').fillna(0) if points_col else 0.0
                filler = df[~df['player_id'].isin(chosen_ids)].sort_values('__pts__', ascending=False).head(remaining)
                for _, player in filler.iterrows():
                    player_name = player.get('web_name') or player.get('name') or (f"{player.get('first_name','')} {player.get('second_name','')}".strip())
                    val = pd.to_numeric(player.get('__pts__', 0), errors='coerce')
                    val = 0.0 if pd.isna(val) else float(val)
                    selected_players.append({
                        'player_id': player['player_id'],
                        'name': player_name,
                        'predicted_points': val,
                        'position': player.get('position_norm', ''),
                        'actual_points': val
                    })
            
            logger.debug(f"Perfect XI selected {len(selected_players)} players with {total_perfect_points:.1f} total points")
            
            # Perfect XI should only be evaluated on selected players to avoid artificial penalties
            # Return only the 11 selected players for fair team-level comparison
            predictions = []
            for selected_player in selected_players:
                predictions.append({
                    'player_id': selected_player['player_id'],
                    'name': selected_player.get('name', ''),
                    'position': selected_player.get('position', ''),
                    'predicted_points': selected_player['predicted_points']
                })
            # Ensure names/positions are present (late fix for data sources missing metadata)
            season_value = None
            try:
                if 'season' in df.columns and not df['season'].isna().all():
                    season_value = str(df['season'].dropna().iloc[0])
            except Exception:
                season_value = None
            predictions = self._ensure_prediction_names(predictions, season_value, df)
            return predictions
            
        except Exception as e:
            logger.error(f"Perfect XI benchmark failed: {e}")
            # Fallback: return top 11 players by points for team-level evaluation
            try:
                top_11 = current_data.nlargest(11, 'total_points')
                return [{'player_id': row['player_id'], 'predicted_points': float(row['total_points'])} for _, row in top_11.iterrows()]
            except Exception as fallback_e:
                logger.error(f"Perfect XI fallback failed: {fallback_e}")
                return []

    def _calculate_gw_metrics(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Calculate performance metrics for a single gameweek"""
        if not predictions:
            return {}

        pred_df = pd.DataFrame(predictions).copy()
        actual_df = pd.DataFrame(actuals).copy()

        # Merge predictions with actuals
        # Drop any overlapping columns (except the join key) to avoid suffix collisions
        try:
            overlap = [c for c in pred_df.columns if c in actual_df.columns and c != 'player_id']
            if overlap:
                pred_df = pred_df.drop(columns=overlap)
        except Exception:
            pass
        merged_df = pd.merge(pred_df, actual_df, on='player_id', how='left')

        # Ensure actual_points exists post-merge to avoid KeyError
        if 'actual_points' not in merged_df.columns:
            if 'event_points' in merged_df.columns:
                merged_df['actual_points'] = merged_df['event_points']
            elif 'total_points' in merged_df.columns:
                merged_df['actual_points'] = merged_df['total_points']
            else:
                # No ground truth available; create zeros to keep function safe
                merged_df['actual_points'] = 0.0

        # Coerce to numeric and drop rows without valid values
        merged_df['predicted_points'] = pd.to_numeric(merged_df.get('predicted_points', np.nan), errors='coerce')
        merged_df['actual_points'] = pd.to_numeric(merged_df['actual_points'], errors='coerce')
        merged_df = merged_df.dropna(subset=['predicted_points', 'actual_points'])

        if merged_df.empty:
            return {}

        y_true = merged_df['actual_points'].values
        y_pred = merged_df['predicted_points'].values

        try:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            mean_bias = float(np.mean(y_pred - y_true))
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return {}

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_bias': mean_bias,
            'sample_size': int(len(merged_df)),
            'n_predictions': int(len(merged_df))
        }

    def _attach_entropy_scores(self, metrics: Dict[str, Dict]) -> None:
        """Attach normalized entropy/skill scores to per-GW metric dicts.

        The score is normalized so that:
        - perfect_xi gets 1.0
        - random gets 0.0
        - other models: 1 - (rmse_model - rmse_perfect) / (rmse_random - rmse_perfect)
          with numeric guards and clamped to [0, 1].

        Args:
            metrics: Mapping of prediction type -> metrics dict
        """
        if not isinstance(metrics, dict) or not metrics:
            return

        rnd = metrics.get('random', {}) if isinstance(metrics.get('random'), dict) else {}
        pxi = metrics.get('perfect_xi', {}) if isinstance(metrics.get('perfect_xi'), dict) else {}
        rmse_random = rnd.get('rmse')
        rmse_perfect = pxi.get('rmse', 0.0) if pxi is not None else 0.0

        # Baselines
        if 'perfect_xi' in metrics and isinstance(metrics['perfect_xi'], dict):
            metrics['perfect_xi']['entropy'] = 1.0
        if 'random' in metrics and isinstance(metrics['random'], dict):
            metrics['random']['entropy'] = 0.0

        # If random rmse is missing or invalid, we cannot compute normalized skill
        if rmse_random is None or not isinstance(rmse_random, (int, float)):
            for k, m in metrics.items():
                if isinstance(m, dict) and k not in ('perfect_xi', 'random'):
                    m['entropy'] = None
            return

        denom = max((rmse_random - (rmse_perfect or 0.0)), 1e-9)
        for k, m in metrics.items():
            if not isinstance(m, dict):
                continue
            if k in ('perfect_xi', 'random'):
                # already set
                continue
            rmse_model = m.get('rmse')
            if rmse_model is None or not isinstance(rmse_model, (int, float)):
                m['entropy'] = None
                continue
            val = 1.0 - ((rmse_model - (rmse_perfect or 0.0)) / denom)
            # clamp to [0, 1]
            m['entropy'] = float(max(0.0, min(1.0, val)))

    def _select_xi_from_predictions(self, current_data: pd.DataFrame, predictions: List[Dict]) -> List[Dict]:
        """Select a 1-4-4-2 XI from predictions under formation constraint (no team cap)."""
        if not predictions or current_data.empty:
            return []
        formation = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        pred_df = pd.DataFrame(predictions)
        df = current_data[['player_id']].copy()
        df_full = current_data.copy()
        df_full = self._enrich_with_player_metadata(df_full)
        df_full = self._add_position_norm(df_full)
        if 'position_norm' not in df_full.columns:
            return []
        df = df_full[['player_id', 'position_norm']].copy()
        m = pred_df.merge(df[['player_id', 'position_norm']], on='player_id', how='inner')
        selected = []
        for pos, count in formation.items():
            pos_players = m[m['position_norm'] == pos].sort_values('predicted_points', ascending=False)
            top_n = pos_players.head(count)
            for _, row in top_n.iterrows():
                selected.append({'player_id': row['player_id'], 'predicted_points': float(row['predicted_points'])})
        return selected

    def _add_position_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a robust 'position_norm' column mapping assorted codes to GK/DEF/MID/FWD.

        Tries multiple source columns: 'position', 'element_type', 'pos', 'position_short'.
        Accepts strings like 'GK', 'GKP', 'Goalkeeper', 'DEF', 'MID', 'FWD', or numeric 1..4.
        Unknowns are set to NaN (not forced to FWD).
        """
        if df is None or df.empty:
            return df
        if 'position_norm' in df.columns:
            return df
        df = df.copy()
        pos_series = None
        if 'position' in df.columns:
            pos_series = df['position']
        elif 'element_type' in df.columns:
            pos_series = df['element_type']
        elif 'pos' in df.columns:
            pos_series = df['pos']
        elif 'position_short' in df.columns:
            pos_series = df['position_short']
        else:
            df['position_norm'] = np.nan
            return df

        def _map_val(v: Any) -> Optional[str]:
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            # Numeric mapping
            try:
                vi = int(v)
                if vi in (1, 2, 3, 4):
                    return {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[vi]
            except Exception:
                pass
            # String mapping
            s = str(v).strip().upper()
            if s in ('GK', 'GKP', 'GOALKEEPER'):
                return 'GK'
            if s in ('DEF', 'D', 'DEFENDER', 'DF'):
                return 'DEF'
            if s in ('MID', 'M', 'MIDFIELDER', 'MF'):
                return 'MID'
            if s in ('FWD', 'FW', 'F', 'FORWARD', 'STRIKER', 'ST'):
                return 'FWD'
            return None

        df['position_norm'] = pos_series.apply(_map_val)
        return df

    def _enrich_with_player_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has player 'name' and 'position' using players table for the same season.

        - Requires 'player_id' and 'season' columns to join on.
        - Only fills missing values to avoid overwriting present data.
        """
        try:
            if df is None or df.empty or 'player_id' not in df.columns:
                return df
            if 'season' not in df.columns:
                # Cannot enrich without season context
                return df
            seasons = df['season'].dropna().unique().tolist()
            if not seasons:
                return df
            # If multiple seasons in df, enrich per season and concat
            parts: list[pd.DataFrame] = []
            for season in seasons:
                sub = df[df['season'] == season].copy()
                try:
                    meta = pd.read_sql_query(
                        "SELECT player_id, name, position FROM players WHERE season = ?",
                        self.db.conn,
                        params=(season,)
                    )
                except Exception:
                    meta = None
                # If season-specific metadata missing, fallback to any season by player_id
                if meta is None or meta.empty:
                    try:
                        ids = sub['player_id'].dropna().unique().tolist()
                        if ids:
                            placeholders = ','.join(['?'] * len(ids))
                            meta_any = pd.read_sql_query(
                                f"SELECT player_id, name, position, season FROM players WHERE player_id IN ({placeholders})",
                                self.db.conn,
                                params=ids
                            )
                            # Prefer the most recent season row per player (lexicographic max works for 'YYYY-YY')
                            if not meta_any.empty:
                                meta_any = meta_any.sort_values('season').drop_duplicates('player_id', keep='last')
                                meta = meta_any[['player_id', 'name', 'position']].copy()
                    except Exception:
                        pass
                if meta is None or meta.empty:
                    parts.append(sub)
                    continue
                merged = sub.merge(meta, on='player_id', how='left', suffixes=('', '_meta'))
                # Fill missing and coalesce empty strings
                if 'name' in merged.columns and 'name_meta' in merged.columns:
                    # Treat blanks as missing
                    merged['name'] = merged['name'].apply(lambda x: None if (isinstance(x, str) and x.strip() == '') else x)
                    merged['name'] = merged['name'].fillna(merged['name_meta'])
                    merged = merged.drop(columns=['name_meta'])
                if 'position' in merged.columns and 'position_meta' in merged.columns:
                    merged['position'] = merged['position'].apply(lambda x: None if (isinstance(x, str) and x.strip() == '') else x)
                    merged['position'] = merged['position'].fillna(merged['position_meta'])
                    merged = merged.drop(columns=['position_meta'])
                parts.append(merged)
            if parts:
                return pd.concat(parts, ignore_index=True)
            return df
        except Exception as e:
            logger.debug(f"_enrich_with_player_metadata failed: {e}")
            return df

    def _ensure_prediction_names(self, preds: List[Dict], season: Optional[str], df_context: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Fill missing 'name' and 'position' fields in prediction dicts.

        Sources, in order of preference:
        1) df_context columns: 'name', 'web_name', 'first_name'+'second_name', 'position_norm', 'position'.
        2) players table for the given season.
        3) players table across any season (most recent entry per player).
        """
        if not preds:
            return preds
        try:
            # Build context maps from df_context
            name_map: dict[int, str] = {}
            pos_map_ctx: dict[int, str] = {}
            if isinstance(df_context, pd.DataFrame) and not df_context.empty:
                tmp = df_context.copy()
                # Compose name candidates
                if 'name' not in tmp.columns and 'web_name' in tmp.columns:
                    tmp['name'] = tmp['web_name']
                if 'name' in tmp.columns:
                    tmp['name'] = tmp['name'].apply(lambda x: None if (isinstance(x, str) and x.strip() == '') else x)
                if 'name' not in tmp.columns or tmp['name'].isna().all():
                    # try first+second
                    if 'first_name' in tmp.columns and 'second_name' in tmp.columns:
                        tmp['name'] = (tmp['first_name'].astype(str).str.strip() + ' ' + tmp['second_name'].astype(str).str.strip()).str.strip()
                if 'position_norm' not in tmp.columns:
                    tmp = self._add_position_norm(tmp)
                for _, r in tmp.iterrows():
                    pid = r.get('player_id')
                    if pd.isna(pid):
                        continue
                    try:
                        pid = int(pid)
                    except Exception:
                        continue
                    nm = r.get('name')
                    if isinstance(nm, str) and nm.strip():
                        name_map[pid] = nm.strip()
                    pm = r.get('position_norm') or r.get('position')
                    if isinstance(pm, str) and pm.strip():
                        pos_map_ctx[pid] = pm.strip()

            # Fetch metadata from DB as needed
            missing_ids = [int(p['player_id']) for p in preds if not p.get('name') or str(p.get('name')).strip() == '']
            if missing_ids:
                # Try season-specific first
                meta_by_id: dict[int, dict] = {}
                if season:
                    try:
                        placeholders = ','.join(['?'] * len(missing_ids))
                        q = f"SELECT player_id, name, position FROM players WHERE season = ? AND player_id IN ({placeholders})"
                        params = [season] + missing_ids
                        meta = pd.read_sql_query(q, self.db.conn, params=params)
                        for _, r in meta.iterrows():
                            meta_by_id[int(r['player_id'])] = {'name': r.get('name'), 'position': r.get('position')}
                    except Exception:
                        pass
                # Fallback any-season for those still missing
                still_missing = [pid for pid in missing_ids if pid not in meta_by_id]
                if still_missing:
                    try:
                        placeholders = ','.join(['?'] * len(still_missing))
                        q = f"SELECT player_id, name, position, season FROM players WHERE player_id IN ({placeholders})"
                        meta_any = pd.read_sql_query(q, self.db.conn, params=still_missing)
                        if not meta_any.empty:
                            meta_any = meta_any.sort_values('season').drop_duplicates('player_id', keep='last')
                            for _, r in meta_any.iterrows():
                                meta_by_id[int(r['player_id'])] = {'name': r.get('name'), 'position': r.get('position')}
                    except Exception:
                        pass

            # Apply fills
            out: List[Dict] = []
            for p in preds:
                pid = p.get('player_id')
                try:
                    pid_i = int(pid)
                except Exception:
                    out.append(p)
                    continue
                name = p.get('name')
                if not isinstance(name, str) or not name.strip():
                    name = name_map.get(pid_i) or (meta_by_id.get(pid_i, {}).get('name') if 'meta_by_id' in locals() else None)
                pos = p.get('position')
                if not isinstance(pos, str) or not pos.strip():
                    pos = pos_map_ctx.get(pid_i) or (meta_by_id.get(pid_i, {}).get('position') if 'meta_by_id' in locals() else None)
                    # Normalize numeric positions to text
                    try:
                        pi = int(pos)
                        pos = {1:'GK',2:'DEF',3:'MID',4:'FWD'}.get(pi, pos)
                    except Exception:
                        pass
                p['name'] = name or ''
                p['position'] = pos or ''
                out.append(p)
            return out
        except Exception:
            return preds

    def _compute_xi_total_cost(self, current_data: pd.DataFrame, xi: List[Dict]) -> float:
        """Compute total XI cost (in millions) from gw 'value' or players 'now_cost'."""
        if not xi or current_data is None or current_data.empty:
            return 0.0
        df = current_data[['player_id']].copy()
        # Prefer per-GW value if available; else fall back to now_cost if present
        if 'value' in current_data.columns:
            try:
                df = current_data[['player_id', 'value']].copy()
                df['__cost__'] = pd.to_numeric(df['value'], errors='coerce') / 10.0
            except Exception:
                df['__cost__'] = 0.0
        elif 'now_cost' in current_data.columns:
            try:
                df = current_data[['player_id', 'now_cost']].copy()
                df['__cost__'] = pd.to_numeric(df['now_cost'], errors='coerce')
            except Exception:
                df['__cost__'] = 0.0
        else:
            df['__cost__'] = 0.0
        cost_map = {int(r['player_id']): float(r['__cost__']) for _, r in df[['player_id', '__cost__']].iterrows() if pd.notna(r['player_id'])}
        total = 0.0
        for item in xi:
            pid = item.get('player_id')
            try:
                pid = int(pid)
            except Exception:
                continue
            total += float(cost_map.get(pid, 0.0))
        return float(total)

    def _select_xi_from_predictions_with_budget(self, current_data: pd.DataFrame, predictions: List[Dict], budget: float = 100.0) -> List[Dict]:
        """Select a 1-4-4-2 XI with a total cost budget in £m (greedy by predicted points)."""
        if not predictions or current_data.empty:
            return []
        formation = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        pred_df = pd.DataFrame(predictions)
        # Prepare cost
        cost_col = None
        merged = current_data[['player_id', 'position']].copy()
        if 'value' in current_data.columns:
            merged['__cost__'] = pd.to_numeric(current_data['value'], errors='coerce') / 10.0
            cost_col = '__cost__'
        elif 'now_cost' in current_data.columns:
            merged['__cost__'] = pd.to_numeric(current_data['now_cost'], errors='coerce')
            cost_col = '__cost__'
        else:
            merged['__cost__'] = 5.0  # conservative default
            cost_col = '__cost__'
        # Normalize position
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD', 'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}
        merged['position_norm'] = merged['position'].map(pos_map).fillna('FWD')
        m = pred_df.merge(merged[['player_id', 'position_norm', cost_col]], on='player_id', how='inner')
        selected: List[Dict] = []
        total_cost = 0.0
        for pos, count in formation.items():
            pos_players = m[m['position_norm'] == pos].sort_values('predicted_points', ascending=False)
            for _, row in pos_players.iterrows():
                if len([s for s in selected if s.get('position') == pos]) >= count:
                    continue
                c = float(pd.to_numeric(row[cost_col], errors='coerce')) if cost_col in row else 5.0
                if total_cost + c <= budget:
                    selected.append({'player_id': row['player_id'], 'position': pos, 'predicted_points': float(row['predicted_points'])})
                    total_cost += c
                if len(selected) == 11:
                    break
        return selected

    def _run_popular_xi_benchmark(self, current_data: pd.DataFrame) -> List[Dict]:
        """Select XI by popularity using per-GW 'selected' (fallback none)."""
        if current_data is None or current_data.empty:
            return []
        if 'selected' not in current_data.columns:
            return []
        formation = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        df = current_data.copy()
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD', 'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}
        df['position_norm'] = df['position'].map(pos_map).fillna('FWD')
        selected: List[Dict] = []
        for pos, count in formation.items():
            pos_players = df[df['position_norm'] == pos].sort_values('selected', ascending=False)
            for _, row in pos_players.head(count).iterrows():
                selected.append({'player_id': row['player_id'], 'predicted_points': float(row.get('selected', 0.0))})
        return selected

    def _run_last_week_xi_benchmark(self, season: str, gw: int) -> List[Dict]:
        """Select XI as last week’s top scorers by position (uses event_points if present)."""
        if gw <= 1:
            return []
        prev = self.db.get_gameweek_data(season, gw - 1)
        if prev is None or prev.empty:
            return []
        formation = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        df = prev.copy()
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position_norm'] = df['position'].map(pos_map).fillna('FWD')
        # Determine points column
        pts_col = 'event_points' if 'event_points' in df.columns and pd.to_numeric(df['event_points'], errors='coerce').fillna(0).abs().sum() > 0 else 'total_points'
        selected: List[Dict] = []
        for pos, count in formation.items():
            pos_players = df[df['position_norm'] == pos].copy()
            if pts_col not in pos_players.columns:
                continue
            pos_players['__pts__'] = pd.to_numeric(pos_players[pts_col], errors='coerce').fillna(0)
            for _, row in pos_players.sort_values('__pts__', ascending=False).head(count).iterrows():
                selected.append({'player_id': row['player_id'], 'predicted_points': float(row['__pts__'])})
        return selected

    def _calculate_team_metrics(self, selected_predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Compute team-level metrics for a selected XI: sums and absolute error."""
        return self._calculate_gw_metrics(selected_predictions, actuals)
    
    def _compute_xi_hit_metrics(self, perfect_xi: List[Dict], selected_xi: List[Dict], actuals: List[Dict]) -> Dict:
        """Compute hit-rate metrics comparing model-selected XI to the actual Best XI.

        Metrics:
        - overlap_count: number of players common to both XIs
        - overlap_ratio: overlap_count / 11
        - model_total: sum of actual points for the model-selected XI
        - best_total: sum of actual points for the actual Best XI
        - points_gap: best_total - model_total (positive => model behind best)
        - points_gap_abs: absolute difference in totals
        """
        if not isinstance(perfect_xi, list):
            perfect_xi = []
        if not isinstance(selected_xi, list):
            selected_xi = []

        best_ids = {int(x.get('player_id')) for x in perfect_xi if 'player_id' in x}
        sel_ids = {int(x.get('player_id')) for x in selected_xi if 'player_id' in x}
        overlap = best_ids & sel_ids
        overlap_count = len(overlap)
        denom = max(len(best_ids), 11) or 11
        overlap_ratio = overlap_count / denom

        # Build actuals map
        actual_map = {}
        for a in (actuals or []):
            pid = a.get('player_id')
            if pid is None:
                continue
            try:
                pid = int(pid)
            except Exception:
                continue
            val = pd.to_numeric(a.get('actual_points', 0), errors='coerce')
            actual_map[pid] = 0.0 if pd.isna(val) else float(val)

        # Totals
        best_total = 0.0
        for x in perfect_xi:
            pid = x.get('player_id')
            if pid is None:
                continue
            try:
                pid = int(pid)
            except Exception:
                continue
            # Prefer actual_points in perfect_xi row; fallback to actual_map
            v = pd.to_numeric(x.get('actual_points', actual_map.get(pid, 0.0)), errors='coerce')
            best_total += 0.0 if pd.isna(v) else float(v)

        model_total = 0.0
        for x in selected_xi:
            pid = x.get('player_id')
            if pid is None:
                continue
            try:
                pid = int(pid)
            except Exception:
                continue
            v = actual_map.get(pid, 0.0)
            model_total += float(v)

        points_gap = float(best_total - model_total)
        return {
            'overlap_count': overlap_count,
            'overlap_ratio': float(overlap_ratio),
            'model_total': float(model_total),
            'best_total': float(best_total),
            'points_gap': points_gap,
            'points_gap_abs': float(abs(points_gap)),
        }
    
    def _calculate_season_summary(self, season_results: Dict) -> Dict:
        """Calculate summary statistics for a season"""
        
        gw_results = season_results.get('by_gameweek', {})
        if not gw_results:
            return {}
        
        # Collect all metrics for primary model
        primary_metrics = []
        for gw_data in gw_results.values():
            if 'primary' in gw_data.get('metrics', {}):
                primary_metrics.append(gw_data['metrics']['primary'])
        
        if not primary_metrics:
            return {}
        
        # Calculate averages (robust to missing keys)
        metrics_df = pd.DataFrame(primary_metrics)
        # Keep only rows where rmse exists to avoid KeyError
        if 'rmse' not in metrics_df.columns:
            return {}
        metrics_df = metrics_df.dropna(subset=['rmse'])
        if metrics_df.empty:
            return {}
        # Use get with defaults for optional columns
        mae_series = metrics_df['mae'] if 'mae' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
        r2_series = metrics_df['r2'] if 'r2' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
        bias_series = metrics_df['mean_bias'] if 'mean_bias' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
        samples_series = metrics_df['sample_size'] if 'sample_size' in metrics_df.columns else pd.Series([0] * len(metrics_df))

        return {
            'avg_rmse': float(metrics_df['rmse'].mean()),
            'avg_mae': float(mae_series.mean()),
            'avg_r2': float(r2_series.mean()),
            'avg_bias': float(bias_series.mean()),
            'total_samples': int(samples_series.sum()),
            'gameweeks_processed': int(len(metrics_df))
        }
    
    def _calculate_overall_stats(self, results: Dict) -> Dict:
        """Calculate overall statistics across all seasons"""
        
        season_results = results['results']['by_season']
        if not season_results:
            return {}
        
        # Collect all season summaries
        summaries = []
        for season_data in season_results.values():
            if season_data.get('summary'):
                summaries.append(season_data['summary'])
        
        if not summaries:
            return {}
        
        summaries_df = pd.DataFrame(summaries)
        
        return {
            'overall_rmse': float(summaries_df['avg_rmse'].mean()),
            'overall_mae': float(summaries_df['avg_mae'].mean()),
            'overall_r2': float(summaries_df['avg_r2'].mean()),
            'overall_bias': float(summaries_df['avg_bias'].mean()),
            'total_samples': int(summaries_df['total_samples'].sum()),
            'seasons_processed': len(summaries)
        }
    
    def _generate_feature_superset(self, seasons: List[str], gw_range: Tuple[int, int], sample_size: int = 5) -> List[str]:
        """Generate a superset of all safe features across sample gameweeks.
        
        Args:
            seasons: List of seasons to sample from
            gw_range: Range of gameweeks to consider
            sample_size: Number of gameweeks to sample for feature discovery
            
        Returns:
            Sorted list of all unique safe features found
        """
        all_features = set()
        
        # Sample gameweeks evenly across the range
        start_gw, end_gw = gw_range
        sample_gws = []
        if end_gw - start_gw + 1 <= sample_size:
            sample_gws = list(range(start_gw, end_gw + 1))
        else:
            # Sample evenly distributed gameweeks
            step = (end_gw - start_gw) // (sample_size - 1)
            sample_gws = [start_gw + i * step for i in range(sample_size - 1)]
            sample_gws.append(end_gw)
        
        logger.info(f"Generating feature superset from {len(seasons)} seasons, sampling GWs: {sample_gws}")
        
        for season in seasons[:2]:  # Sample first 2 seasons to avoid excessive scanning
            for gw in sample_gws:
                try:
                    # Get player data for this gameweek
                    df = self.db.get_player_data(season, gw, exclude_outliers=True)
                    if df.empty:
                        continue
                    
                    # Apply safe feature filtering
                    safe_df = self._filter_safe_features(df, include_target=False)
                    
                    # Collect numeric features, excluding identifiers and target
                    exclude_cols = {'player_id', 'gw', 'position', 'name', 'team', 'web_name', 
                                   'total_points', 'element', 'fixture', 'opponent_team'}
                    feature_cols = [col for col in safe_df.columns 
                                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(safe_df[col])]
                    
                    all_features.update(feature_cols)
                    logger.debug(f"Season {season} GW{gw}: found {len(feature_cols)} features")
                    
                except Exception as e:
                    logger.warning(f"Error sampling features from {season} GW{gw}: {e}")
                    continue
        
        superset_features = sorted(all_features)
        logger.info(f"Generated feature superset with {len(superset_features)} total features")
        
        # Save to file for persistence
        superset_path = Path('models') / 'fpl_safe_superset_features.txt'
        superset_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(superset_path, 'w') as f:
            for feat in superset_features:
                f.write(f"{feat}\n")
        
        logger.info(f"Saved feature superset to {superset_path}")
        return superset_features
    
    def _run_benchmarks(
        self, 
        results: Dict, 
        seasons: List[str], 
        gw_range: Tuple[int, int]
    ) -> Dict:
        """Calculate benchmark comparison statistics"""
        
        benchmark_stats = {}
        
        # Collect all gameweek results
        all_gw_results = []
        for season_data in results['results']['by_season'].values():
            for gw_data in season_data.get('by_gameweek', {}).values():
                all_gw_results.append(gw_data)
        
        if not all_gw_results:
            return {}
        
        # Calculate stats for each prediction type
        prediction_types = ['primary', 'decision_tree', 'random_forest', 'perfect_xi', 'random',
                            'lastweek_xi',
                            'primary_xi', 'decision_tree_xi', 'random_forest_xi']
        
        for pred_type in prediction_types:
            type_metrics = []
            for gw_result in all_gw_results:
                if pred_type in gw_result.get('metrics', {}):
                    type_metrics.append(gw_result['metrics'][pred_type])

            if not type_metrics:
                continue

            metrics_df = pd.DataFrame(type_metrics)
            # Require rmse to compute averages; skip if none available
            if 'rmse' not in metrics_df.columns:
                continue
            metrics_df = metrics_df.dropna(subset=['rmse'])
            if metrics_df.empty:
                continue

            mae_series = metrics_df['mae'] if 'mae' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
            r2_series = metrics_df['r2'] if 'r2' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
            bias_series = metrics_df['mean_bias'] if 'mean_bias' in metrics_df.columns else pd.Series([0.0] * len(metrics_df))
            entropy_series = pd.to_numeric(metrics_df['entropy'], errors='coerce') if 'entropy' in metrics_df.columns else pd.Series([], dtype=float)
            xi_overlap_series = pd.to_numeric(metrics_df['xi_overlap_ratio'], errors='coerce') if 'xi_overlap_ratio' in metrics_df.columns else pd.Series([], dtype=float)
            xi_gap_series = pd.to_numeric(metrics_df['xi_points_gap'], errors='coerce') if 'xi_points_gap' in metrics_df.columns else pd.Series([], dtype=float)
            xi_gap_abs_series = pd.to_numeric(metrics_df['xi_points_gap_abs'], errors='coerce') if 'xi_points_gap_abs' in metrics_df.columns else pd.Series([], dtype=float)

            benchmark_stats[pred_type] = {
                'avg_rmse': float(metrics_df['rmse'].mean()),
                'avg_mae': float(mae_series.mean()),
                'avg_r2': float(r2_series.mean()),
                'avg_bias': float(bias_series.mean()),
                'avg_entropy': float(entropy_series.mean()) if not entropy_series.empty else None,
                'avg_xi_overlap': float(xi_overlap_series.mean()) if not xi_overlap_series.empty else None,
                'avg_xi_points_gap': float(xi_gap_series.mean()) if not xi_gap_series.empty else None,
                'avg_xi_points_gap_abs': float(xi_gap_abs_series.mean()) if not xi_gap_abs_series.empty else None
            }
        
        return benchmark_stats
    
    def _update_rolling_stats(self, results: Dict, gw_result: Dict):
        """Update rolling statistics"""
        
        primary_metrics = gw_result.get('metrics', {}).get('primary')
        if not primary_metrics:
            return
        
        # Guard against missing keys
        rmse_val = primary_metrics.get('rmse')
        r2_val = primary_metrics.get('r2')
        sample_size_val = primary_metrics.get('sample_size')
        if rmse_val is None or r2_val is None or sample_size_val is None:
            return

        results['rolling_stats'].append({
            'timestamp': datetime.now().isoformat(),
            'season': gw_result['season'],
            'gameweek': gw_result['gameweek'],
            'rmse': rmse_val,
            'r2': r2_val,
            'sample_size': sample_size_val
        })
    
    def _save_checkpoint(self, results: Dict, checkpoint_file: Path):
        """Save intermediate results as checkpoint"""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_results(self, results: Dict):
        """Save final results"""
        
        # Save to database
        try:
            results_json = json.dumps(results, indent=2, default=str)
            
            self.db.conn.execute("""
                INSERT INTO backtest_runs 
                (run_id, timestamp, season_start, season_end, gw_start, gw_end, 
                 model_type, config, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results['run_id'],
                results['timestamp'],
                results['config']['seasons'][0] if results['config']['seasons'] else '',
                results['config']['seasons'][-1] if results['config']['seasons'] else '',
                results['config']['gw_range'][0],
                results['config']['gw_range'][1],
                results['config']['model_type'],
                json.dumps(results['config']),
                results_json
            ))
            self.db.conn.commit()
            
            logger.info(f"Results saved to database (Run ID: {results['run_id']})")
            
        except Exception as e:
            logger.error(f"Failed to save results to database: {e}")
        
        # Also save to file
        try:
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "reports"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"historical_backtest_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to file: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")
    
    def load_checkpoint(self, checkpoint_file: Path) -> Optional[Dict]:
        """Load checkpoint file"""
        try:
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return results
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        self.db.close()
