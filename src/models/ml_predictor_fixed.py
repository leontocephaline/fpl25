# src/models/ml_predictor.py - XGBoost/LightGBM models

import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text
import matplotlib.pyplot as plt
import os
import json

# Make SHAP optional to avoid build tool requirements on Windows
try:
    import shap  # type: ignore
except Exception:  # ImportError or any runtime import issue
    shap = None  # fall back; SHAP-based analyses will be skipped with warnings

# Optional ONNX/DirectML imports
try:
    import onnx  # noqa
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    _onnx_available = True
except ImportError:
    convert_sklearn = None
    FloatTensorType = None
    ort = None
    _onnx_available = False

class MLPredictor:
    """Handles ML model training, prediction, and analysis."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        # Read params and model selection from config
        ml_cfg = self.config.get('ml', {}) if isinstance(self.config.get('ml', {}), dict) else {}
        self.model_type = ml_cfg.get('model_type', 'ensemble')  # ensemble | xgboost | lightgbm
        # Prefer nested ml.xgboost/lightgbm if present, else fall back to legacy top-level keys
        self.xgb_params = ml_cfg.get('xgboost', self.config.get('xgboost_params', {}))
        self.lgb_params = ml_cfg.get('lightgbm', self.config.get('lightgbm_params', {}))
        # Training diagnostics for XGBoost
        # Decision Tree diagnostics
        self.enable_dtree_diagnostics = ml_cfg.get('enable_dtree_diagnostics', False)
        self.dtree_max_depth = ml_cfg.get('dtree_max_depth', 2)
        self.dtree_model = None
        self.dtree_metrics = {}
        self.dtree_rules = ""
        # Prediction archival for backtesting
        self.enable_prediction_archival = ml_cfg.get('enable_prediction_archival', False)
        self.xgb_early_stopping_rounds = ml_cfg.get('xgb_early_stopping_rounds', 20)
        self.xgb_verbose_eval = ml_cfg.get('xgb_verbose_eval', False)
        self.model_save_path = Path(self.config.get('model_save_path', 'models/'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        # Inference backend: 'cpu' (default) or 'onnx'
        self.inference_backend = ml_cfg.get('inference_backend', 'cpu')
        # Capability flags
        self._shap_available = shap is not None
        self._onnx_export_available = convert_sklearn is not None
        self._onnx_export_disabled_reason = None
        # SHAP control flag (default false to reduce noise and speed up backtests)
        self.enable_shap = ml_cfg.get('enable_shap', False)

    def _onnx_providers(self) -> list:
        """Preferred execution providers for ONNX Runtime."""
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def _register_onnx_converters(self):
        """Register ONNX converters for XGBoost and LightGBM models."""
        if not _onnx_available:
            return False
            
        try:
            # Import converter modules
            from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
            from onnxmltools.convert.xgboost.shape_calculators.Regressor import calculate_linear_regressor_output_shapes as xgb_shapes
            from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
            from onnxmltools.convert.lightgbm.shape_calculators.Regressor import calculate_linear_regressor_output_shapes as lgb_shapes
            
            # Register XGBoost regressor converter
            update_registered_converter(
                xgb.XGBRegressor, "XGBoostXGBRegressor",
                xgb_shapes, convert_xgboost,
                options={"nocl": [True, False], "zipmap": [True, False, "columns"]}
            )
            
            # Register LightGBM regressor converter  
            update_registered_converter(
                lgb.LGBMRegressor, "LightGbmLGBMRegressor",
                lgb_shapes, convert_lightgbm,
                options={"nocl": [True, False], "zipmap": [True, False, "columns"]}
            )
            
            self.logger.debug("Successfully registered ONNX converters for XGBoost and LightGBM")
            return True
            
        except ImportError as e:
            self.logger.debug(f"Could not import ONNX converters: {e}")
            return False
        except Exception as e:
            self.logger.debug(f"Failed to register ONNX converters: {e}")
            return False

    def _export_to_onnx(self, model, X_sample: pd.DataFrame, save_path: Path) -> bool:
        """Export a trained sklearn-compatible model to ONNX format.
        Returns True if export succeeded."""
        if not self._onnx_export_available:
            return False
            
        # Register converters on first export attempt
        if not hasattr(self, '_converters_registered'):
            self._converters_registered = self._register_onnx_converters()
            
        try:
            initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            self.logger.debug(f"Exported ONNX model to {save_path}")
            return True
        except Exception as e:
            # Only disable after multiple failures or if converters aren't available
            if not self._converters_registered or "shape calculator" in str(e).lower():
                self._onnx_export_available = False
                self._onnx_export_disabled_reason = str(e)
                self.logger.debug(f"ONNX export not supported for this model type: {type(model).__name__}. Using CPU inference.")
            else:
                self.logger.debug(f"ONNX export failed for {save_path.name}: {e}")
            return False

    def _predict_with_onnx(self, onnx_path: Path, X: pd.DataFrame) -> np.ndarray | None:
        """Run inference using ONNX Runtime. Returns predictions or None on failure."""
        if ort is None:
            self.logger.warning("onnxruntime not available; falling back to CPU models.")
            return None
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=self._onnx_providers())
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            X_input = X.astype(np.float32).to_numpy()
            preds = sess.run([output_name], {input_name: X_input})[0]
            return preds.reshape(-1)
        except Exception as e:
            self.logger.warning(f"ONNX inference failed for {onnx_path.name}: {e}")
            return None

    def predict_player_points(self, df: pd.DataFrame, feature_lineage: dict, cleaned_stats: pd.DataFrame, horizon: int, retrain: bool = False) -> tuple[pd.DataFrame, dict]:
        """
        Orchestrates model training/loading and prediction for all positions.
        """
        self.logger.info(f"Starting ML prediction for horizon {horizon}. Retrain: {retrain}")
        
        all_predictions = []
        models_data = {}
        
        target = self.config.get('target_variable', 'total_points')
        positions = df['position'].unique()
        
        model_dir = self.model_save_path

        for position in positions:
            self.logger.info(f"Processing position: {position}")
            model_key = f"{position}_{horizon}"
            
            xgb_model_path = model_dir / f"{model_key}_xgb.joblib"
            lgbm_model_path = model_dir / f"{model_key}_lgbm.joblib"
            xgb_onnx_path = model_dir / f"{model_key}_xgb.onnx"
            lgbm_onnx_path = model_dir / f"{model_key}_lgbm.onnx"

            # If models exist and we are not retraining, load any available models per model_type
            if not retrain and (xgb_model_path.exists() or lgbm_model_path.exists()):
                self.logger.info(f"Loading saved models for {position} at horizon {horizon}...")
                if model_key not in self.models:
                    self.models[model_key] = {}
                # Load conditionally based on availability and configured model_type
                if self.model_type in ('ensemble', 'xgboost') and xgb_model_path.exists():
                    self.models[model_key]['xgb'] = joblib.load(xgb_model_path)
                if self.model_type in ('ensemble', 'lightgbm') and lgbm_model_path.exists():
                    self.models[model_key]['lgbm'] = joblib.load(lgbm_model_path)
                
                # When loading, we still need to populate models_data for reporting
                # We need to recreate test sets for SHAP values
                position_data = df[df['position'] == position]
                X = self._prepare_features(position_data)
                y = position_data[target]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Load persisted training histories if available
                xgb_hist_path = model_dir / f"{model_key}_xgb_history.json"
                lgb_hist_path = model_dir / f"{model_key}_lgbm_history.json"
                loaded_xgb_history = {}
                loaded_lgb_history = {}
                try:
                    if xgb_hist_path.exists():
                        with open(xgb_hist_path, 'r', encoding='utf-8') as f:
                            loaded_xgb_history = json.load(f)
                    if lgb_hist_path.exists():
                        with open(lgb_hist_path, 'r', encoding='utf-8') as f:
                            loaded_lgb_history = json.load(f)
                except Exception as e:
                    self.logger.info(f"Could not load training history for {model_key}: {e}")

                for model_type in [m for m in ['xgb', 'lgbm'] if m in self.models.get(model_key, {})]:
                    model = self.models[model_key][model_type]
                    # SHAP analysis only if enabled and available
                    explainer = None
                    shap_values = None
                    if self.enable_shap and self._shap_available:
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test)
                        except Exception as e:
                            self.logger.info(
                                f"Skipping SHAP for loaded {model_key}_{model_type}: {e}"
                            )

                    models_data[f'{model_key}_{model_type}'] = {
                        'model': model,
                        'X_test': X_test,
                        'y_test': y_test,
                        'history': loaded_xgb_history if model_type == 'xgb' else loaded_lgb_history,
                        'explainer': explainer,
                        'shap_values': shap_values,
                        'shap_plot_path': None  # SHAP plots are not regenerated on load
                    }

            else:
                if retrain:
                    self.logger.debug(f"'--retrain' flag is set. Forcing model retraining for {position}.")
                else:
                    self.logger.debug(f"No saved models found for {position}. Training new models...")
                
                position_data = df[df['position'] == position]
                trained_models_data = self._train_position_model(position_data, position, feature_lineage, horizon)
                models_data.update(trained_models_data)

            # Make predictions with the loaded/trained models
            if model_key in self.models and any(k in self.models[model_key] for k in ['xgb', 'lgbm']):
                position_df = df[df['position'] == position].copy()
                X_predict = self._prepare_features(position_df)
                
                xgb_model = self.models[model_key].get('xgb')
                lgbm_model = self.models[model_key].get('lgbm')
                
                # Ensure columns match training features (handle missing/extra features gracefully)
                def _align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
                    # Use pd.concat to avoid DataFrame fragmentation warnings
                    feature_data = {}
                    for f in feature_names:
                        if f in X.columns:
                            feature_data[f] = X[f]
                        else:
                            feature_data[f] = 0.0
                    return pd.DataFrame(feature_data, index=X.index)[feature_names]

                xgb_features = getattr(xgb_model, 'feature_names_in_', None) if xgb_model is not None else None
                lgb_features = getattr(lgbm_model, 'feature_names_in_', None) if lgbm_model is not None else None

                X_predict_xgb = _align_features(X_predict, list(xgb_features)) if xgb_features is not None else X_predict
                X_predict_lgb = _align_features(X_predict, list(lgb_features)) if lgb_features is not None else X_predict

                # Decide inference path
                use_onnx = (self.inference_backend == 'onnx')
                xgb_preds = None
                lgbm_preds = None

                if use_onnx:
                    # Ensure ONNX models exist or try to export from in-memory models
                    if xgb_model is not None and not xgb_onnx_path.exists():
                        self._export_to_onnx(xgb_model, X_predict_xgb.iloc[:1], xgb_onnx_path)
                    if lgbm_model is not None and not lgbm_onnx_path.exists():
                        self._export_to_onnx(lgbm_model, X_predict_lgb.iloc[:1], lgbm_onnx_path)

                    xgb_preds = self._predict_with_onnx(xgb_onnx_path, X_predict_xgb) if xgb_onnx_path.exists() else None
                    lgbm_preds = self._predict_with_onnx(lgbm_onnx_path, X_predict_lgb) if lgbm_onnx_path.exists() else None

                # Fallback to CPU sklearn models if ONNX path not available
                if xgb_model is not None and xgb_preds is None:
                    xgb_preds = xgb_model.predict(X_predict_xgb)
                if lgbm_model is not None and lgbm_preds is None:
                    lgbm_preds = lgbm_model.predict(X_predict_lgb)

                # Combine predictions based on available models
                if xgb_preds is not None and lgbm_preds is not None:
                    position_df['predicted_points'] = (xgb_preds + lgbm_preds) / 2
                    position_df['prediction_confidence'] = 1 / (1 + np.abs(xgb_preds - lgbm_preds))
                elif xgb_preds is not None:
                    position_df['predicted_points'] = xgb_preds
                    position_df['prediction_confidence'] = 1.0
                elif lgbm_preds is not None:
                    position_df['predicted_points'] = lgbm_preds
                    position_df['prediction_confidence'] = 1.0
                else:
                    self.logger.warning(f"No predictions available for {model_key}; skipping position rows.")
                    continue
                all_predictions.append(position_df)

        if not all_predictions:
            self.logger.error("No predictions were generated. Returning empty DataFrame.")
            return pd.DataFrame(), {}

        final_predictions = pd.concat(all_predictions)
        
        # Archive predictions for historical backtesting if configured
        current_gw = self.config.get('current_gameweek', horizon)
        if hasattr(self, 'enable_prediction_archival') and self.enable_prediction_archival:
            self.save_predictions_for_backtest(final_predictions, current_gw)
        
        # Log diagnostic summary if enabled
        if self.enable_dtree_diagnostics:
            self.log_diagnostic_summary()
        
        self.logger.info("ML prediction complete.")
        return final_predictions, models_data

    def _prepare_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Selects and prepares features for modeling.
        
        Args:
            df: Input DataFrame containing features
            training: If True, store the feature names for later validation
            
        Returns:
            DataFrame with prepared features
            
        Raises:
            ValueError: If no features are available or feature validation fails
        """
        if df.empty:
            self.logger.error("Empty DataFrame provided to _prepare_features")
            raise ValueError("Cannot prepare features from an empty DataFrame")
        
        # Get features to exclude from config
        features_to_exclude = self.config.get('features_to_exclude', []) + [
            self.config.get('target_variable', 'total_points'), 
            'position', 'name', 'team', 'id', 'code', 'web_name', 'team_name',
            'element_type', 'kickoff_time', 'kickoff_date', 'fixture', 'opponent_team',
            'was_home', 'round', 'season', 'fixture_id', 'player_id', 'team_id',
            'opponent_team_id', 'is_home', 'is_away', 'fixture_difficulty',
            'opponent_team_strength', 'team_strength', 'value_form', 'value_season',
            'transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event',
            'loaned_in', 'loaned_out', 'in_dreamteam', 'dreamteam_count', 'points_per_game',
            'ep_this', 'ep_next', 'event_points', 'form', 'influence', 'creativity',
            'threat', 'ict_index', 'influence_rank', 'influence_rank_type',
            'creativity_rank', 'creativity_rank_type', 'threat_rank', 'threat_rank_type',
            'ict_index_rank', 'ict_index_rank_type', 'corners_and_indirect_freekicks_order',
            'corners_and_indirect_freekicks_text', 'direct_freekicks_order',
            'direct_freekicks_text', 'penalties_order', 'penalties_text',
            'expected_goals_conceded', 'expected_goals_conceded_per_90',
            'expected_assists_per_90', 'expected_goal_involvements',
            'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90',
            'expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90',
            'goals_conceded_per_90', 'saves_per_90', 'clean_sheets_per_90', 'goals_scored_per_90',
            'assists_per_90', 'minutes_per_90', 'starts_per_90', 'bonus_per_90', 'bps_per_90',
            'influence_per_90', 'creativity_per_90', 'threat_per_90', 'ict_index_per_90',
            'total_points_per_90', 'expected_goals_per_90', 'expected_assists_per_90'
        ]
        
        # Get all numeric features excluding the ones to exclude
        numeric_features = [
            col for col in df.columns
            if (col not in features_to_exclude and 
                pd.api.types.is_numeric_dtype(df[col]) and 
                not any(excl in col for excl in ['_rank', '_text', '_order', '_date', '_time', 'id_', 'name_']))
        ]
        
        # Log feature counts
        self.logger.debug(f"Found {len(numeric_features)} numeric features out of {len(df.columns)} total columns")
        
        if not numeric_features:
            self.logger.error(f"No valid numeric features found in DataFrame. Available columns: {df.columns.tolist()}")
            raise ValueError("No valid numeric features found in the input data")
        
        # If this is training, store the expected features
        if training:
            self.expected_features = sorted(numeric_features)  # Sort for consistent ordering
            self.logger.info(f"Stored {len(self.expected_features)} expected features for model training")
            self.logger.debug(f"Expected features: {self.expected_features}")
            return df[self.expected_features]
        
        # During prediction, ensure we have all expected features
        if not hasattr(self, 'expected_features') or not self.expected_features:
            self.logger.warning("No expected features stored. Using available numeric features.")
            self.expected_features = sorted(numeric_features)
            return df[self.expected_features] if all(f in df.columns for f in self.expected_features) else df[numeric_features]
        
        # Log feature alignment status
        common_features = set(self.expected_features).intersection(df.columns)
        missing_features = set(self.expected_features) - set(df.columns)
        extra_features = set(df.columns) - set(self.expected_features)
        
        self.logger.debug(
            f"Feature alignment - Expected: {len(self.expected_features)}, "
            f"Found: {len(common_features)}, Missing: {len(missing_features)}, Extra: {len(extra_features)}"
        )
        
        # Ensure we have all expected features, fill missing with 0
        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} features in prediction data. Filling with 0.")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Missing features: {sorted(missing_features)}")
            for feature in missing_features:
                df[feature] = 0.0
        
        # Log any extra features that will be dropped
        if extra_features:
            self.logger.debug(f"Dropping {len(extra_features)} extra features not in expected features")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Extra features: {sorted(extra_features)}")
        
        # Return only the expected features in the correct order
        try:
            return df[self.expected_features]
        except KeyError as e:
            self.logger.error(f"Failed to select expected features: {str(e)}")
            self.logger.error(f"Expected features: {self.expected_features}")
            self.logger.error(f"Available features: {df.columns.tolist()}")
            raise

    def _train_position_model(self, df: pd.DataFrame, position: str, feature_lineage: dict, target_gw: int) -> dict:
        """Trains and evaluates models for a specific player position."""
        self.logger.debug(f"Training model for position: {position}")
        target = self.config.get('target_variable', 'total_points')
        X = self._prepare_features(df, training=True)  # This will store expected features
        y = df[target]

        if X.empty or len(X) < 10:
            self.logger.warning(f"Not enough data to train model for position: {position}. Skipping.")
            return {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_key = f"{position}_{target_gw}"
        if model_key not in self.models:
            self.models[model_key] = {}

        models_data = {}

        xgb_model = None
        lgb_model = None
        xgb_history = {}
        lgb_history = {}

        # Train XGBoost if enabled
        if self.model_type in ('ensemble', 'xgboost'):
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            eval_set_xgb = [(X_train, y_train), (X_test, y_test)]
            # Fit with early stopping for convergence diagnostics
            try:
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=eval_set_xgb,
                    eval_metric='rmse',
                    verbose=self.xgb_verbose_eval,
                    early_stopping_rounds=self.xgb_early_stopping_rounds,
                )
            except TypeError:
                # Fallback if xgboost version doesn't support early_stopping_rounds in sklearn API
                xgb_model.fit(X_train, y_train, eval_set=eval_set_xgb, verbose=self.xgb_verbose_eval)
            self.models[model_key]['xgb'] = xgb_model
            xgb_history = xgb_model.evals_result()
            # Log concise convergence info
            try:
                best_it = getattr(xgb_model, 'best_iteration', None)
                last_rmse = list(xgb_history.get('validation_1', {}).get('rmse', []))[-1] if 'validation_1' in xgb_history else None
                if best_it is not None and last_rmse is not None:
                    self.logger.info(f"XGB {model_key}: best_iteration={best_it}, val_rmse_last={last_rmse:.4f}")
            except Exception:
                pass
            
            # Log top feature importances
            self._log_feature_importance(xgb_model, X_train, 'XGBoost', model_key)

        # Train LightGBM if enabled
        if self.model_type in ('ensemble', 'lightgbm'):
            lgb_params = self.lgb_params.copy()
            lgb_params['verbose'] = -1  # Suppress all output
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            eval_set_lgb = [(X_test, y_test)]
            lgb_model.fit(X_train, y_train, eval_set=eval_set_lgb, callbacks=[lgb.early_stopping(10, verbose=False)])
            self.models[model_key]['lgbm'] = lgb_model
            lgb_history = lgb_model.evals_result_
            
            # Log top feature importances
            self._log_feature_importance(lgb_model, X_train, 'LightGBM', model_key)

        # Store test data for diagnostics if enabled
        if self.enable_dtree_diagnostics and (xgb_model is not None or lgb_model is not None):
            # Store test metrics for XGBoost/LightGBM models
            if xgb_model is not None:
                y_pred_test_xgb = xgb_model.predict(X_test)
                xgb_test_metrics = {
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_xgb)),
                    'test_mae': mean_absolute_error(y_test, y_pred_test_xgb),
                    'test_r2': r2_score(y_test, y_pred_test_xgb)
                }
                if model_key not in self.model_performance:
                    self.model_performance[model_key] = {}
                self.model_performance[model_key]['xgb_metrics'] = xgb_test_metrics
                
            if lgb_model is not None:
                y_pred_test_lgb = lgb_model.predict(X_test)
                lgb_test_metrics = {
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_lgb)),
                    'test_mae': mean_absolute_error(y_test, y_pred_test_lgb),
                    'test_r2': r2_score(y_test, y_pred_test_lgb)
                }
                if model_key not in self.model_performance:
                    self.model_performance[model_key] = {}
                self.model_performance[model_key]['lgb_metrics'] = lgb_test_metrics

        # Train Decision Tree for diagnostics
        if self.enable_dtree_diagnostics:
            self._train_decision_tree_diagnostic(X_train, X_test, y_train, y_test, model_key)

        # Optional ONNX export
        if self.inference_backend == 'onnx' and self._onnx_export_available:
            xgb_onnx = self.model_save_path / f"{model_key}_xgb.onnx"
            lgbm_onnx = self.model_save_path / f"{model_key}_lgbm.onnx"
            # Export attempts; internal method will silence after first failure
            if xgb_model is not None:
                self._export_to_onnx(xgb_model, X_test.iloc[:1], xgb_onnx)
            if lgb_model is not None:
                self._export_to_onnx(lgb_model, X_test.iloc[:1], lgbm_onnx)

        # SHAP Analysis and Data Collection (optional)
        report_dir = Path(self.config.get('report_path', 'reports/'))
        report_dir.mkdir(exist_ok=True, parents=True)

        for model_type, model, history in [
            item for item in [('xgb', xgb_model, xgb_history), ('lgbm', lgb_model, lgb_history)] if item[1] is not None
        ]:
            if self.enable_shap and self._shap_available:
                try:
                    explainer = None
                    shap_values = None
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)

                    # Generate SHAP summary plot with proper error handling
                    try:
                        plt.figure(figsize=(10, 6))
                        if shap_values is not None:
                            shap.summary_plot(shap_values, X_test, show=False)
                        else:
                            raise RuntimeError("SHAP values not computed")
                        shap_plot_path = os.path.join(self.config.get('plots_dir', 'plots'), f'shap_summary_{model_key}_{model_type}.png')
                        os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
                        plt.savefig(shap_plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        self.logger.info(f"SHAP plot saved to {shap_plot_path}")
                    except Exception as shap_error:
                        self.logger.info(f"Skipping SHAP plot for {model_key}_{model_type}: {shap_error}")
                        shap_plot_path = None
                        plt.close()

                    models_data[f'{model_key}_{model_type}'] = {
                        'model': model, 'X_test': X_test, 'y_test': y_test,
                        'history': history, 'explainer': explainer, 'shap_values': shap_values,
                        'shap_plot_path': shap_plot_path
                    }
                except Exception as e:
                    self.logger.info(f"Skipping SHAP analysis for {model_key}_{model_type}: {e}")
                    models_data[f'{model_key}_{model_type}'] = {
                        'model': model, 'X_test': X_test, 'y_test': y_test,
                        'history': history, 'explainer': None, 'shap_values': None, 'shap_plot_path': None
                    }
            else:
                # SHAP disabled; still record model metadata without SHAP fields
                models_data[f'{model_key}_{model_type}'] = {
                    'model': model, 'X_test': X_test, 'y_test': y_test,
                    'history': history, 'explainer': None, 'shap_values': None, 'shap_plot_path': None
                }

        if xgb_model is not None:
            self._evaluate_model(X_test, y_test, f"{model_key}_xgb")
            self._save_model(f"{model_key}_xgb")
        if lgb_model is not None:
            self._evaluate_model(X_test, y_test, f"{model_key}_lgbm")
            self._save_model(f"{model_key}_lgbm")

        # Persist histories for future diagnostics
        try:
            if xgb_model is not None:
                with open(self.model_save_path / f"{model_key}_xgb_history.json", 'w', encoding='utf-8') as f:
                    json.dump(xgb_history, f)
            if lgb_model is not None:
                with open(self.model_save_path / f"{model_key}_lgbm_history.json", 'w', encoding='utf-8') as f:
                    json.dump(lgb_history, f)
        except Exception as e:
            self.logger.info(f"Could not persist training history for {model_key}: {e}")

        # Add diagnostic comparison to models_data if enabled
        if self.enable_dtree_diagnostics and model_key in self.dtree_metrics:
            if 'diagnostic_comparison' not in models_data:
                models_data['diagnostic_comparison'] = {}
            models_data['diagnostic_comparison'][model_key] = self.get_model_diagnostics_comparison(model_key)

        return models_data

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_key: str):
        """Evaluate model performance"""
        model_group, model_type = model_key.rsplit('_', 1)
        model = self.models[model_group][model_type]
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        self.model_performance[model_key] = {'rmse': rmse}
        self.logger.info(f"Model {model_key} RMSE: {rmse:.4f}")

        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_key] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

    def _train_decision_tree_diagnostic(self, X_train, X_test, y_train, y_test, model_key):
        """Train Decision Tree for diagnostic comparison and interpretability"""
        try:
            # Train simple decision tree for transparency
            dt_model = DecisionTreeRegressor(
                max_depth=self.dtree_max_depth,
                random_state=42
            )
            dt_model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred_train_dt = dt_model.predict(X_train)
            y_pred_test_dt = dt_model.predict(X_test)
            
            # Calculate comprehensive metrics
            dtree_metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_dt)),
                'train_mae': mean_absolute_error(y_train, y_pred_train_dt),
                'train_r2': r2_score(y_train, y_pred_train_dt),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_dt)),
                'test_mae': mean_absolute_error(y_test, y_pred_test_dt),
                'test_r2': r2_score(y_test, y_pred_test_dt)
            }
            
            # Store metrics for reporting
            self.dtree_metrics[model_key] = dtree_metrics
            
            # Generate human-readable rules
            try:
                rules_text = export_text(dt_model, feature_names=list(X_train.columns), max_depth=self.dtree_max_depth)
                self.dtree_rules = rules_text
                
                # Save rules to file for reference
                rules_path = self.model_save_path / f"dtree_rules_{model_key}.txt"
                with open(rules_path, 'w') as f:
                    f.write(f"Decision Tree Rules for {model_key}\n")
                    f.write("="*50 + "\n\n")
                    f.write(rules_text)
                    f.write(f"\n\nMetrics:\n")
                    for metric, value in dtree_metrics.items():
                        f.write(f"{metric}: {value:.4f}\n")
                        
            except Exception as e:
                self.logger.warning(f"Could not export decision tree rules: {e}")
            
            # Store model for potential future use
            if model_key not in self.models:
                self.models[model_key] = {}
            self.models[model_key]['dtree_diagnostic'] = dt_model
            
            self.logger.info(f"Decision Tree diagnostic for {model_key} - Test RMSE: {dtree_metrics['test_rmse']:.4f}, R²: {dtree_metrics['test_r2']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train decision tree diagnostic for {model_key}: {e}")

    def get_model_diagnostics_comparison(self, model_key):
        """Generate side-by-side model comparison for diagnostics"""
        if not self.enable_dtree_diagnostics or model_key not in self.dtree_metrics:
            return None
            
        comparison_data = {
            'Model': ['XGBoost', 'DTree (d2)'],
            'RMSE': [None, None],
            'MAE': [None, None],
            'R²': [None, None]
        }
        
        # Get XGBoost metrics if available
        if model_key in self.model_performance and 'xgb_metrics' in self.model_performance[model_key]:
            xgb_metrics = self.model_performance[model_key]['xgb_metrics']
            comparison_data['RMSE'][0] = xgb_metrics['test_rmse']
            comparison_data['MAE'][0] = xgb_metrics['test_mae']
            comparison_data['R²'][0] = xgb_metrics['test_r2']
        
        # Get Decision Tree metrics
        dt_metrics = self.dtree_metrics[model_key]
        comparison_data['RMSE'][1] = dt_metrics['test_rmse']
        comparison_data['MAE'][1] = dt_metrics['test_mae']
        comparison_data['R²'][1] = dt_metrics['test_r2']
        
        return comparison_data

    def save_predictions_for_backtest(self, predictions_df, gameweek):
        """Archive predictions for historical backtesting"""
        try:
            # Create backtest directory
            backtest_dir = Path('data/backtest')
            backtest_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            predictions_df = predictions_df.copy()
            predictions_df['gameweek'] = gameweek
            predictions_df['prediction_date'] = pd.Timestamp.now()
            
            # Save with timestamp
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions_gw{gameweek:02d}_{timestamp}.csv'
            filepath = backtest_dir / filename
            
            predictions_df.to_csv(filepath, index=False)
            self.logger.info(f"Archived predictions for GW{gameweek} to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save predictions for backtest: {e}")
            return None

    def run_historical_backtest(self, actual_results_df, gameweek_range=None):
        """Compare historical predictions with actual FPL results"""
        try:
            backtest_dir = Path('data/backtest')
            if not backtest_dir.exists():
                self.logger.warning("No backtest data directory found")
                return None
                
            # Load all prediction files
            prediction_files = list(backtest_dir.glob('predictions_gw*.csv'))
            if not prediction_files:
                self.logger.warning("No prediction files found for backtest")
                return None
                
            all_predictions = []
            for pred_file in prediction_files:
                try:
                    pred_df = pd.read_csv(pred_file)
                    all_predictions.append(pred_df)
                except Exception as e:
                    self.logger.warning(f"Could not load {pred_file}: {e}")
                    
            if not all_predictions:
                return None
                
            # Combine all predictions
            combined_preds = pd.concat(all_predictions, ignore_index=True)
            
            # Filter by gameweek range if specified
            if gameweek_range:
                combined_preds = combined_preds[
                    combined_preds['gameweek'].between(gameweek_range[0], gameweek_range[1])
                ]
            
            # Align join key if predictions use 'id' instead of 'player_id'
            if 'id' in combined_preds.columns and 'player_id' not in combined_preds.columns:
                combined_preds = combined_preds.rename(columns={'id': 'player_id'})
            
            # Merge with actual results
            merged = combined_preds.merge(
                actual_results_df,
                on=['player_id', 'gameweek'],
                how='inner',
                suffixes=('_pred', '_actual')
            )
            
            if merged.empty:
                self.logger.warning("No matching predictions and actual results found")
                return None
                
            # Calculate backtest metrics
            merged['error'] = merged['actual_points'] - merged['predicted_points']
            merged['abs_error'] = merged['error'].abs()
            
            backtest_results = {
                'overall': {
                    'rmse': np.sqrt(mean_squared_error(merged['actual_points'], merged['predicted_points'])),
                    'mae': mean_absolute_error(merged['actual_points'], merged['predicted_points']),
                    'r2': r2_score(merged['actual_points'], merged['predicted_points']),
                    'mean_bias': merged['error'].mean(),
                    'n_predictions': len(merged)
                },
                'by_position': {},
                'by_gameweek': {},
                'drift_alerts': []
            }
            
            # Position-level analysis
            for position in merged['position'].unique():
                # Ensure JSON-serializable key type (numpy.str_ -> str)
                position_key = str(position)
                pos_data = merged[merged['position'] == position]
                if len(pos_data) > 0:
                    backtest_results['by_position'][position_key] = {
                        'rmse': np.sqrt(mean_squared_error(pos_data['actual_points'], pos_data['predicted_points'])),
                        'mae': mean_absolute_error(pos_data['actual_points'], pos_data['predicted_points']),
                        'mean_bias': pos_data['error'].mean(),
                        'n_predictions': len(pos_data)
                    }
            
            # Gameweek-level drift detection
            for gw in sorted(merged['gameweek'].unique()):
                # Ensure JSON-serializable key type (numpy.int64 -> int)
                gw_key = int(gw)
                gw_data = merged[merged['gameweek'] == gw]
                if len(gw_data) > 0:
                    gw_rmse = np.sqrt(mean_squared_error(gw_data['actual_points'], gw_data['predicted_points']))
                    gw_bias = gw_data['error'].mean()
                    backtest_results['by_gameweek'][gw_key] = {
                        'rmse': gw_rmse,
                        'mean_bias': gw_bias,
                        'n_predictions': len(gw_data)
                    }
                    
                    # Flag significant bias (>0.5 points average error)
                    if abs(gw_bias) > 0.5:
                        backtest_results['drift_alerts'].append(
                            f"GW{gw}: High bias {gw_bias:+.2f} pts/player ({len(gw_data)} predictions)"
                        )
            
            # Save backtest report
            report_path = Path('reports') / f'backtest_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(backtest_results, f, indent=2, default=str)
                
            self.logger.info(f"Backtest analysis completed. Report saved to {report_path}")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Failed to run historical backtest: {e}")
            return None

    def generate_diagnostic_report_section(self):
        """Generate diagnostic report section for inclusion in main reports"""
        if not self.enable_dtree_diagnostics or not self.dtree_metrics:
            return ""
        
        diagnostic_sections = []
        diagnostic_sections.append("# Model Diagnostics\n")
        
        # Generate comparison tables for each position model
        for model_key in self.dtree_metrics.keys():
            comparison = self.get_model_diagnostics_comparison(model_key)
            if comparison:
                diagnostic_sections.append(f"## {model_key} Model Comparison\n")
                diagnostic_sections.append("| Model        | RMSE  | MAE   | R²    |")
                diagnostic_sections.append("|--------------|-------|-------|-------|")
                
                for i, model_name in enumerate(comparison['Model']):
                    rmse = f"{comparison['RMSE'][i]:.3f}" if comparison['RMSE'][i] is not None else "N/A"
                    mae = f"{comparison['MAE'][i]:.3f}" if comparison['MAE'][i] is not None else "N/A" 
                    r2 = f"{comparison['R²'][i]:.3f}" if comparison['R²'][i] is not None else "N/A"
                    diagnostic_sections.append(f"| {model_name:<12} | {rmse:<5} | {mae:<5} | {r2:<5} |")
                diagnostic_sections.append("")
                
                # Add decision tree rules if available
                if self.dtree_rules and model_key in self.dtree_metrics:
                    diagnostic_sections.append(f"### Decision Tree Rules for {model_key}")
                    diagnostic_sections.append("```")
                    # Truncate rules for report readability
                    rules_lines = self.dtree_rules.split('\n')[:20]  # First 20 lines
                    diagnostic_sections.append('\n'.join(rules_lines))
                    if len(self.dtree_rules.split('\n')) > 20:
                        diagnostic_sections.append("... (truncated for readability)")
                    diagnostic_sections.append("```\n")
        
        return '\n'.join(diagnostic_sections)

    def log_diagnostic_summary(self):
        """Log diagnostic summary to console for immediate feedback"""
        if not self.enable_dtree_diagnostics or not self.dtree_metrics:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("MODEL DIAGNOSTICS SUMMARY")
        self.logger.info("=" * 60)
        
        for model_key, metrics in self.dtree_metrics.items():
            comparison = self.get_model_diagnostics_comparison(model_key)
            if comparison:
                self.logger.info(f"\n{model_key} Comparison:")
                xgb_rmse = comparison['RMSE'][0]
                dt_rmse = comparison['RMSE'][1]
                dt_r2 = comparison['R²'][1]

                xgb_rmse_str = f"{xgb_rmse:.3f}" if xgb_rmse is not None else "N/A"
                dt_rmse_str = f"{dt_rmse:.3f}" if dt_rmse is not None else "N/A"
                dt_r2_str = f"{dt_r2:.3f}" if dt_r2 is not None else "N/A"

                self.logger.info(f"  XGBoost  - RMSE: {xgb_rmse_str}")
                self.logger.info(f"  DTree(d2)- RMSE: {dt_rmse_str}, R²: {dt_r2_str}")
                
                # Indicate performance difference
                if (xgb_rmse is not None) and (dt_rmse is not None) and dt_rmse != 0:
                    improvement = ((dt_rmse - xgb_rmse) / dt_rmse) * 100
                    self.logger.info(f"  XGBoost improvement: {improvement:+.1f}% vs Decision Tree")
        
        self.logger.info("=" * 60)

    def _save_model(self, model_key: str):
        """Save a trained model to disk"""
        model_group, model_type = model_key.rsplit('_', 1)
        model = self.models[model_group][model_type]
        file_path = self.model_save_path / f"{model_key}.joblib"
        joblib.dump(model, file_path)
        self.logger.info(f"Saved model {model_key} to {file_path}")
    
    def fit(self, train_df: pd.DataFrame, position: str, feature_lineage: dict, horizon: int):
        """
        Train the XGBoost/LightGBM model for a given position using only historical data.
        
        Args:
            train_df: Historical training data (gameweeks < current)
            position: Player position (DEF, MID, FWD, GKP)
            feature_lineage: Feature configuration
            horizon: Current gameweek number
        """
        model_key = f"{position}_{horizon}"
        target = self.config.get('target_variable', 'total_points')
        
        if train_df.empty:
            self.logger.warning(f"No training data for {model_key}")
            return
            
        X = self._prepare_features(train_df)
        y = train_df[target]
        
        if model_key not in self.models:
            self.models[model_key] = {}
        
        # XGBoost fit
        if self.model_type in ('ensemble', 'xgboost'):
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.fit(X, y)
            self.models[model_key]['xgb'] = xgb_model
            self.logger.debug(f"Trained XGBoost for {model_key} on {len(X)} samples")
        
        # LightGBM fit
        if self.model_type in ('ensemble', 'lightgbm'):
            lgbm_model = lgb.LGBMRegressor(**self.lgb_params)
            lgbm_model.fit(X, y)
            self.models[model_key]['lgbm'] = lgbm_model
            self.logger.debug(f"Trained LightGBM for {model_key} on {len(X)} samples")
    
    def predict(self, predict_df: pd.DataFrame, position: str, horizon: int) -> pd.Series:
        """
        Generate predictions using the previously fit model for this position.
        
        Args:
            predict_df: Test data to predict on (current gameweek)
            position: Player position
            horizon: Gameweek number
            
        Returns:
            pd.Series: Predicted points for each player
            
        Raises:
            RuntimeError: If no model is available for the specified position and horizon
        """
        model_key = f"{position}_{horizon}"
        
        if model_key not in self.models:
            raise RuntimeError(f"No model available for {model_key} - call fit() first")
        
        # Log the features we're using for prediction
        self.logger.debug(f"Generating predictions for {model_key} with {len(predict_df)} players")
            
        try:
            # Prepare features - this will handle missing features by filling with 0
            X_pred = self._prepare_features(predict_df)
            self.logger.debug(f"Using {len(X_pred.columns)} features for prediction")
            
            xgb_model = self.models[model_key].get('xgb')
            lgbm_model = self.models[model_key].get('lgbm')
            
            # Generate predictions from available models
            predictions = []
            model_names = []
            
            if xgb_model is not None:
                xgb_preds = xgb_model.predict(X_pred)
                predictions.append(xgb_preds)
                model_names.append('XGBoost')
                
            if lgbm_model is not None:
                lgbm_preds = lgbm_model.predict(X_pred)
                predictions.append(lgbm_preds)
                model_names.append('LightGBM')
            
            if not predictions:
                raise RuntimeError(f"No valid models available for {model_key}")
            
            # Average predictions if we have multiple models
            avg_predictions = np.mean(predictions, axis=0)
            
            self.logger.debug(f"Generated predictions using {len(model_names)} models: {', '.join(model_names)}")
            return pd.Series(avg_predictions, index=predict_df.index)
            
        except Exception as e:
            self.logger.error(f"Error during prediction for {model_key}: {str(e)}")
            self.logger.debug(f"Features available: {', '.join(predict_df.columns.tolist())}")
            if hasattr(self, 'expected_features'):
                self.logger.debug(f"Expected features: {', '.join(self.expected_features)}")
            raise
    
    def _log_feature_importance(self, model, X_train: pd.DataFrame, model_type: str, model_key: str, top_n: int = 15):
        """Log top-N feature importances for the trained model"""
        try:
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_train.columns
            else:
                self.logger.warning(f"Model {model_type} for {model_key} does not have feature_importances_ attribute")
                return
                
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Log top-N features at INFO level
            top_features = importance_df.head(top_n)
            self.logger.info(f"🔍 Top {top_n} features for {model_type} {model_key}:")
            for idx, (_, row) in enumerate(top_features.iterrows(), 1):
                self.logger.info(f"  {idx:2d}. {row['feature']:<25} | {row['importance']:.4f}")
                
            # Log summary statistics at INFO level
            total_importance = importance_df['importance'].sum()
            top_10_importance = importance_df.head(10)['importance'].sum()
            top_5_importance = importance_df.head(5)['importance'].sum()
            
            self.logger.info(f"📊 Feature importance summary for {model_type} {model_key}:")
            self.logger.info(f"   Top 5 features: {top_5_importance/total_importance:.1%} of total importance")
            self.logger.info(f"   Top 10 features: {top_10_importance/total_importance:.1%} of total importance")
            self.logger.info(f"   Total features: {len(feature_names)}")
            
            # Also log the full feature importance to a file for debugging
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            importance_file = log_dir / f"feature_importance_{model_type}_{model_key}.csv"
            importance_df.to_csv(importance_file, index=False)
            self.logger.info(f"Full feature importance saved to {importance_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log feature importance for {model_type} {model_key}: {e}")
