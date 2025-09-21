# src/utils/config.py - Configuration and constants
"""
Configuration Manager - Handles all system configuration and constants
Provides centralized configuration management with validation and defaults
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class Config:
    """Configuration manager for FPL Optimizer"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()
        self._validate_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {str(e)}")
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'system': {
                'log_level': 'INFO',
                'log_file': 'fpl_optimizer.log',
                'log_dir': './logs',
                'log_max_size_mb': 10,
                'log_backup_count': 5,
                'log_console': True,
                'log_to_file': True,
                'log_format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'log_datefmt': '%Y-%m-%d %H:%M:%S',
                'optimization_horizon': 4,
                'model_save_path': './data/models/',
                'data_cache_path': './data/processed/'
            },
            'api': {
                'base_url': 'https://fantasy.premierleague.com/api',
                'rate_limit_delay': 1.0,
                'timeout': 30,
                'retry_attempts': 3
            },
            'optimization': {
                'budget': 100.0,
                'squad_size': 15,
                'max_per_team': 3,
                'bench_weights': [0.2, 0.1, 0.05]
            },
            'strategy': {
                'premium_limit': 2,
                'premium_thresholds': {'GK': 5.5, 'DEF': 6.0, 'MID': 8.0, 'FWD': 8.0}
            }
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        self._config = default_config
    
    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['system', 'api', 'optimization', 'strategy']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate specific values
        if self._config['optimization']['budget'] <= 0:
            raise ValueError("Budget must be positive")
        
        if self._config['optimization']['squad_size'] != 15:
            raise ValueError("Squad size must be 15")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    # Provide mapping-like access for tests that expect dict-style behavior
    def __getitem__(self, key: str):
        return self._config.get(key)

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    @property
    def config(self) -> Dict[str, Any]:
        """Expose the underlying dict for direct assignment in tests."""
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        if not isinstance(value, dict):
            raise ValueError("Config.config must be set to a dictionary")
        self._config = value
    
    # Convenience properties for commonly used values
    @property
    def log_level(self) -> str:
        return self.get('system.log_level', 'INFO')
    
    @property
    def optimization_horizon(self) -> int:
        return self.get('system.optimization_horizon', 4)
    
    @property
    def model_save_path(self) -> str:
        return self.get('system.model_save_path', './data/models/')
    
    @property
    def api_base_url(self) -> str:
        return self.get('api.base_url', 'https://fantasy.premierleague.com/api')
    
    @property
    def api_rate_limit(self) -> float:
        return self.get('api.rate_limit_delay', 1.0)
    
    @property
    def budget(self) -> float:
        return self.get('optimization.budget', 100.0)
    
    @property
    def squad_size(self) -> int:
        return self.get('optimization.squad_size', 15)
    
    @property
    def premium_limit(self) -> int:
        return self.get('strategy.premium_limit', 2)
    
    @property
    def premium_thresholds(self) -> Dict[str, float]:
        return self.get('strategy.premium_thresholds', {'GK': 5.5, 'DEF': 6.0, 'MID': 8.0, 'FWD': 8.0})
    
    @property
    def bench_weights(self) -> list:
        return self.get('optimization.bench_weights', [0.2, 0.1, 0.05])
    
    def get_ml_params(self, model_type: str) -> Dict:
        """Get ML model parameters"""
        return self.get(f'ml.{model_type}', {})
    
    def get_formation_preferences(self) -> Dict:
        """Get formation preferences"""
        return self.get('optimization.formations', {
            '3-5-2': {'weight': 1.0, 'priority': 1},
            '3-4-3': {'weight': 0.8, 'priority': 2}
        })

# src/utils/logger.py - Logging setup
"""
Logger Setup - Configures logging for the FPL Optimizer system
Provides structured logging with file and console output
"""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "FPLOptimizer", level: str = None, config: Config = None) -> logging.Logger:
    """
    Set up structured logging for the application with configurable options.
    
    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional Config instance for custom settings
        
    Returns:
        Configured logger instance
    """
    # Get configuration
    if config is None:
        config = Config()
        
    # Get log level from config if not specified
    if level is None:
        level = config.get('system.log_level', 'INFO')
    
    # Create logs directory
    log_dir = Path(config.get('system.log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file settings
    log_file = log_dir / config.get('system.log_file', 'fpl_optimizer.log')
    max_bytes = config.get('system.log_max_size_mb', 10) * 1024 * 1024  # Convert MB to bytes
    backup_count = config.get('system.log_backup_count', 5)
    log_console = config.get('system.log_console', True)
    log_to_file = config.get('system.log_to_file', True)
    log_format = config.get('system.log_format', 
                          '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    log_datefmt = config.get('system.log_datefmt', '%Y-%m-%d %H:%M:%S')
    
    # Configure logging
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': log_format,
                'datefmt': log_datefmt
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d: %(message)s',
                'datefmt': log_datefmt
            }
        },
        'handlers': {},
        'loggers': {
            name: {
                'level': 'DEBUG',
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': []
        }
    }
    
    # Add console handler if enabled
    if log_console:
        log_config['handlers']['console'] = {
            'level': level,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': sys.stdout
        }
        log_config['loggers'][name]['handlers'].append('console')
        log_config['root']['handlers'].append('console')
    
    # Add file handler if enabled
    if log_to_file:
        log_config['handlers']['file'] = {
            'level': 'DEBUG',  # File handler captures all levels
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'encoding': 'utf8'
        }
        log_config['loggers'][name]['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(log_config)
    
    # Get logger instance
    logger = logging.getLogger(name)
    
    # Log startup information at DEBUG level to avoid cluttering normal operation
    logger.debug("Logging configuration:")
    logger.debug(f"  Log level: {level}")
    logger.debug(f"  Log file: {log_file}")
    logger.debug(f"  Max file size: {max_bytes/1024/1024}MB")
    logger.debug(f"  Backup count: {backup_count}")
    logger.debug(f"  Console logging: {'enabled' if log_console else 'disabled'}")
    logger.debug(f"  File logging: {'enabled' if log_to_file else 'disabled'}")
    
    # Log startup at INFO level
    logger.info(f"FPL Optimizer logging initialized - Logger: {name}")
    logger.info(f"Log level set to: {level}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)

# Performance logging decorator
def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    
    return wrapper

# src/utils/helpers.py - Utility functions
"""
Helper Functions - Common utility functions used across the FPL Optimizer
Provides data manipulation, validation, and formatting utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def normalize_values(values: Union[List[float], np.ndarray], 
                    min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize values to specified range"""
    values = np.array(values)
    
    if len(values) == 0:
        return values
    
    val_min, val_max = values.min(), values.max()
    
    if val_max == val_min:
        return np.full_like(values, (min_val + max_val) / 2)
    
    normalized = (values - val_min) / (val_max - val_min)
    return normalized * (max_val - min_val) + min_val

def calculate_rolling_average(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Calculate rolling average with minimum periods"""
    return series.rolling(window=window, min_periods=min_periods).mean()

def get_position_name(element_type: int) -> str:
    """Convert element type to position name"""
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    return position_map.get(element_type, 'Unknown')

def get_element_type(position: str) -> int:
    """Convert position name to element type"""
    position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    return position_map.get(position.upper(), 0)

def format_currency(amount: float, symbol: str = '£', decimals: int = 1) -> str:
    """Format currency amount"""
    return f"{symbol}{amount:.{decimals}f}m"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage value"""
    return f"{value:.{decimals}f}%"

def calculate_fixture_difficulty(home_strength: float, away_strength: float, 
                               is_home: bool = True) -> float:
    """Calculate fixture difficulty score"""
    if is_home:
        difficulty = (away_strength - home_strength) / 2 + 3
    else:
        difficulty = (home_strength - away_strength) / 2 + 3
    
    return max(1, min(5, difficulty))

def validate_player_data(player: Dict) -> bool:
    """Validate player data structure"""
    required_fields = ['id', 'web_name', 'element_type', 'team', 'now_cost']
    
    try:
        for field in required_fields:
            if field not in player or player[field] is None:
                return False
        
        # Validate data types
        if not isinstance(player['id'], int):
            return False
        if not isinstance(player['element_type'], int) or player['element_type'] not in [1, 2, 3, 4]:
            return False
        if not isinstance(player['now_cost'], (int, float)) or player['now_cost'] <= 0:
            return False
        
        return True
    except Exception:
        return False

def filter_available_players(players_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only available players (not injured, suspended, etc.)"""
    # Basic availability filters
    available = players_df[
        (players_df['status'] == 'a') &  # Available
        (players_df['team'] > 0) &       # Has a team
        (players_df['now_cost'] > 0)     # Has a valid price
    ].copy()
    
    return available

def create_player_summary(player: Dict) -> Dict:
    """Create summary information for a player"""
    return {
        'id': player.get('id'),
        'name': player.get('web_name', ''),
        'position': get_position_name(player.get('element_type', 0)),
        'team': player.get('team_name', ''),
        'cost': format_currency(player.get('now_cost', 0)),
        'points': player.get('total_points', 0),
        'form': player.get('form', '0'),
        'ownership': format_percentage(player.get('selected_by_percent', 0.0)),
    }
