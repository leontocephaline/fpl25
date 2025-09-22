from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class APIConfig(BaseModel):
    base_url: str = "https://fantasy.premierleague.com/api"
    rate_limit_delay: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3


class UpdaterSettings(BaseSettings):
    # FPL
    fpl_team_id: Optional[int] = Field(default=None, env="FPL_TEAM_ID")
    fpl_session_cookie: Optional[str] = Field(default=None, env="FPL_SESSION_COOKIE")
    fpl_api_bearer: Optional[str] = Field(default=None, env="FPL_API_BEARER")
    fpl_email: Optional[str] = Field(default=None, env="FPL_EMAIL")
    fpl_csrf_token: Optional[str] = Field(default=None, env="FPL_CSRF_TOKEN")
    fpl_browser_headless: bool = Field(default=True, env="FPL_BROWSER_HEADLESS")
    fpl_browser: str = Field(default="edge", env="FPL_BROWSER")  # edge|chrome|chromium|firefox

    # Perplexity
    perplexity_api_key: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")
    perplexity_max_queries: int = Field(default=20, env="PERPLEXITY_MAX_QUERIES")
    # Perplexity model: use a valid model name. 'sonar' is broadly available.
    perplexity_model: str = Field(default="sonar", env="PERPLEXITY_MODEL")

    # Betting (Betfair)
    betfair_app_key: Optional[str] = Field(default=None, env="BETFAIR_APP_KEY")
    betfair_session_token: Optional[str] = Field(default=None, env="BETFAIR_SESSION_TOKEN")

    # General
    config_yaml_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    report_output_dir: Path = Field(default=Path.home() / "Desktop", env="REPORT_OUTPUT_DIR")
    force_model_retrain: bool = Field(default=False, env="FORCE_MODEL_RETRAIN")

    # Derived / nested
    api: APIConfig = Field(default_factory=APIConfig)

    class Config:
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"

    def to_json(self) -> str:
        payload = self.dict()
        # avoid exposing secrets
        for key in ("fpl_session_cookie", "fpl_api_bearer", "perplexity_api_key", "betfair_app_key", "betfair_session_token"):
            if payload.get(key):
                payload[key] = "***"
        return json.dumps(payload, default=str, indent=2)


def load_settings() -> UpdaterSettings:
    settings = UpdaterSettings()
    # Load API sub-config from YAML
    cfg_path = settings.config_yaml_path if settings.config_yaml_path else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        api_cfg = data.get("api", {})
        settings.api = APIConfig(**api_cfg)
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s, using API defaults", cfg_path)
    except Exception:
        logger.exception("Failed to parse config.yaml; using API defaults")
    logger.debug("Loaded settings: %s", settings.to_json())
    return settings
