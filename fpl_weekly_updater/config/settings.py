from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


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
        env_file = DEFAULT_ENV_PATH
        env_file_encoding = "utf-8"

    def to_json(self) -> str:
        payload = self.dict()
        # avoid exposing secrets
        for key in ("fpl_session_cookie", "fpl_api_bearer", "perplexity_api_key", "betfair_app_key", "betfair_session_token"):
            if payload.get(key):
                payload[key] = "***"
        return json.dumps(payload, default=str, indent=2)


def _resolve_env_path(default_path: Path | None = None) -> Optional[Path]:
    """Resolve the appropriate .env path for secrets."""

    candidates: list[Path] = []

    override = os.getenv("ENV_FILE_PATH")
    if override:
        candidates.append(Path(override))

    if default_path:
        candidates.append(default_path)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / ".env")

    if getattr(sys, "frozen", False):
        try:
            candidates.append(Path(sys.executable).resolve().parent / ".env")
        except Exception:
            pass

    try:
        candidates.append(Path.cwd() / ".env")
    except Exception:
        pass

    for candidate in candidates:
        try:
            if candidate and candidate.exists():
                return candidate
        except Exception:
            continue

    return None


def _resolve_config_path(default_path: Path) -> Path:
    """Resolve the most appropriate config.yaml path.

    Search order:
    1. Explicit CONFIG_YAML_PATH environment variable.
    2. Pydantic-provided default (usually project root when running from source).
    3. PyInstaller extraction directory (`_MEIPASS`).
    4. Directory containing the executable (supports placing a symlink next to the binary).
    5. Current working directory.
    """

    candidates: list[Path] = []

    env_path = os.getenv("CONFIG_YAML_PATH")
    if env_path:
        candidates.append(Path(env_path))

    if default_path:
        candidates.append(default_path)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "config.yaml")

    if getattr(sys, "frozen", False):  # Running under PyInstaller
        try:
            candidates.append(Path(sys.executable).resolve().parent / "config.yaml")
        except Exception:
            pass

    try:
        candidates.append(Path.cwd() / "config.yaml")
    except Exception:
        pass

    for candidate in candidates:
        try:
            if candidate and candidate.exists():
                return candidate
        except Exception:
            continue

    return default_path


def load_settings() -> UpdaterSettings:
    env_path = _resolve_env_path(DEFAULT_ENV_PATH)
    if env_path:
        settings = UpdaterSettings(_env_file=str(env_path))
    else:
        settings = UpdaterSettings()
    # Load API sub-config from YAML
    resolved_cfg_path = _resolve_config_path(settings.config_yaml_path or DEFAULT_CONFIG_PATH)
    settings.config_yaml_path = resolved_cfg_path

    cfg_path = resolved_cfg_path
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
