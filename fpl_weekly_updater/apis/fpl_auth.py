from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import keyring

logger = logging.getLogger(__name__)

SERVICE_NAME = "fpl-weekly-updater"
TEAM_DATA_FILE = "team_data.json"


def get_password(email: str) -> Optional[str]:
    """Retrieve password from OS keyring for the given email."""
    try:
        return keyring.get_password(SERVICE_NAME, email)
    except Exception:
        logger.exception("Failed to read password from keyring")
        return None


async def _fetch_picks_async(team_id: int, email: str, password: str, event_id: int) -> Optional[Dict[str, Any]]:
    try:
        # Lazy import to keep dependencies optional at import time
        import aiohttp
        from fpl import FPL

        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(email, password)
            user = await fpl.get_user(team_id)
            picks = await user.get_picks(event_id)
            # Normalize to our shape similar to /my-team/
            norm = {"picks": [{"element": p.get("element")} for p in picks or []]}
            return norm
    except Exception:
        logger.exception("fpl library fetch failed")
        return None


def _is_team_file_fresh(team_data: Dict[str, Any], file_path: str, max_age_hours: int) -> bool:
    try:
        # Prefer explicit timestamp inside JSON if present
        ts_str = team_data.get("picks_last_updated")
        if ts_str:
            try:
                file_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                file_dt = None
        else:
            file_dt = None

        if file_dt is None:
            # Fall back to filesystem modified time
            file_dt = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)

        age = datetime.now(timezone.utc) - file_dt
        return age <= timedelta(hours=max_age_hours)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to validate freshness of %s: %s", file_path, exc)
        return False


def fetch_picks(
    team_id: int,
    email: str,
    password: str,
    event_id: int,
    *,
    max_file_age_hours: int = 24,
) -> Optional[Dict[str, Any]]:
    """
    Fetch team picks either via the FPL API or from extracted page data.
    
    Args:
        team_id: FPL team ID
        email: FPL account email
        password: FPL account password
        event_id: Gameweek number
        
    Returns:
        Team picks data or None if not available
    """
    # First try to load from extracted data if available
    stale_team_data: Optional[Dict[str, Any]] = None
    if os.path.exists(TEAM_DATA_FILE):
        try:
            with open(TEAM_DATA_FILE, 'r') as f:
                team_data = json.load(f)
                if 'picks' in team_data and _is_team_file_fresh(team_data, TEAM_DATA_FILE, max_file_age_hours):
                    logger.info("Using recently extracted team data from %s", TEAM_DATA_FILE)
                    return team_data
                elif 'picks' in team_data:
                    logger.info("Stale team data detected; refreshing via FPL API")
                    stale_team_data = team_data
        except Exception as e:
            logger.warning(f"Failed to load extracted team data: {e}")
    
    # Fall back to API if extraction failed or no data file exists
    logger.info("Falling back to FPL API for team data")
    fresh_data = asyncio.run(_fetch_picks_async(team_id, email, password, event_id))
    if fresh_data:
        return fresh_data

    if stale_team_data:
        logger.warning(
            "Returning stale team data from %s because live refresh failed.",
            TEAM_DATA_FILE,
        )
        return stale_team_data

    return None
