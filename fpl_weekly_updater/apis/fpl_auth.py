from __future__ import annotations

import asyncio
import json
import logging
import os
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


def fetch_picks(team_id: int, email: str, password: str, event_id: int) -> Optional[Dict[str, Any]]:
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
    if os.path.exists(TEAM_DATA_FILE):
        try:
            with open(TEAM_DATA_FILE, 'r') as f:
                team_data = json.load(f)
                if 'picks' in team_data:
                    logger.info("Using team data extracted from page")
                    return team_data
        except Exception as e:
            logger.warning(f"Failed to load extracted team data: {e}")
    
    # Fall back to API if extraction failed or no data file exists
    logger.info("Falling back to FPL API for team data")
    return asyncio.run(_fetch_picks_async(team_id, email, password, event_id))
