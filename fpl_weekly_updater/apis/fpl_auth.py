from __future__ import annotations

import asyncio
import getpass
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from typing import TYPE_CHECKING, Any, Dict, Optional

import keyring
from yarl import URL

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    import aiohttp

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


def _parse_cookie_header(raw_cookie: str) -> Dict[str, str]:
    """Parse a raw cookie header string into individual cookies."""

    morsels = SimpleCookie()
    morsels.load(raw_cookie)
    return {key: morsel.value for key, morsel in morsels.items()}


def _prompt_for_session_credentials() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Interactively request session fallback credentials from the user.

    Returns:
        Tuple containing optional values for cookie, bearer token and CSRF token.
    """

    if not sys.stdin or not sys.stdin.isatty():
        logger.warning("Cannot prompt for FPL session credentials in a non-interactive environment.")
        return None, None, None

    print("FPL session credentials are required to continue.")
    print("Provide values copied from your authenticated browser session.")

    cookie = input("Session cookie (pl_profile=... or full cookie header) [leave blank to skip]: ").strip()
    bearer = getpass.getpass("API bearer token (Authorization header) [leave blank to skip]: ").strip()
    csrf = input("CSRF token (csrftoken value) [leave blank to skip]: ").strip()

    return cookie or None, bearer or None, csrf or None


def _apply_session_fallback(session: "aiohttp.ClientSession", *, interactive: bool = True) -> bool:
    """Inject cookie/bearer tokens from environment when interactive login fails."""

    cookie = (os.getenv("FPL_SESSION_COOKIE") or "").strip() or None
    bearer = (os.getenv("FPL_API_BEARER") or "").strip() or None
    csrf = (os.getenv("FPL_CSRF_TOKEN") or "").strip() or None

    if interactive and not any((cookie, bearer, csrf)):
        cookie, bearer, csrf = _prompt_for_session_credentials()
        if cookie:
            os.environ["FPL_SESSION_COOKIE"] = cookie.strip()
        if bearer:
            os.environ["FPL_API_BEARER"] = bearer.strip()
        if csrf:
            os.environ["FPL_CSRF_TOKEN"] = csrf.strip()

    applied = False

    if cookie:
        cookies_to_apply = _parse_cookie_header(cookie) if "=" in cookie else {"pl_profile": cookie}
        session.cookie_jar.update_cookies(
            cookies_to_apply,
            response_url=URL("https://users.premierleague.com/"),
        )
        applied = True

    if csrf:
        session.cookie_jar.update_cookies(
            {"csrftoken": csrf},
            response_url=URL("https://users.premierleague.com/"),
        )
        applied = True

    if bearer:
        # pylint: disable=protected-access - aiohttp exposes _default_headers for updates
        session._default_headers["Authorization"] = f"Bearer {bearer}"
        applied = True

    if applied:
        logger.info("Applied session fallback credentials from environment variables.")
    else:
        if interactive:
            logger.warning(
                "FPL login fallback requested but neither FPL_SESSION_COOKIE nor FPL_API_BEARER was provided."
            )
        else:
            logger.debug(
                "Session fallback skipped because no preconfigured credentials were available in the environment."
            )

    return applied


def _obtain_cookie_via_browser(email: str) -> Optional[str]:
    """Use the Selenium-based browser login to obtain a fresh session cookie."""

    try:
        from fpl_weekly_updater.apis.fpl_browser_login import login_and_get_cookie
    except Exception:  # pragma: no cover - defensive
        logger.debug("Browser login helper unavailable", exc_info=True)
        return None

    browser = os.getenv("FPL_BROWSER", "edge")
    headless_env = os.getenv("FPL_BROWSER_HEADLESS", "true").strip().lower()
    headless = headless_env not in {"false", "0", "no"}

    team_env = os.getenv("FPL_TEAM_ID")
    team_id = None
    try:
        if team_env:
            team_id = int(team_env)
            logger.info(f"Passing team_id={team_id} to browser login for team data extraction")
        else:
            logger.warning("FPL_TEAM_ID not set - team data extraction will be skipped")
    except ValueError:
        logger.debug("Invalid FPL_TEAM_ID '%s' provided; proceeding without it", team_env)

    try:
        cookie = login_and_get_cookie(email, headless=headless, browser=browser, team_id=team_id)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Automated browser login failed")
        return None

    if cookie:
        logger.info("Obtained session cookie via automated browser login.")
    else:
        logger.warning("Automated browser login did not return a session cookie.")

    return cookie


async def _fetch_picks_async(team_id: int, email: str, password: str, event_id: int) -> Optional[Dict[str, Any]]:
    try:
        # Lazy import to keep dependencies optional at import time
        import aiohttp
        from fpl import FPL

        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)

            login_completed = False
            try:
                await fpl.login(email, password)
                login_completed = True
            except KeyError as exc:
                logger.warning(
                    "FPL login failed because the redirect omitted the 'state' parameter (python-fpl bug).",
                )

                def _has_authenticated_cookies() -> bool:
                    try:
                        domains = (
                            "https://fantasy.premierleague.com/",
                            "https://users.premierleague.com/",
                        )
                        for domain in domains:
                            try:
                                cjar = session.cookie_jar.filter_cookies(domain)
                            except Exception:  # pragma: no cover - defensive
                                continue
                            if cjar.get("pl_profile") or cjar.get("sessionid") or cjar.get("session"):
                                return True

                        jar_dict = getattr(session.cookie_jar, "_cookies", {})
                        for _, path_map in jar_dict.items():
                            for _, cookies in path_map.items():
                                if any(name in cookies for name in ("pl_profile", "sessionid", "session")):
                                    return True
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("Cookie inspection failed", exc_info=True)
                    return False

                if _has_authenticated_cookies():
                    logger.info("Login appears successful despite missing 'state'; proceeding with existing session cookies.")
                    login_completed = True
                else:
                    logger.warning("Attempting session credential fallback because no authenticated cookies were detected.")
                    if _apply_session_fallback(session, interactive=False):
                        login_completed = True
                    else:
                        cookie_header = _obtain_cookie_via_browser(email)
                        if cookie_header:
                            os.environ["FPL_SESSION_COOKIE"] = cookie_header
                            if _apply_session_fallback(session, interactive=False):
                                login_completed = True
                        if not login_completed:
                            raise RuntimeError(
                                "FPL login failed due to missing 'state' redirect parameter and no fallback credentials provided."
                            ) from exc
            except Exception as exc:
                logger.exception("Unexpected error during FPL login; attempting credential fallback")
                if _apply_session_fallback(session):
                    login_completed = True
                else:
                    raise RuntimeError("Unable to authenticate with FPL API; login attempt raised an unexpected error.") from exc

            if not login_completed:
                raise RuntimeError("Unable to authenticate with FPL API; no credential strategy succeeded.")

            user = await fpl.get_user(team_id)
            picks = await user.get_picks(event_id)
            # Normalize to our shape similar to /my-team/
            normalized_picks = []
            for p in picks or []:
                if isinstance(p, dict):
                    normalized_picks.append({"element": p.get("element")})
                elif isinstance(p, int):
                    normalized_picks.append({"element": p})
                else:
                    logger.warning("Unexpected pick type: %s", type(p))
            norm = {"picks": normalized_picks}
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
