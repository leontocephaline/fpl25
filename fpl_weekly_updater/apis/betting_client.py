from __future__ import annotations

import logging
from typing import Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class BettingClient:
    """
    Betting odds client.

    Primary target: Betfair API (requires app key + session token).
    If not available, you may integrate an alternative provider similarly and map
    to a common odds schema used by the scorer.
    """

    def __init__(
        self,
        betfair_app_key: Optional[str] = None,
        betfair_session_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.app_key = betfair_app_key
        self.session_token = betfair_session_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FPL-Weekly-Updater/0.1",
            "Accept": "application/json",
        })
        if self.app_key:
            self.session.headers["X-Application"] = self.app_key
        if self.session_token:
            self.session.headers["X-Authentication"] = self.session_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True,
           retry=retry_if_exception_type(requests.RequestException))
    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_configured(self) -> bool:
        return bool(self.app_key and self.session_token)

    def get_match_odds_for_team(self, team_name: str) -> Optional[Dict[str, float]]:
        """
        Placeholder: Implement Betfair market lookup for next GW fixture of `team_name`.
        Return a dict with probabilities for clean sheet, goal involvement, and cards as available.
        For now, returns None if not configured.
        """
        if not self.is_configured():
            logger.info("Betting client not configured; skipping odds for %s", team_name)
            return None
        try:
            # TODO: Implement actual Betfair API calls (listEvents -> listMarketCatalogue -> listRunnerBook)
            # Return normalized probabilities, e.g., {"clean_sheet": 0.32, "anytime_goal": 0.45, ...}
            return None
        except Exception:
            logger.exception("Error fetching betting odds for %s", team_name)
            return None
