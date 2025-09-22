from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class FPLClient:
    """
    Lightweight FPL API client.

    Public endpoints require no auth (e.g., bootstrap-static, fixtures).
    Private endpoints like current team require an authenticated session cookie.
    Provide either `session_cookie` or full headers via `auth_headers`.
    """

    def __init__(
        self,
        base_url: str = "https://fantasy.premierleague.com/api",
        timeout: int = 30,
        rate_limit_delay: float = 1.0,
        retry_attempts: int = 3,
        session_cookie: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
        csrf_token: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FPL-Weekly-Updater/0.1 (+GitHub:local)",
            "Accept": "application/json",
        })
        if session_cookie:
            self.session.headers.update({"Cookie": session_cookie})
        if auth_headers:
            self.session.headers.update(auth_headers)
        # CSRF: prefer explicit token, else parse from cookie string
        token = csrf_token
        if not token and session_cookie:
            try:
                # naive parse of csrftoken from cookie header
                parts = [p.strip() for p in session_cookie.split(';')]
                for p in parts:
                    if p.lower().startswith('csrftoken='):
                        token = p.split('=', 1)[1]
                        break
            except Exception:
                token = None
        if token:
            self.session.headers.update({
                "x-csrftoken": token,
                "Referer": "https://fantasy.premierleague.com/",
            })
        self.retry_attempts = retry_attempts

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True,
           retry=retry_if_exception_type(requests.RequestException))
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # Public endpoints
    def get_bootstrap(self) -> Dict[str, Any]:
        return self._get("bootstrap-static/")

    def get_fixtures(self, event: Optional[int] = None) -> Any:
        params = {"event": event} if event else None
        return self._get("fixtures/", params=params)

    def get_element_summary(self, element_id: int) -> Dict[str, Any]:
        return self._get(f"element-summary/{element_id}/")

    # Auth-required endpoint
    def get_my_team(self, team_id: int) -> Optional[Dict[str, Any]]:
        """
        Returns squad for the given team id if authenticated; otherwise None.
        You must supply a valid FPL session cookie in settings for this to work.
        """
        try:
            return self._get(f"my-team/{team_id}/")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403, 503):
                logger.warning(
                    f"FPL get_my_team failed with status {e.response.status_code}. "
                    "Provide a valid session cookie or check API status."
                )
                return None
            raise

    # Public endpoint for entry data (does not require session cookie)
    def get_entry(self, team_id: int) -> Optional[Dict[str, Any]]:
        try:
            return self._get(f"entry/{team_id}/")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (404, 503):
                logger.warning(
                    f"FPL entry/{team_id} not found or service unavailable "
                    f"(status {e.response.status_code})."
                )
                return None
            raise
