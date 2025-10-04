from __future__ import annotations

import json
import logging
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class PerplexityClient:
    """
    Minimal Perplexity API client wrapper for research queries.
    Note: You'll need an API key and to manage budget usage. We expose `max_queries`
    and a simple token/requests budget cap to avoid overspending.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = "https://api.perplexity.ai",
        timeout: int = 30,
        max_queries: int = 20,
        model: str = "sonar",
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl_hours: int = 168,  # 1 week default TTL
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FPL-Weekly-Updater/0.1",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._queries_made = 0
        self.max_queries = max_queries
        # Coerce unsupported models to a broadly available default
        allowed = {"sonar", "sonar-pro"}
        if model not in allowed:
            logger.warning("Perplexity model '%s' not supported; using 'sonar'", model)
            self.model = "sonar"
        else:
            self.model = model
        
        # Initialize caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "fpl_perplexity"
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache: Dict[str, dict] = {}
        self._load_cache()
            
        # Define the response schema for player analysis
        self.response_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "injury_status": {"type": "string"},
                "expected_return": {"type": ["string", "null"]},
                "summary": {"type": "string"},
                "predicted_points": {"type": "number"},
                "predicted_minutes": {"type": "number"},
                "predicted_goals": {"type": "number"},
                "predicted_assists": {"type": "number"},
                "predicted_clean_sheet": {"type": "boolean"},
                "next_fixture": {"type": "string"},
                "fixture_difficulty": {"type": "number"},
                "form": {"type": "number"},
                "minutes_played": {"type": "number"},
                "goals_scored": {"type": "number"},
                "assists": {"type": "number"},
                "clean_sheets": {"type": "number"},
                "yellow_cards": {"type": "number"},
                "red_cards": {"type": "number"},
                "bonus_points": {"type": "number"},
                "bps": {"type": "number"},
                "influence": {"type": "number"},
                "creativity": {"type": "number"},
                "threat": {"type": "number"},
                "ict_index": {"type": "number"},
                "news": {"type": "string"},
                "news_added": {"type": "string"},
                "chance_of_playing_next_round": {"type": "number"},
                "chance_of_playing_this_round": {"type": "number"},
                "value_form": {"type": "string"},
                "value_season": {"type": "string"},
                "points_per_game": {"type": "string"},
                "ep_this": {"type": "number"},
                "ep_next": {"type": "number"},
                "dreamteam_count": {"type": "number"},
                "in_dreamteam": {"type": "boolean"},
                "form_rank": {"type": "number"},
                "form_rank_type": {"type": "number"},
                "points_per_game_rank": {"type": "number"},
                "points_per_game_rank_type": {"type": "number"},
                "selected_by_percent": {"type": "string"},
                "transfers_in_event": {"type": "number"},
                "transfers_out_event": {"type": "number"},
                "transfers_in": {"type": "number"},
                "transfers_out": {"type": "number"},
                "cost_change_event": {"type": "number"},
                "cost_change_event_fall": {"type": "number"},
                "cost_change_start": {"type": "number"},
                "cost_change_start_fall": {"type": "number"},
                "now_cost": {"type": "number"},
                "now_cost_rank": {"type": "number"},
                "now_cost_rank_type": {"type": "number"},
                "influence_rank": {"type": "number"},
                "influence_rank_type": {"type": "number"},
                "creativity_rank": {"type": "number"},
                "creativity_rank_type": {"type": "number"},
                "threat_rank": {"type": "number"},
                "threat_rank_type": {"type": "number"},
                "ict_index_rank": {"type": "number"},
                "ict_index_rank_type": {"type": "number"}
            },
            "required": ["status", "summary"]
        }

    def analyze_player(self, player_name: str, context: Dict[str, Any] = None, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed analysis for a player including status, injury info, and predictions.
        Returns structured data that can be used for team selection and transfer decisions.
        
        Args:
            player_name: Name of the player to analyze
            context: Additional context for the analysis
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            Dict containing player analysis or None if analysis fails
        """
        if not self.can_query():
            logger.warning("Cannot query Perplexity API - rate limit or budget reached")
            return None
            
        # Create a unique cache key based on player name and context
        cache_key_data = {
            'player': player_name.lower(),
            'context': context or {}
        }
        cache_key = f"player_analysis_{self._get_cache_key(cache_key_data)}"
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"Using cached data for {player_name}")
                return cached_data
        
        logger.debug(f"Fetching fresh data for {player_name}")
        
        try:
            # Build a comprehensive prompt with context
            prompt = self._build_analysis_prompt(player_name, context)
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            response = self._post("/chat/completions", payload)
            
            # Parse and validate the response
            if not response:
                logger.warning("Empty response from Perplexity API")
                return self._create_default_response(player_name)
                
            if 'choices' not in response or not response['choices'] or len(response['choices']) == 0:
                logger.warning("No choices in Perplexity API response")
                return self._create_default_response(player_name)
                
            message = response['choices'][0].get('message', {})
            content = message.get('content')
            
            if not content:
                logger.warning("No content in Perplexity API response")
                return self._create_default_response(player_name)
                
            try:
                # Try to extract JSON from markdown code blocks first
                json_match = None
                if '```json' in content:
                    import re
                    json_match = re.search(r'```json\s*({[^`]+})\s*```', content, re.DOTALL)
                elif '```' in content:
                    import re
                    json_match = re.search(r'```\s*({[^`]+})\s*```', content, re.DOTALL)
                
                if json_match:
                    content = json_match.group(1)
                
                # Try to parse as JSON
                result = json.loads(content)
                if not isinstance(result, dict):
                    logger.warning("Parsed JSON is not a dictionary")
                    result = self._extract_info_from_text(content, player_name)
            except json.JSONDecodeError:
                # If not JSON, extract key information from text
                result = self._extract_info_from_text(content, player_name)
            
            # Ensure we have required fields
            if not isinstance(result, dict):
                result = {}
            
            # Add default values for required fields if missing
            if 'summary' not in result or not result['summary']:
                result['summary'] = f"No recent updates available for {player_name}."
            if 'status' not in result:
                result['status'] = 'unknown'
            
            # Add timestamp
            result['last_updated'] = datetime.now().isoformat()
                
            # Save to cache if we got a valid result
            if result.get('summary') and result['summary'] != f"No recent updates available for {player_name}.":
                self._save_to_cache(cache_key, result)
            
            return result
                        
        except Exception as e:
            logger.error(f"Error analyzing player {player_name}: {str(e)}")
            
        # If we get here, something went wrong
        return self._create_default_response(player_name)

    def get_basic_player_info(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Get basic player information with minimal API usage"""
        if not self.can_query():
            return None
            
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": f"Provide a brief one-sentence summary of {player_name}'s current status and team."}],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            # For now, let's remove response_format to get basic functionality working
            # The Perplexity API seems to have specific requirements we need to investigate
            # if self.model in ["sonar", "sonar-pro"]:
            #     payload["response_format"] = {
            #         "type": "json_schema",
            #         "json_schema": {
            #             "name": "basic_player_info",
            #             "schema": {
            #                 "type": "object",
            #                 "properties": {
            #                     "summary": {"type": "string"},
            #                     "status": {"type": "string"},
            #                     "team": {"type": "string"}
            #                 },
            #                 "required": ["summary", "status"]
            #             }
            #         }
            #     }
            
            response = self._post("/chat/completions", payload)
            
            if response and 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0].get('message', {}).get('content')
                if content:
                    try:
                        # Try to parse as JSON first
                        result = json.loads(content)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        pass
                    
                    # If not JSON, use the text content
                    return {"summary": content, "status": "Unknown", "team": "Unknown"}
            
            return {"summary": f"No information available for {player_name}", "status": "Unknown", "team": "Unknown"}
                        
        except Exception as e:
            logger.error(f"Error getting basic info for {player_name}: {str(e)}")
            return {"summary": f"Error retrieving information for {player_name}."}
        
    def _create_default_response(self, player_name: str) -> Dict[str, Any]:
        """Create a default response when API call fails or returns unexpected data"""
        return {
            "summary": f"No recent updates available for {player_name}.",
            "status": "unknown",
            "last_updated": datetime.now().isoformat(),
            "confidence": 0.0,
            "injury_status": None,
            "expected_return": None,
            "start_probability": None
        }
        
    def _extract_info_from_text(self, content: str, player_name: str) -> Dict[str, Any]:
        """Extract information from plain text response"""
        if not content or not isinstance(content, str):
            return self._create_default_response(player_name)
            
        result = {"summary": content, "status": "Available"}

        # Try to detect injury/suspension status from text, respecting negations
        text = content.lower()
        # Common negations that imply availability
        negations = [
            'no injury', 'no injuries', 'no recent injury', 'no injury concerns',
            'no fitness issues', 'not injured', 'no doubts', 'no doubt',
            'not doubtful', 'no suspension', 'not suspended', 'no issues', 'no concerns'
        ]
        if any(neg in text for neg in negations):
            result["status"] = "Available"
            return result

        # Suspensions
        if 'suspended' in text:
            result["status"] = "Suspended"
            return result

        # Injuries (avoid generic 'out' which creates false positives)
        injury_triggers = ['injury', 'injured', 'ruled out', 'out injured']
        if any(trig in text for trig in injury_triggers):
            result["status"] = "Injured"
            return result

        # Doubtful indicators
        doubtful_triggers = ['doubt', 'doubtful', 'uncertain', 'late fitness test', 'fitness test', 'assess']
        if any(trig in text for trig in doubtful_triggers):
            result["status"] = "Doubtful"

        return result
        
    def _build_analysis_prompt(self, player_name: str, context: Dict[str, Any] = None) -> str:
        """Build a detailed, reliability-focused prompt for player analysis."""
        from datetime import datetime, timedelta
        now_utc = datetime.utcnow()
        window_days = 7
        start_utc = (now_utc - timedelta(days=window_days)).date().isoformat()
        current_utc = now_utc.date().isoformat()

        team_name = context.get('current_team', 'their club') if context else 'their club'
        next_event = context.get('next_fixture', 'the upcoming Premier League gameweek') if context else 'the upcoming Premier League gameweek'

        # We keep the output aligned with fields consumed by analyze_players/analyze_player in our codebase.
        prompt = f"""
Role: You are an FPL injury/availability analyst. Provide verified, recent status for a specific Premier League player.

Current date (UTC): {current_utc}
Time window: last {window_days} days inclusive → {start_utc} to {current_utc}

Task:
- Report FPL-relevant injury/availability for the player below for {next_event}.

Hard constraints (recency and reliability):
- Only use sources published within the time window above. Discard anything older.
- Do not reference or infer from dates after {current_utc}.
- If no qualifying source exists within the window, set status = "No recent update".
- Ignore prior-season or preseason notes unless reaffirmed by a source within the window.

Source hierarchy (use highest authority and most recent):
1) Official club site/medical bulletin/squad updates
2) Manager press conference transcript or direct quotes
3) FA/PL disciplinary updates
4) Tier-1 journalism (BBC, The Athletic, Sky Sports, Guardian, Times) and trusted local beat reporters
- Exclude unverified social media, fan blogs, and generic aggregators unless quoting a primary source within the window.

Conflict resolution:
- Prefer the most recent higher-authority source.
- If credible sources conflict, pick the most recent higher-authority and note any conflict in notes.

Output format (strict JSON object; no additional text):
{{
  "player": "{player_name}",
  "team": "{team_name}",
  "status": "Fit" | "Injured" | "Doubtful" | "Suspended" | "Unavailable" | "No recent update",
  "injury_status": "e.g., hamstring, rib, illness" | null,
  "expected_return": "YYYY-MM-DD" | null,
  "start_probability": 0-100,
  "confidence": "Low" | "Medium" | "High",
  "summary": "1–3 sentences in British English; concise; no URLs"
}}

Validation rules before answering:
1) All source dates are within {start_utc}–{current_utc}
2) No future-dated references
3) Do not fabricate expected_return; use null when unknown
4) British English; ISO dates only; no speculation

Player:
- {player_name} ({team_name})
"""

        return prompt

    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate a cache key from the request data."""
        # Create a stable string representation of the data
        data_str = json.dumps(data, sort_keys=True)
        # Hash the string to create a fixed-length key
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "perplexity_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        self.cache[key] = {
            'timestamp': datetime.now().timestamp(),
            'data': data
        }
        # Save cache to disk
        try:
            cache_file = self.cache_dir / "perplexity_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return number of cleared entries."""
        if not self.cache:
            return 0
            
        now = datetime.now().timestamp()
        expired = [k for k, v in self.cache.items() 
                  if now - v.get('timestamp', 0) > self.cache_ttl.total_seconds()]
        
        for k in expired:
            del self.cache[k]
            
        if expired:
            # Save the updated cache
            try:
                cache_file = self.cache_dir / "perplexity_cache.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache after cleanup: {e}")
        
        return len(expired)

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if it exists and is not expired."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        now = datetime.now().timestamp()
        
        if now - entry.get('timestamp', 0) > self.cache_ttl.total_seconds():
            return None
            
        return entry['data']

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True,
           retry=retry_if_exception_type(requests.RequestException))
    def _post(self, path: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._queries_made += 1
        logger.info(f"Making Perplexity API request ({self._queries_made}/{self.max_queries}) to {url}")
        logger.debug(f"Request payload: {json.dumps(json_data, indent=2)}")
        
        try:
            resp = self.session.post(url, json=json_data, timeout=self.timeout)
            logger.debug(f"API response status: {resp.status_code}")
            resp.raise_for_status()
            
            if not resp.text.strip():
                logger.warning("Received empty response from Perplexity API")
                return {}
            
            return resp.json()

        except requests.RequestException as e:
            logger.error(f"Perplexity API request failed: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}. Response text: {resp.text[:200]}")
            return {}

    def can_query(self) -> bool:
        has_key = self.api_key is not None
        under_budget = self._queries_made < self.max_queries
        logger.info("Perplexity can_query check: has_key=%s, queries_made=%d, max_queries=%d, under_budget=%s", 
                   has_key, self._queries_made, self.max_queries, under_budget)
        return has_key and under_budget

    def research_player(self, player_name: str) -> Optional[Dict[str, str]]:
        """
        Perform a conservative research query about a player's injury/rotation/probable start.
        Returns a dict with 'summary' and 'start_probability' (0-100) where possible.
        If API not configured or budget reached, returns None.
        """
        if not self.can_query():
            logger.info("Perplexity budget or API key not available; skipping query for %s", player_name)
            return None
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an assistant summarizing up-to-date football player news for FPL decisions."},
                    {"role": "user", "content": (
                        f"In the last 7 days only, summarize reliable news on {player_name}: injury status, likely return date, rotation risk, manager quotes, and tactical role. "
                        "Use reputable web sources and cautious social media signals (official club/X posts, trusted journalists), avoid speculation, and prefer corroborated reports. "
                        "Return a concise 1-2 sentence summary and an estimated probability (0-100) of starting the next Premier League match." 
                    )},
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            }
            data = self._post("/chat/completions", json_data=payload)
            # Parse response (structure may vary by API version).
            summary = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content")
                    summary = content
            # Very simple heuristic to extract a probability
            start_prob = None
            if summary:
                import re
                m = re.search(r"(\d{1,3})%", summary)
                if m:
                    try:
                        start_prob = max(0, min(100, int(m.group(1))))
                    except Exception:
                        start_prob = None
            return {"summary": summary or "No summary parsed.", "start_probability": start_prob}
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                logger.warning("Perplexity unauthorized or forbidden. Check API key.")
            return None
        except Exception:
            logger.exception("Unexpected error querying Perplexity for %s", player_name)
            return None
