import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NewsCache:
    def __init__(self, cache_dir: Optional[str] = None, ttl_days: int = 1):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".fpl_news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "news_cache.json"
        self.ttl = timedelta(days=ttl_days)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving news cache: {e}")

    def get_news(self, player_name: str) -> Optional[Dict[str, Any]]:
        player_data = self.cache.get(player_name)
        if not player_data:
            return None

        last_updated = datetime.fromisoformat(player_data.get('last_updated', '1970-01-01'))
        if datetime.now() - last_updated > self.ttl:
            logger.debug(f"Cache expired for {player_name}")
            return None
        
        return player_data.get('data')

    def save_news(self, player_name: str, news_data: Dict[str, Any]) -> None:
        if not news_data:
            return
        
        self.cache[player_name] = {
            'last_updated': datetime.now().isoformat(),
            'data': news_data
        }
        self._save_cache()

    def get_all_news(self) -> Dict[str, Dict[str, Any]]:
        # This is now more efficient as it reads from the in-memory cache
        all_news = {}
        for player_name, player_data in self.cache.items():
            if self.get_news(player_name): # Check for expiration
                all_news[player_name] = player_data.get('data')
        return all_news

    def clear_expired(self) -> int:
        initial_count = len(self.cache)
        now = datetime.now()
        self.cache = {
            player_name: data
            for player_name, data in self.cache.items()
            if now - datetime.fromisoformat(data.get('last_updated', '1970-01-01')) <= self.ttl
        }
        expired_count = initial_count - len(self.cache)
        if expired_count > 0:
            self._save_cache()
        return expired_count

    # New helper methods used by news_analyzer
    def has(self, player_name: str) -> bool:
        """Return True if we have non-expired news for the player in cache."""
        return self.get_news(player_name) is not None

    def delete(self, player_name: str) -> None:
        """Remove a player's news from cache and persist the change."""
        if player_name in self.cache:
            del self.cache[player_name]
            self._save_cache()
