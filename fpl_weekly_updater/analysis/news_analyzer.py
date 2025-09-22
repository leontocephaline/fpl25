from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..apis.perplexity_client import PerplexityClient
from ..utils.news_cache import NewsCache

logger = logging.getLogger(__name__)


def analyze_players(
    client: PerplexityClient,
    player_names: List[str],
    team_context: Dict[str, Any],
    force_refresh_players: List[str] | None = None,
    predictions: Dict[str, float] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping: player_name -> {
        'summary': str,
        'start_probability': int|None,
        'status': str,
        'last_updated': str,
        'confidence': float,
        'transfer_recommendation': str|None
    }
    
    Args:
        client: Perplexity client instance
        player_names: List of player names to analyze
        team_context: Optional team context information
        force_refresh_players: List of player names to force refresh
        predictions: Optional dictionary of player predictions
        cache_dir: Optional directory for news cache
    """
    results: Dict[str, Dict[str, Any]] = {}
    
    # Initialize cache
    news_cache = NewsCache(cache_dir=cache_dir)
    
    # Clear cache for players that need a forced refresh
    if force_refresh_players:
        for player_name in force_refresh_players:
            if news_cache.has(player_name):
                news_cache.delete(player_name)
                logger.info(f"Cleared cache for player: {player_name}")
    
    # Clear expired cache entries
    expired_count = news_cache.clear_expired()
    if expired_count > 0:
        logger.info(f"Cleared {expired_count} expired cache entries")
    
    # Get current timestamp for all analyses
    current_time = datetime.utcnow().isoformat()
    
    for name in player_names:
        logger.info(f"Processing player: {name}")

        # Check cache first
        cached_data = news_cache.get_news(name)
        if cached_data:
            logger.info(f"Using cached news for {name}")
            results[name] = cached_data
            continue

        # If not in cache, define a default result and query the API
        result = {
            'summary': f"No recent updates available for {name}.",
            'start_probability': None,
            'status': 'unknown',
            'last_updated': current_time,
            'confidence': 0.0,
            'transfer_recommendation': None,
            'injury_status': None,
            'expected_return': None
        }

        if client and client.can_query():
            try:
                logger.info(f"Querying API for player: {name}")
                context = {
                    'current_team': team_context.get('team_name') if team_context else None,
                    'predicted_points': predictions.get(name, 0) if predictions else 0,
                    'current_gw': team_context.get('current_event') if team_context else None,
                    'next_fixture': team_context.get('next_event') if team_context else None
                }
                
                analysis = client.analyze_player(name, context=context, force_refresh=False)
                
                if analysis:
                    logger.info(f"Received analysis for {name}")
                    summary = analysis.get('summary')
                    if not summary or summary == f"No recent updates available for {name}.":
                        summary = f"No recent updates available for {name}."
                    
                    result.update({
                        'summary': summary,
                        'start_probability': analysis.get('start_probability'),
                        'status': analysis.get('status', 'unknown'),
                        'confidence': analysis.get('confidence', 0.0),
                        'injury_status': analysis.get('injury_status'),
                        'expected_return': analysis.get('expected_return'),
                        'last_updated': current_time
                    })
                    
                    if analysis.get('start_probability', 0) < 50 and team_context and name in team_context.get('current_team', []):
                        result['transfer_recommendation'] = generate_transfer_recommendation(name, analysis, context)
                    
                    if result.get('summary') and result['summary'] != f"No recent updates available for {name}.":
                        news_cache.save_news(name, result)
                        logger.info(f"Successfully saved to cache for {name}")

            except Exception as e:
                logger.error(f"Error analyzing player {name}: {str(e)}", exc_info=True)

        # Add the result (either from API or default) to the results dictionary
        results[name] = result
    
    return results


def generate_transfer_recommendation(player_name: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Generate a transfer recommendation based on player analysis and team context"""
    if analysis.get('injury_status') in ['injured', 'doubtful']:
        return f"Consider transferring out due to {analysis['injury_status']} status"
        
    if analysis.get('start_probability', 0) < 50:
        return "Low chance of starting next match - consider alternatives"
        
    if analysis.get('expected_return') and analysis['expected_return'] > context.get('current_gw', 0) + 2:
        return f"Expected to return in GW{analysis['expected_return']} - consider short-term replacements"
        
    return None


def validate_team_selection(team: Dict[str, Any], player_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate team selection against player analyses and return any issues found"""
    issues = []
    
    for player_name, player_data in player_analyses.items():
        if player_data.get('status') in ['injured', 'doubtful'] and player_name in team.get('starting_xi', []):
            issues.append({
                'player': player_name,
                'issue': f"{player_name} is marked as {player_data['status']} but in starting XI",
                'severity': 'high',
                'suggestion': f"Consider benching {player_name} or transferring out"
            })
            
        if player_data.get('start_probability', 0) < 30 and player_name in team.get('starting_xi', []):
            issues.append({
                'player': player_name,
                'issue': f"{player_name} has low chance of starting ({player_data.get('start_probability')}%)",
                'severity': 'medium',
                'suggestion': f"Verify {player_name}'s status before the deadline"
            })
            
    return issues
