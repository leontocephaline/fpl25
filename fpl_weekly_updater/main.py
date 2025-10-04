from __future__ import annotations

import json
import logging
import sys
import os

# Ensure 'src' package directory is importable in all environments (dev, PyInstaller onedir/onefile)
# We try multiple candidate roots and add the first matching 'src' to sys.path.
from pathlib import Path as _Path
def _ensure_src_on_path() -> None:
    candidates = []
    # 1) Project root when running from source tree (parent of this package)
    try:
        pkg_parent = _Path(__file__).resolve().parents[1]
        candidates.append(pkg_parent)
    except Exception:
        pass
    # 2) PyInstaller onefile extraction dir
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(_Path(meipass))
    # 3) Directory of the executable (PyInstaller onedir)
    try:
        candidates.append(_Path(sys.executable).parent)
    except Exception:
        pass
    # 4) Current working directory (when launched from project root)
    try:
        candidates.append(_Path.cwd())
    except Exception:
        pass
    for root in candidates:
        try:
            src_dir = (root / "src")
            if src_dir.exists() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                break
        except Exception:
            continue
_ensure_src_on_path()

import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union, Any
from src.models.ml_predictor import MLPredictor
import inspect
from fpl_weekly_updater.apis.fpl_client import FPLClient
from fpl_weekly_updater.apis.fpl_auth import get_password, fetch_picks
from fpl_weekly_updater.apis.fpl_browser_login import login_and_get_cookie
from fpl_weekly_updater.apis.perplexity_client import PerplexityClient
from fpl_weekly_updater.apis.betting_client import BettingClient
from fpl_weekly_updater.analysis.player_scorer import PlayerInputs, score_player
from fpl_weekly_updater.analysis.news_analyzer import analyze_players
from fpl_weekly_updater.analysis.transfer_optimizer import recommend_transfers
from fpl_weekly_updater.config.settings import load_settings
from fpl_weekly_updater.reporting.pdf_generator import generate_pdf
from fpl_weekly_updater.reporting.appendix_generator import generate_appendix_pdf

# Ensure cache directory exists
CACHE_DIR = os.path.expanduser('~/.fpl_news_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Set up logging (default WARNING; override via LOG_LEVEL env var)
_env_log = os.getenv("LOG_LEVEL", "WARNING").upper()
_default_level = getattr(logging, _env_log, logging.WARNING)
logging.basicConfig(
    level=_default_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fpl_weekly_update.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_weekly_update(**kwargs) -> Path | None:
    settings = load_settings()
    
    # Initialize news dictionary to store player news
    news: Dict[str, Any] = {}
    # Runtime overrides
    skip_news: bool = bool(kwargs.get('skip_news', False))
    custom_report_dir = kwargs.get('report_dir')
    generate_appendix: bool = bool(kwargs.get('appendix', False))
    appendix_only: bool = bool(kwargs.get('appendix_only', False))

    extra_headers: Dict[str, str] = {}
    if settings.fpl_api_bearer:
        token = settings.fpl_api_bearer.strip()
        auth_value = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        extra_headers["Authorization"] = auth_value

    logger.debug("Initializing FPLClient...")
    fpl = FPLClient(
        base_url=settings.api.base_url,
        timeout=settings.api.timeout,
        rate_limit_delay=settings.api.rate_limit_delay,
        retry_attempts=settings.api.retry_attempts,
        session_cookie=settings.fpl_session_cookie,
        auth_headers=extra_headers or None,
        csrf_token=settings.fpl_csrf_token,
    )
    logger.info("Perplexity API key loaded: %s", "***" if settings.perplexity_api_key else "None")
    # Initialize Perplexity client for news analysis
    perplexity = None
    if (not skip_news) and settings.perplexity_api_key:
        logger.debug("Initializing Perplexity client...")
        logger.info("Perplexity API key found in settings")
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.perplexity_cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
        
        # Initialize with 1-day TTL for cache
        try:
            perplexity = PerplexityClient(
                api_key=settings.perplexity_api_key,
                max_queries=settings.perplexity_max_queries,
                model=settings.perplexity_model,
                cache_dir=cache_dir,
                cache_ttl_hours=24  # 1 day cache TTL
            )
            logger.info("Perplexity client initialized successfully")
            logger.debug("Perplexity client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Perplexity client: {str(e)}")
    else:
        logger.warning("No Perplexity API key found in settings. News analysis will be limited.")
    
    # Clear expired cache entries
    if perplexity:
        cleared = perplexity.clear_expired_cache()
        if cleared > 0:
            logger.info(f"Cleared {cleared} expired cache entries")

    betting = BettingClient(
        betfair_app_key=settings.betfair_app_key,
        betfair_session_token=settings.betfair_session_token,
    )

    # Initialize ML Predictor
        # DEBUG: Print the file path of the MLPredictor class to resolve module conflicts
    logging.info(f"MLPredictor class loaded from: {inspect.getfile(MLPredictor)}")

    ml_predictor = MLPredictor(config=settings.model_dump())


    logger.debug("FPLClient initialized.")

    # 1) Fetch core public data
    logger.debug("Fetching bootstrap data...")
    bootstrap = fpl.get_bootstrap()
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    element_types = {et["id"]: et for et in bootstrap.get("element_types", [])}
    logger.debug("Bootstrap data fetched successfully.")
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    # 2) Fetch current squad if authenticated
    logger.debug("Fetching current squad...")
    team_id = settings.fpl_team_id
    my_team = None
    # Attempt to get team data through different methods
    team_sources = []
    
    # Method 1: Secure FPL library with keyring
    if team_id and settings.fpl_email:
        try:
            pwd = get_password(settings.fpl_email)
            if pwd:
                # Derive current event id from bootstrap
                current_event = None
                events = bootstrap.get("events", [])
                for ev in events:
                    if ev.get("is_current"):
                        current_event = ev.get("id")
                        break
                if not current_event and events:
                    # Fallback to next event or last available id
                    nxt = next((ev for ev in events if ev.get("is_next")), None)
                    current_event = (nxt or events[-1]).get("id")
                
                logger.info(f"Attempting to fetch team data using secure FPL API...")
                my_team = fetch_picks(team_id, settings.fpl_email, pwd, current_event or 1)
                if my_team:
                    team_sources.append("Secure FPL API")
                    logger.info("Successfully fetched team data using secure FPL API")
        except Exception as e:
            logger.warning(f"Secure FPL auth failed: {str(e)}")
    
    # Method 2: Direct API with existing session
    if not my_team and team_id:
        try:
            logger.info("Attempting to fetch team data using direct API...")
            my_team = fpl.get_my_team(team_id)
            if my_team:
                team_sources.append("Direct API")
                logger.info("Successfully fetched team data using direct API")
        except Exception as e:
            logger.warning(f"Direct API fetch failed: {str(e)}")
    
    # Method 3: Headless browser login
    if not my_team and team_id and settings.fpl_email:
        try:
            logger.info("Attempting headless browser login to fetch team data...")
            new_cookie = login_and_get_cookie(
                settings.fpl_email,
                headless=settings.fpl_browser_headless,
                browser=settings.fpl_browser,
                team_id=team_id,
            )
            
            # If browser flow extracted data directly, load it
            if new_cookie == "DATA_EXTRACTED" and os.path.exists("team_data.json"):
                try:
                    with open("team_data.json", "r", encoding="utf-8") as f:
                        my_team = json.load(f)
                        team_sources.append("Browser-extracted JSON")
                        logger.info("Successfully loaded team data from extracted page JSON")
                except Exception as e:
                    logger.warning(f"Failed to load team_data.json: {str(e)}")
            elif new_cookie and new_cookie != "DATA_EXTRACTED":
                # Build a fresh FPL client with the new cookie
                fpl_fresh = FPLClient(
                    base_url=settings.api.base_url,
                    timeout=settings.api.timeout,
                    rate_limit_delay=settings.api.rate_limit_delay,
                    retry_attempts=settings.api.retry_attempts,
                    session_cookie=new_cookie,
                    auth_headers=extra_headers or None,
                    csrf_token=None,
                )
                my_team = fpl_fresh.get_my_team(team_id)
                if my_team:
                    team_sources.append("Browser-authenticated API")
                    logger.info("Successfully fetched team data using browser-authenticated API")
        except Exception as e:
            logger.warning(f"Headless browser login failed: {str(e)}")
    
    # Fallback to default team data if all methods failed
    if not my_team or not isinstance(my_team, dict):
        logger.warning("Could not fetch authenticated team. Using default team data.")
        my_team = {
            "picks": [],
            "name": "My Team",
            "summary_overview_points": 0,
            "summary_event_points": 0,
            "value": 0,
            "bank": 0,
            "chips": []
        }
        team_sources.append("Default (fallback)")
    else:
        # Ensure all required fields exist
        my_team.setdefault("picks", [])
        my_team.setdefault("name", "My Team")
        my_team.setdefault("summary_overview_points", 0)
        my_team.setdefault("summary_event_points", 0)
        my_team.setdefault("last_deadline_value", 0)
        my_team.setdefault("last_deadline_bank", 0)
        my_team.setdefault("chips", [])
    
    logger.debug("Squad data fetched.")
    logger.info(f"Team data sources tried: {', '.join(team_sources) or 'None'}")
    logger.info(f"Team data structure: {json.dumps(my_team, indent=2) if my_team else 'None'}")


    # Log available player count for debugging
    logger.info(f"Loaded {len(elements)} players from bootstrap data")
    
    # Log a few sample players to verify data structure
    logger.debug("Sample players from bootstrap data:")
    for pid, player in list(elements.items())[:5]:  # Show first 5 players as sample
        logger.debug(f"  - {player.get('first_name')} {player.get('second_name')} (ID: {pid}, Team: {player.get('team')}, Pos: {player.get('element_type')})")
    
    # Find David Raya's player ID with more flexible matching
    raya_id = None
    for pid, player in elements.items():
        first_name = player.get('first_name', '').lower()
        second_name = player.get('second_name', '').lower()
        full_name = f"{first_name} {second_name}"
        
        # More flexible matching for David Raya
        if 'raya' in full_name:
            raya_id = pid
            logger.info(f"Found David Raya with ID: {raya_id}, Full Name: {player.get('first_name')} {player.get('second_name')}")
            break
    
    # Build player data structures and process team
    player_ids: List[int] = []
    player_names: set[str] = set()
    element_to_name: Dict[int, str] = {}
    element_to_pos: Dict[int, str] = {}
    element_to_team_name: Dict[int, str] = {}
    element_to_team: Dict[int, str] = {}
    element_to_web_name: Dict[int, str] = {}

    # Process team picks if available
    logger.debug("Processing team picks...")
    logger.info(f"Checking my_team before processing: {my_team is not None and 'picks' in my_team}")
    if my_team and "picks" in my_team:
        logger.info(f"Processing {len(my_team['picks'])} players from team")
        
        # Log the raw picks data for debugging
        logger.debug(f"Raw picks data: {json.dumps(my_team['picks'], indent=2)}")
        
        # Log team data structure for debugging
        logger.debug(f"Team data structure: {json.dumps({k: v for k, v in my_team.items() if k != 'picks'}, indent=2)}")
        
        # First, process all picks to build the team
        for pick in my_team["picks"]:
            try:
                # Log the raw pick data
                logger.debug(f"Processing pick: {pick}")
                
                # Get player ID and ensure it's an integer
                pid = pick.get("element")
                if pid is None:
                    logger.warning("Found player with no element ID in picks")
                    continue
                    
                try:
                    pid = int(pid)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid player ID format: {pid}, error: {str(e)}")
                    continue
                
                # Debug: Log the player ID being processed
                logger.debug(f"Processing player ID: {pid} (type: {type(pid)})")
                
                # Get player data from elements
                el = elements.get(pid)
                
                # Debug: Log if player data was found
                if not el:
                    logger.warning(f"No data found for player ID: {pid}")
                    # Log available player IDs for debugging
                    available_pids = list(elements.keys())
                    logger.debug(f"Available player IDs (first 20): {available_pids[:20]}")
                    if pid not in available_pids:
                        logger.warning(f"Player ID {pid} not found in elements data")
                    continue
                
                # Check for Matz Sels replacement
                if el.get('first_name') == 'Matz' and el.get('second_name') == 'Sels' and raya_id:
                    logger.info(f"Replacing Matz Sels with David Raya in the team")
                    old_pid = pid
                    pid = raya_id
                    el = elements.get(pid)
                    if not el:
                        logger.warning(f"David Raya (ID: {raya_id}) not found in elements")
                        # Revert to original player if Raya not found
                        pid = old_pid
                        el = elements.get(pid)
                        if not el:
                            logger.error(f"Original player (ID: {old_pid}) not found either, skipping")
                            continue
                
                # Get player details with better error handling
                first_name = el.get('first_name', '')
                second_name = el.get('second_name', '')
                player_name = f"{first_name} {second_name}".strip()
                
                # Log player details for debugging
                logger.debug(f"Player details - ID: {pid}, Name: {player_name}, Position: {el.get('element_type')}, Team: {el.get('team')}")
                
                # Only add if we have a valid name
                if not player_name or player_name == ' ':
                    logger.warning(f"Empty player name for ID {pid}, skipping")
                    logger.debug(f"Player data: {el}")
                    continue

                # Store player information
                player_ids.append(pid)
                player_names.add(player_name)
                element_to_name[pid] = player_name

                # Map position
                pos_code = "GK" if el.get("element_type") == 1 else \
                         "DEF" if el.get("element_type") == 2 else \
                         "MID" if el.get("element_type") == 3 else "FWD"
                element_to_pos[pid] = pos_code

                # Map team
                team_id_map = el.get("team")
                team_name = teams.get(team_id_map, {}).get("name", "Unknown")
                element_to_team_name[pid] = team_name
                element_to_team[pid] = team_id_map

                # Log the mapping
                logger.debug(f"Mapped player: {player_name} (ID: {pid}) - Position: {pos_code}, Team: {team_name}")

            except Exception as e:
                logger.error(f"Error processing player pick {pick}: {str(e)}")
                continue

        logger.info(f"Player names after processing loop: {player_names}")
    logger.debug("Team picks processed.")
            

    # Convert set to sorted list for consistent ordering
    player_names_list = sorted(list(player_names))

    # 3) Enhanced news analysis with team context and predictions
    current_event = next((ev["id"] for ev in bootstrap.get("events", []) if ev.get("is_current")), None)
    next_event = next((ev["id"] for ev in bootstrap.get("events", []) if ev.get("is_next")), None)
        # Log team summary
    logger.info(f"Processed team with {len(player_names_list)} unique players")
    logger.info(f"Player names: {', '.join(player_names_list) if player_names_list else 'None'}")
    
    # If no player names, log more details for debugging
    if not player_names_list:
        logger.warning("No player names found in team. Element to name mapping:")
        for pid, name in element_to_name.items():
            logger.warning(f"  - {pid}: {name} (Position: {element_to_pos.get(pid, '?')}, Team: {element_to_team_name.get(pid, '?')})")
    
    logger.info(f"Final list of player names for analysis: {player_names_list}")

    team_context = {
        'team_name': my_team.get('name', 'My Team') if my_team else 'My Team',
        'current_event': current_event,
        'next_event': next_event,
        'current_team': player_names_list,
        'element_to_name': element_to_name,
        'element_to_pos': element_to_pos,
        'element_to_team': element_to_team_name,
        'team_stats': {
            'Team Name': my_team.get('name', 'My Team'),
            'Overall Points': my_team.get('summary_overview_points', 0),
            'Gameweek Points': my_team.get('summary_event_points', 0),
            'Team Value': f"£{my_team.get('last_deadline_value', 0) / 10:.1f}m",
            'Bank': f"£{my_team.get('last_deadline_bank', 0) / 10:.1f}m",
            'Chips': ', '.join([c['name'] for c in my_team.get('chips', []) if c.get('status_for_entry') == 'available'])
        }
    }
    
    
    # ML Model Training and Prediction
    # Convert elements data to a DataFrame for ML processing
    elements_df = pd.DataFrame(bootstrap.get("elements", []))

    # Get current gameweek for training horizon
    current_gameweek = team_context.get('current_event') or team_context.get('next_event')

    # If a specific gameweek is provided for prediction generation, use it
    gameweek_to_predict = kwargs.get('generate_predictions_for_gw') or current_gameweek

    if gameweek_to_predict:
        # Add position to the DataFrame for filtering
        elements_df['position'] = elements_df['element_type'].map({et['id']: et['singular_name_short'] for et in bootstrap.get('element_types', [])})

        # If generating for a specific gameweek, we might need to train first
        if kwargs.get('generate_predictions_for_gw'):
            logger.info(f"Training models for Gameweek {gameweek_to_predict}...")
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                logger.info(f"Training model for position: {position}")
                position_data = elements_df[elements_df['position'] == position]
                if position_data.empty:
                    logger.warning(f"No training data for position {position}. Skipping fit.")
                    continue

                # Train and save the model for the target gameweek
                ml_predictor.train_position_model(
                    df=position_data, 
                    position=position, 
                    feature_lineage={}, 
                    target_gw=gameweek_to_predict
                )

        # Generate predictions for the target gameweek
        logger.info(f"Generating player point predictions for Gameweek {gameweek_to_predict}...")
        predictions_df, _ = ml_predictor.predict_player_points(elements_df, {}, {}, horizon=gameweek_to_predict)

        if not predictions_df.empty:
            logger.info(f"Generated predictions for {len(predictions_df)} players.")
            
            # If generating for backtesting, save to the backtest directory
            if kwargs.get('generate_predictions_for_gw'):
                backtest_dir = Path('data/backtest')
                backtest_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = backtest_dir / f"predictions_gw{gameweek_to_predict:02d}_{timestamp}.csv"
                predictions_df.to_csv(filename, index=False)
                logger.info(f"Saved backtest predictions to {filename}")

            # Create a score dictionary from predictions for the report
            scores = {row['id']: {'total': row['predicted_points']} for index, row in predictions_df.iterrows()}
        else:
            logger.warning("Prediction DataFrame is empty. Falling back to heuristic scoring.")
            scores: Dict[int, Dict[str, float]] = {}
    else:
        logger.error("Could not determine current gameweek. Skipping model training and prediction.")
        scores: Dict[int, Dict[str, float]] = {}
    
    
    # Analyze player news if the client is available
    if perplexity and player_names_list:
        logger.debug("Starting news analysis...")
        logger.info(f"Analyzing news for {len(player_names_list)} players...")
        try:
            news = analyze_players(perplexity, player_names_list, team_context)
            logger.info("News analysis complete.")
            # Save news to a file for debugging
            with open("player_news.json", "w", encoding="utf-8") as f:
                json.dump(news, f, indent=2)
            logger.debug("News analysis finished.")
        except Exception as e:
            logger.error(f"Error during news analysis: {str(e)}")
    else:
        logger.warning("Skipping news analysis because Perplexity client is not available or no players found.")

    # Initialize scores for all players
    for pid in player_ids:
        if pid not in elements:
            logger.warning(f"Player ID {pid} not found in elements, skipping")
            continue
            
        player_data = elements[pid]
        player_name = element_to_name.get(pid, f"Player {pid}")
        
        # Get form value
        form_str = player_data.get("form", "0")
        try:
            form_val = float(form_str)
        except (ValueError, TypeError):
            form_val = 0.0
            
        # Map form to trend (-1.0 to 1.0) with 3.0 as neutral
        form_trend = max(-1.0, min(1.0, (form_val - 3.0) / 3.0))
        
        # Get player news if available
        player_news = news.get(player_name, {}) if news else {}
        
        # Adjust form trend based on news analysis
        if player_news.get('status') in ['injured', 'doubtful']:
            form_trend = max(-1.0, form_trend - 0.3)  # Penalize injured/doubtful players
            
        # Get start probability from news or fall back to FPL data
        start_prob = player_news.get('start_probability')
        if start_prob is None:
            start_prob = player_data.get("chance_of_playing_next_round", 100)
            try:
                start_prob = int(start_prob) if start_prob is not None else 100
            except (ValueError, TypeError):
                start_prob = 100
        
        # Create player inputs with enhanced data
        inputs = PlayerInputs(
            player_id=pid,
            name=player_name,
            position=element_to_pos.get(pid, "MID"),
            team=element_to_team_name.get(pid, ""),
            prob_goal=player_data.get("goals_per_game"),
            prob_assist=player_data.get("assists_per_game"),
            prob_clean_sheet=player_data.get("clean_sheets_per_game"),
            prob_card=player_data.get("yellow_cards_per_game"),
            xg_per90=player_data.get("expected_goals_per_90"),
            xa_per90=player_data.get("expected_assists_per_90"),
            xpts_per90=player_data.get("expected_goal_involvements_per_90"),
            minutes_per_game=player_data.get("minutes_per_game"),
            fixture_difficulty=player_data.get("fixture_difficulty", 1.0),
            form_trend=form_trend,
            start_probability=start_prob,
            injury_status=player_news.get('injury_status'),
            expected_return=player_news.get('expected_return'),
            confidence=player_news.get('confidence', 0.5) if isinstance(player_news.get('confidence'), (int, float)) else 0.5
        )
        
        # Score the player with enhanced inputs
        scores[pid] = score_player(inputs)
        
        # Adjust score based on news confidence and transfer recommendations
        if player_news.get('transfer_recommendation'):
            # Slightly reduce score for players with transfer recommendations
            scores[pid]['total'] *= 0.9

    # 5) Enhanced position mapping with better GK detection
    def get_player_position(element: dict, element_types: dict) -> str:
        """Get player position from element data with enhanced GK detection."""
        etid = element.get("element_type")
        et = element_types.get(etid, {}) if etid else {}
        
        # Try to get position from singular_name_short first
        singular_short = (et.get("singular_name_short") or "").upper()
        
        # If not found or invalid, try singular_name
        if singular_short not in {"GK", "DEF", "MID", "FWD"}:
            singular_name = (et.get("singular_name") or "").upper()
            if any(kw in singular_name for kw in ["GOALKEEPER", "GOAL_KEEPER"]):
                singular_short = "GK"
            elif "DEFENDER" in singular_name:
                singular_short = "DEF"
            elif "MIDFIELDER" in singular_name:
                singular_short = "MID"
            elif "FORWARD" in singular_name or "STRIKER" in singular_name:
                singular_short = "FWD"
        
        # Fallback to element_type if still not determined
        if singular_short not in {"GK", "DEF", "MID", "FWD"}:
            if etid == 1:
                singular_short = "GK"
            elif etid == 2:
                singular_short = "DEF"
            elif etid == 3:
                singular_short = "MID"
            elif etid == 4:
                singular_short = "FWD"
            else:
                singular_short = "MID"  # Default fallback
        
        return singular_short
    
    # Create position mappings for all players
    all_pos: Dict[int, str] = {}
    for pid, el in elements.items():
        pos = get_player_position(el, element_types)
        all_pos[pid] = pos
        
        # Log position mapping for debugging
        player_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
        logger.debug(f"Player position - {player_name} (ID: {pid}): {pos} (element_type: {el.get('element_type')})")
    
    # Log all goalkeepers found
    gk_players = [f"{elements[pid].get('first_name')} {elements[pid].get('second_name')} (ID: {pid})" 
                 for pid, pos in all_pos.items() if pos == "GK"]
    logger.info(f"Found {len(gk_players)} goalkeepers in elements: {', '.join(gk_players) if gk_players else 'None'}")
    
    # Log position mappings for debugging
    logger.debug("Position mappings for all players:")
    for pid, el in elements.items():
        player_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
        logger.debug(f"  {player_name}: {all_pos.get(pid)} (element_type: {el.get('element_type')})")
    
    # Get all players with their scores and positions
    player_data = []
    for pid in player_ids:
        if pid not in all_pos:
            logger.warning(f"Player ID {pid} not found in position mapping, skipping")
            continue

        player_name = element_to_name.get(pid, f"Player {pid}")
        position = all_pos.get(pid, "MID")  # Default to MID if position not found

        # Log detailed info for goalkeepers
        if position == "GK":
            logger.info(f"Processing goalkeeper: {player_name} (ID: {pid})")

        player_data.append({
            'id': pid,
            'name': player_name,
            'position': position,
            'score': scores.get(pid, {}).get('total', 0),
            'team': element_to_team_name.get(pid, "Unknown")
        })

    # Market-wide quick scores using bootstrap heuristic (initialize early)
    market_scores: Dict[int, float] = {}
    for pid, el in elements.items():
        # Use bootstrap form (string) roughly mapped to form_trend; use chance_of_playing_next_round as start prob
        form_str = el.get("form") or "0"
        try:
            form_val = float(form_str)
        except Exception:
            form_val = 0.0
        # Map form ~0..10 to -1..1 centered ~2.5-3.0
        form_trend = max(-1.0, min(1.0, (form_val - 3.0) / 3.0))
        start_prob = el.get("chance_of_playing_next_round")
        try:
            start_prob = int(start_prob) if start_prob is not None else 100
        except Exception:
            start_prob = 100
        pos_code = all_pos.get(pid, "MID")
        team_name = teams.get(el.get("team"), {}).get("name", "")
        pinputs = PlayerInputs(
            player_id=pid,
            name=f"{el.get('first_name','')} {el.get('second_name','')}".strip(),
            position=pos_code,
            team=team_name,
            prob_goal=None,
            prob_assist=None,
            prob_clean_sheet=None,
            prob_card=None,
            xg_per90=None,
            xa_per90=None,
            xpts_per90=None,
            minutes_per_game=None,
            fixture_difficulty=1.0,
            form_trend=form_trend,
            start_probability=start_prob,
        )
        market_scores[pid] = score_player(pinputs)["total"]

    # Fallback: If no player data from team, use all available players
    if not player_data:
        logger.warning("No player data from team - using all available players as fallback")

        for pid, el in elements.items():
            if pid not in all_pos:
                continue

            player_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
            if not player_name or player_name == ' ':
                player_name = f"Player {pid}"

            position = all_pos.get(pid, "MID")

            # Use market scores as fallback
            score = market_scores.get(pid, 0)

            player_data.append({
                'id': pid,
                'name': player_name,
                'position': position,
                'score': score,
                'team': teams.get(el.get('team'), {}).get('name', 'Unknown')
            })

        logger.info(f"Created fallback player_data with {len(player_data)} players")

    # Log the full player_data list before filtering for debugging
    logger.info(f"Full player_data before filtering: {json.dumps(player_data, indent=2)}")

    # ... (rest of the code remains the same)
    # Log final team composition
    logger.info("\nFinal team composition before selection:")
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = [p for p in player_data if p['position'] == pos]
        logger.info(f"  {pos}: {len(pos_players)} players")
        for p in sorted(pos_players, key=lambda x: x['score'], reverse=True):
            logger.info(f"    - {p['name']} ({p['team']}): {p['score']:.2f}")
    
    # Sort players by score (descending)
    player_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply availability-based score penalties instead of excluding players
    penalized_players = []
    for player in player_data:
        player_name = player['name']
        pid = player['id']
        player_news = news.get(player_name, {}) or {}
        
        el = elements.get(pid, {})
        fpl_code = str(el.get('status', 'a')).lower()
        # Penalty multipliers: keep everyone but downweight non-available
        # a=1.0, d=0.8, i=0.5, s/u/n=0.2
        penalty = {
            'a': 1.0,
            'd': 0.8,
            'i': 0.5,
            's': 0.2,
            'u': 0.2,
            'n': 0.2,
        }.get(fpl_code, 1.0)

        # Keep news fields for display
        status = player_news.get('status', 'available')
        # Prefer news start probability if present, else bootstrap chance
        start_prob = player_news.get('start_probability')
        if start_prob is None:
            start_prob = el.get('chance_of_playing_next_round', 100)
        try:
            start_prob = int(start_prob) if start_prob is not None else 100
        except Exception:
            start_prob = 100

        adjusted_score = float(player['score']) * penalty

        penalized_players.append({
            'id': pid,
            'name': player_name,
            'position': player['position'],
            'score': adjusted_score,
            'team': player.get('team', 'Unknown'),
            'status': status,
            'start_prob': start_prob
        })
    
    player_data = penalized_players
    logger.info(f"After availability penalties, {len(player_data)} players remain")
    logger.info(f"Player data after penalties: {json.dumps(player_data, indent=2)}")

    
    
    # Log final team composition after filtering
    logger.info("\nFinal team composition after filtering:")
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = [p for p in player_data if p['position'] == pos]
        logger.info(f"  {pos}: {len(pos_players)} players")
        for p in sorted(pos_players, key=lambda x: x['score'], reverse=True):
            logger.info(f"    - {p['name']} ({p['team']}): {p['score']:.2f}")
    
    # Prepare team selection
    
    # DEBUG: Log player data before sorting by position
    logger.info(f"Player data before sorting by position ({len(player_data)} players): {json.dumps(player_data, indent=2)}")

    # Sort players by position and then by score (descending)
    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3, 'UNK': 4}
    player_data.sort(key=lambda x: (position_order.get(x['position'], 99), -x['score']))
    
    # DEBUG: Log player data after sorting by position
    logger.info(f"Player data after sorting by position ({len(player_data)} players): {json.dumps(player_data, indent=2)}")

    # Select optimal team (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
    gk = [p for p in player_data if p['position'] == 'GK']
    defs = [p for p in player_data if p['position'] == 'DEF']
    mids = [p for p in player_data if p['position'] == 'MID']
    fwds = [p for p in player_data if p['position'] == 'FWD']

    # Always include at least one GK
    if gk:
        selected = gk[:1]
    else:
        logger.warning("No goalkeepers found in player data. Adding the highest-scoring GK from all available players as a fallback.")
        all_available_gks = [p for p in player_data if p['position'] == 'GK']
        if all_available_gks:
            best_gk = max(all_available_gks, key=lambda p: p.get('score', 0))
            selected = [best_gk]
            logger.info(f"Added fallback goalkeeper: {best_gk['name']}")
        else:
            selected = []
            logger.error("CRITICAL: No goalkeepers available in the entire dataset to select.")

# ... (rest of the code remains the same)
    # Define team composition constraints
    max_def = 5
    max_mid = 5
    max_fwd = 3
    min_players = {
        'DEF': 3,
        'MID': 2,
        'FWD': 1
    }

    # Add minimum required outfield players
    selected.extend(defs[:min_players['DEF']])
    selected.extend(mids[:min_players['MID']])
    selected.extend(fwds[:min_players['FWD']])
    
    # Add remaining best players, ensuring we don't exceed position limits
    remaining_defs = [p for p in defs if p not in selected]
    remaining_mids = [p for p in mids if p not in selected]
    remaining_fwds = [p for p in fwds if p not in selected]
    
    # Sort remaining players by score
    remaining_players = remaining_defs + remaining_mids + remaining_fwds
    remaining_players.sort(key=lambda x: -x['score'])
    
    # Add remaining players until we have 11 starters
    while len(selected) < 11 and remaining_players:
        next_player = remaining_players.pop(0)
        pos = next_player['position']
        
        # Check if we can add this player without exceeding position limits
        current_pos_count = sum(1 for p in selected if p['position'] == pos)
        if (pos == 'GK' and current_pos_count < 1) or \
           (pos == 'DEF' and current_pos_count < max_def) or \
           (pos == 'MID' and current_pos_count < max_mid) or \
           (pos == 'FWD' and current_pos_count < max_fwd):
            selected.append(next_player)
    
    # Add remaining players to make up to 15 total players (full squad: 11 starters + 4 subs)
    # Full squad limits: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    all_players = player_data.copy()
    all_players.sort(key=lambda x: -x['score'])
    
    for player in all_players:
        if len(selected) >= 15:
            break
        if player not in selected:
            pos = player['position']
            current_pos_count = sum(1 for p in selected if p['position'] == pos)
            if current_pos_count < squad_limits.get(pos, 0):
                selected.append(player)
    
    # Final sort by score (but maintain starting 11 order)
    selected.sort(key=lambda x: -x['score'])
    
    # Ensure we have a GK in starting 11 and at least one on the bench
    gks_in_team = [p for p in selected if p['position'] == 'GK']
    if gks_in_team:
        # Make sure best GK is in starting 11
        best_gk = gks_in_team[0]
        if best_gk not in selected[:11]:
            # Find worst non-GK in starting 11 to swap with
            for i in range(10, -1, -1):
                if selected[i]['position'] != 'GK':
                        selected.remove(best_gk)
                        selected.insert(i, best_gk)
                        break
    
    # Final selection - ensure we have exactly 11 starters and 4 subs
    final_selected = selected[:15]  # Take first 15 players after sorting
    starters = [p['id'] for p in final_selected[:11]]
    subs = [p['id'] for p in final_selected[11:15]]
    
    # Log the team with proper positions
    logger.info("\nSelected Team:")
    logger.info("Starting 11:")
    for i, p in enumerate(final_selected[:11], 1):
        logger.info(f"{i:2d}. {p['name']:20} {p['position']:4} Score: {p['score']:.2f}")
    
    if len(final_selected) > 11:
        logger.info("\nSubstitutes:")
        for i, p in enumerate(final_selected[11:15], 1):
            logger.info(f"{i:2d}. {p['name']:20} {p['position']:4} Score: {p['score']:.2f}")
    
    # Log the optimal team
    logger.info("Optimal starting 11 based on scores:")
    for i, pid in enumerate(starters, 1):
        player = next((p for p in player_data if p['id'] == pid), None)
        if player:
            logger.info("%2d. %-20s %-4s Score: %.2f", i, player['name'], player['position'], player['score'])
    
    # Log the subs
    if subs:
        logger.info("\nSubstitutes:")
        for i, pid in enumerate(subs, 1):
            player = next((p for p in player_data if p['id'] == pid), None)
            if player:
                logger.info("%2d. %-20s %-4s Score: %.2f", i, player['name'], player['position'], player['score'])

    # Market-wide maps from bootstrap
    # Prices for all players (now_cost is in tenths of a million)
    all_prices: Dict[int, float] = {pid: (el.get("now_cost") or 0) / 10.0 for pid, el in elements.items()}
    # Names for all players
    all_names: Dict[int, str] = {pid: f"{el.get('first_name','')} {el.get('second_name','')}".strip() for pid, el in elements.items()}

    # Market-wide quick scores using bootstrap heuristic (initialize early)
    market_scores: Dict[int, float] = {}
    for pid, el in elements.items():
        # Use bootstrap form (string) roughly mapped to form_trend; use chance_of_playing_next_round as start prob
        form_str = el.get("form") or "0"
        try:
            form_val = float(form_str)
        except Exception:
            form_val = 0.0
        # Map form ~0..10 to -1..1 centered ~2.5-3.0
        form_trend = max(-1.0, min(1.0, (form_val - 3.0) / 3.0))
        start_prob = el.get("chance_of_playing_next_round")
        try:
            start_prob = int(start_prob) if start_prob is not None else 100
        except Exception:
            start_prob = 100
        pos_code = all_pos.get(pid, "MID")
        team_name = teams.get(el.get("team"), {}).get("name", "")
        pinputs = PlayerInputs(
            player_id=pid,
            name=f"{el.get('first_name','')} {el.get('second_name','')}".strip(),
            position=pos_code,
            team=team_name,
            prob_goal=None,
            prob_assist=None,
            prob_clean_sheet=None,
            prob_card=None,
            xg_per90=None,
            xa_per90=None,
            xpts_per90=None,
            minutes_per_game=None,
            fixture_difficulty=1.0,
            form_trend=form_trend,
            start_probability=start_prob,
        )
        market_scores[pid] = score_player(pinputs)["total"]
    # Bank in millions from entry endpoint (tenths of a million)
    bank_val = 0.0
    if team_id:
        try:
            entry = fpl.get_entry(team_id)
            if entry:
                # Prefer last_deadline_bank if present; else try bank
                raw_bank = entry.get("last_deadline_bank")
                if raw_bank is None:
                    raw_bank = entry.get("bank")
                if raw_bank is not None:
                    bank_val = float(raw_bank) / 10.0
        except Exception:
            logger.exception("Failed to fetch entry bank; defaulting bank to 0.0")

    # Fetch news for all players in the team first
    try:
        # Get all player names that we need news for
        team_player_names = [element_to_name.get(pid, f"Player {pid}") for pid in player_ids]
        logger.info(f"Team player names: {team_player_names}")
        
        # Only fetch news for players we don't already have
        missing_names = [n for n in team_player_names if n and n not in news]
        logger.info(f"Missing names for news: {missing_names}")
        logger.info(f"Perplexity client available: {perplexity is not None}")
        
        if missing_names and perplexity:
            logger.info(f"Fetching news for {len(missing_names)} players: {missing_names}")
            player_news = analyze_players(
                perplexity=perplexity,
                player_names=missing_names,
                cache_dir=CACHE_DIR
            )
            logger.info(f"Analyze players returned: {player_news}")
            if player_news:
                news.update(player_news)
                logger.info(f"Updated news with data for players: {list(player_news.keys())}")
            else:
                logger.warning("No news data returned from analyze_players")
        else:
            if not missing_names:
                logger.info("No missing names to fetch news for")
            if not perplexity:
                logger.warning("Perplexity client is not available")
    except Exception as e:
        logger.exception(f"Failed to fetch player news: {str(e)}")
        # Continue with whatever news we have

    # Build availability map for current players using news start_probability or bootstrap fallback
    avail_map: Dict[int, int] = {}
    for pid in player_ids:
        nm = element_to_name.get(pid, f"Player {pid}")
        sp = None
        if news and nm in news and news[nm].get("start_probability") is not None:
            sp = int(news[nm]["start_probability"])  # type: ignore[arg-type]
        if sp is None:
            ch = elements.get(pid, {}).get("chance_of_playing_next_round")
            try:
                sp = int(ch) if ch is not None else 100
            except Exception:
                sp = 100
        avail_map[pid] = sp

    transfers = recommend_transfers(
        player_ids,
        scores,
        budget=100.0,
        free_transfers=1,
        starters=starters,
        hit_cost=4,
        prices=all_prices,
        bank=bank_val,
        market_scores=market_scores,
        pos_map=all_pos,
        names=all_names,
        availabilities=avail_map,
        max_suggestions=3,
        protected_player_ids=[raya_id] if raya_id else None,
        # broaden suggestions to surface up to 3 even if net <= 0
        suggestions_per_position=2,
        include_bench_out_candidates=True,
        bench_availability_threshold=50,
        require_positive_net=False,
    )
    logger.info(f"Transfer recommendations: {len(transfers)} transfers generated")
    for idx, t in enumerate(transfers, 1):
        logger.info(f"  {idx}. {t.get('out_name', 'Unknown')} → {t.get('in_name', 'Unknown')}: gain={t.get('gain', 0):.2f}, hit={t.get('hit', 0)}, net={t.get('net', 0):.2f}")

    # Apply the top recommended transfer to the squad (user rule: OUT must be bench; IN can start)
    # This presents the optimized team as if the top recommendation is executed first
    if transfers:
        top = transfers[0]
        out_id = top.get('out')
        in_id = top.get('in')
        if out_id and out_id in player_ids:
            try:
                player_ids.remove(out_id)
                logger.info("Removed transferred-out player %s from squad for lineup computation", element_to_name.get(out_id, out_id))
            except ValueError:
                pass
        if in_id and in_id not in player_ids:
            player_ids.append(in_id)
            el_in = elements.get(in_id, {})
            in_name = element_to_name.get(in_id)
            if not in_name:
                in_name = f"{el_in.get('first_name', '')} {el_in.get('second_name', '')}".strip() or f"Player {in_id}"
                element_to_name[in_id] = in_name
            # Ensure auxiliary mappings stay in sync for the new player
            element_to_team_name[in_id] = teams.get(el_in.get('team'), {}).get('name', 'Unknown') if el_in else 'Unknown'
            all_pos[in_id] = get_player_position(el_in, element_types) if el_in else all_pos.get(in_id, 'MID')
            logger.info("Added transferred-in player %s to squad for lineup computation", in_name)

        # Recompute player_data for updated squad
        updated_player_data: List[Dict[str, Any]] = []
        for pid in player_ids:
            pos = all_pos.get(pid)
            if not pos:
                continue
            el = elements.get(pid, {})
            name = element_to_name.get(pid, f"Player {pid}")
            # score fallback to market_scores if not in scores
            base_score = 0.0
            if pid in scores:
                base_score = float(scores[pid].get('total', 0.0))
            else:
                base_score = float(market_scores.get(pid, 0.0))
            # availability penalty
            fpl_code = str(el.get('status', 'a')).lower()
            penalty = {'a':1.0,'d':0.8,'i':0.5,'s':0.2,'u':0.2,'n':0.2}.get(fpl_code,1.0)
            adjusted_score = base_score * penalty
            updated_player_data.append({
                'id': pid,
                'name': name,
                'position': pos,
                'score': adjusted_score,
                'team': el.get('team', 'Unknown')
            })

        # Re-run selection on updated_player_data
        gk = [p for p in updated_player_data if p['position'] == 'GK']
        defs = [p for p in updated_player_data if p['position'] == 'DEF']
        mids = [p for p in updated_player_data if p['position'] == 'MID']
        fwds = [p for p in updated_player_data if p['position'] == 'FWD']
        gk.sort(key=lambda x: -x['score'])
        defs.sort(key=lambda x: -x['score'])
        mids.sort(key=lambda x: -x['score'])
        fwds.sort(key=lambda x: -x['score'])

        selected = gk[:1] if gk else []
        max_def, max_mid, max_fwd = 5, 5, 3
        min_players = {'DEF':3,'MID':2,'FWD':1}
        selected.extend(defs[:min_players['DEF']])
        selected.extend(mids[:min_players['MID']])
        selected.extend(fwds[:min_players['FWD']])
        remaining_defs = [p for p in defs if p not in selected]
        remaining_mids = [p for p in mids if p not in selected]
        remaining_fwds = [p for p in fwds if p not in selected]
        remaining_players = remaining_defs + remaining_mids + remaining_fwds
        remaining_players.sort(key=lambda x: -x['score'])
        while len(selected) < 11 and remaining_players:
            next_player = remaining_players.pop(0)
            pos = next_player['position']
            current_pos_count = sum(1 for p in selected if p['position'] == pos)
            if (pos == 'GK' and current_pos_count < 1) or \
               (pos == 'DEF' and current_pos_count < max_def) or \
               (pos == 'MID' and current_pos_count < max_mid) or \
               (pos == 'FWD' and current_pos_count < max_fwd):
                selected.append(next_player)
        # fill to 15 using full-squad limits
        squad_limits = {'GK':2,'DEF':5,'MID':5,'FWD':3}
        all_players_sorted = sorted(updated_player_data, key=lambda x: -x['score'])
        for p in all_players_sorted:
            if len(selected) >= 15:
                break
            if p in selected:
                continue
            pos = p['position']
            if sum(1 for q in selected if q['position']==pos) < squad_limits[pos]:
                selected.append(p)
        # ensure best GK in XI
        selected.sort(key=lambda x: -x['score'])
        gks_in = [p for p in selected if p['position']=='GK']
        if gks_in:
            best_gk = gks_in[0]
            if best_gk not in selected[:11]:
                for i in range(10,-1,-1):
                    if selected[i]['position']!='GK':
                        selected.remove(best_gk)
                        selected.insert(i,best_gk)
                        break
        final_selected = selected[:15]
        starters = [p['id'] for p in final_selected[:11]]
        subs = [p['id'] for p in final_selected[11:15]]

    # Debug log for news data
    logger.info(f"News data before PDF generation: {news}")
    if news:
        logger.info(f"News items found for players: {list(news.keys())}")
    else:
        logger.warning("No news data available for PDF generation")

    # 6) PDF output - Create optimized team summary
    optimized_team = {
        "name": f"{my_team.get('name', 'My Team')} (Optimized)",
        "team_id": team_id,
        "players": ", ".join([element_to_name.get(pid, f"Player {pid}") for pid in starters + subs]) if player_names else "(not available)",
        "summary_overall_rank": my_team.get("summary_overall_rank", 0),
        "summary_event_points": my_team.get("summary_event_points", 0),
        "value": my_team.get("last_deadline_value") or my_team.get("value", 0),
        "bank": my_team.get("last_deadline_bank") or my_team.get("bank", 0),
        "chips": [chip["name"] for chip in my_team.get("chips", []) if chip.get("status_for_entry") == "active"]
    }

    # Create optimized player lineup
    optimized_lineup = []
    for i, pid in enumerate(starters + subs, 1):
        player_name = element_to_name.get(pid, f"Player {pid}")
        player_score = scores.get(pid, {}).get('total', 0) if pid in scores else 0
        player_pos = all_pos.get(pid, "UNK")
        
        # Get player news/status
        player_news = news.get(player_name, {})
        status = player_news.get('status', 'available')
        start_prob = player_news.get('start_probability', 100) or 100
        
        optimized_lineup.append({
            "id": pid,
            "name": player_name,
            "position": player_pos,
            "score": player_score,
            "status": status,
            "start_probability": start_prob,
            "is_starter": i <= 11
        })
    # Build mappings for display
    pid_to_name: Dict[int, str] = dict(element_to_name)
    name_to_pid: Dict[str, int] = {v: k for k, v in element_to_name.items()}
    
    # Get detailed team information
    team_entry = None
    try:
        team_entry = fpl.get_entry(team_id) if team_id else None
    except Exception as e:
        logger.warning(f"Failed to fetch team entry data: {e}")
    
    # Get team name from entry data if available, otherwise use my_team or default
    team_name = "My Team"
    if team_entry and 'name' in team_entry:
        team_name = team_entry['name']
    elif my_team and 'name' in my_team:
        team_name = my_team['name']
    
    # Get overall rank
    overall_rank = 0
    if team_entry and 'summary_overall_rank' in team_entry:
        overall_rank = team_entry['summary_overall_rank']
    elif my_team and 'summary_overall_rank' in my_team:
        overall_rank = my_team['summary_overall_rank']
    
    # Get total points
    total_points = 0
    if team_entry and 'summary_event_points' in team_entry:
        total_points = team_entry['summary_event_points']
    elif my_team and 'summary_event_points' in my_team:
        total_points = my_team['summary_event_points']
    
    # Get team value and bank value, with robust fallbacks
    raw_team_value = 0
    if my_team:
        raw_team_value = my_team.get('last_deadline_value') or my_team.get('value') or 0
    if (not raw_team_value) and team_entry:
        # entry.value is in tenths of a million
        raw_team_value = team_entry.get('value') or 0
    # Bank already computed in millions (bank_val)
    team_value_formatted = f"£{(raw_team_value or 0)/10:.1f}m"
    bank_value_formatted = f"£{(bank_val or 0.0):.1f}m"
    
    # Get active chips
    chips = []
    if team_entry and 'chips' in team_entry:
        chips = team_entry['chips']
    elif my_team and 'chips' in my_team:
        chips = my_team['chips']
    
    active_chips = [chip["name"] for chip in chips 
                   if chip.get("status_for_entry") == "active"]
    chip_usage = ", ".join(active_chips) if active_chips else "None"
    
    team_summary = {
        "Team Name": team_name,
        "Overall Rank": f"{overall_rank:,}" if overall_rank else "N/A",
        "Total Points": total_points,
        "Team Value": team_value_formatted,
        "Bank": bank_value_formatted,
        "Chip Usage": chip_usage,
    }
    output_dir_path = (Path(custom_report_dir) if custom_report_dir else Path(settings.report_output_dir))

    pdf_path: Path | None = None
    if not appendix_only:
        # Build display scores with fallback to market_scores for any missing players
        display_scores = {pid: {"total": (scores.get(pid, {}) or {}).get("total", market_scores.get(pid, 0))} for pid in starters + subs}
        pdf_path = generate_pdf(
            output_dir=output_dir_path,
            team_summary=team_summary,
            player_scores=display_scores,
            transfers=transfers,
            news_summaries=news,  # Fixed parameter name
            starters=starters,
            subs=subs,
            name_to_pid=name_to_pid,
            pid_to_name=element_to_name,
            elements=elements,  # Pass the elements data for player status
            my_team=my_team,   # Pass the team data for starting 11
            bootstrap=bootstrap  # Pass the bootstrap data for team information
        )
        logger.info("Report generated: %s", pdf_path)

    # Appendix generation (uses fixtures for the next GW if available)
    appendix_path: Path | None = None
    if generate_appendix or appendix_only:
        gw_for_fixtures = team_context.get('next_event') or team_context.get('current_event')
        try:
            fixtures = fpl.get_fixtures(event=gw_for_fixtures) if gw_for_fixtures else fpl.get_fixtures()
        except Exception as e:
            logger.warning(f"Failed to fetch fixtures for appendix: {e}")
            fixtures = []
        appendix_display_scores = {pid: {"total": (scores.get(pid, {}) or {}).get("total", market_scores.get(pid, 0))} for pid in starters + subs} if scores else None
        appendix_path = generate_appendix_pdf(
            output_dir=output_dir_path,
            team_summary=team_summary,
            player_scores=appendix_display_scores,
            elements=elements,
            starters=starters,
            subs=subs,
            bootstrap=bootstrap,
            fixtures=fixtures,
        )
        logger.info("Appendix generated: %s", appendix_path)

    # Return the primary PDF if created; otherwise return appendix path
    return pdf_path or appendix_path


def cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the FPL weekly update.")
    parser.add_argument('--gameweek', type=int, help='Generate predictions for a specific gameweek for backtesting.')
    parser.add_argument('--log-level', choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG'], help='Set log level for this run.')
    parser.add_argument('-q','--quiet', action='store_true', help='Alias for --log-level WARNING.')
    parser.add_argument('-v','--verbose', action='store_true', help='Alias for --log-level INFO.')
    args = parser.parse_args()

    # Resolve desired log level
    level = None
    if args.log_level:
        level = getattr(logging, args.log_level)
    elif args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.INFO
    if level is not None:
        logging.getLogger().setLevel(level)

    run_kwargs = {}
    if args.gameweek:
        run_kwargs['generate_predictions_for_gw'] = args.gameweek

    run_weekly_update(**run_kwargs)


if __name__ == "__main__":
    cli()
