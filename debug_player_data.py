import json
import logging
import os
import sys
from pathlib import Path

try:
    from fpl_weekly_updater.apis.fpl_auth import get_team_data
    from fpl_weekly_updater.config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Set up logging
log_file = Path('debug_player_processing.log')
if log_file.exists():
    log_file.unlink()  # Remove old log file if it exists

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Log environment info
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Script location: {os.path.abspath(__file__)}")

def load_bootstrap_data():
    """Load bootstrap data from file."""
    bootstrap_path = Path('data/bootstrap.json')
    if not bootstrap_path.exists():
        logger.error(f"Bootstrap file not found at {bootstrap_path.absolute()}")
        return None, None
        
    try:
        with open(bootstrap_path, 'r', encoding='utf-8') as f:
            bootstrap = json.load(f)
            
        elements = {el['id']: el for el in bootstrap.get('elements', [])}
        teams = {t['id']: t for t in bootstrap.get('teams', [])}
        logger.info(f"Loaded {len(elements)} players and {len(teams)} teams from bootstrap data")
        return elements, teams
        
    except Exception as e:
        logger.error(f"Error loading bootstrap data: {e}", exc_info=True)
        return None, None

def debug_player_processing():
    """Debug function to test player data processing."""
    try:
        # Get team data
        logger.info("Fetching team data...")
        team_data = get_team_data()
        
        if not team_data:
            logger.error("No team data returned from get_team_data()")
            return
            
        if "picks" not in team_data:
            logger.error(f"No 'picks' key in team data. Available keys: {list(team_data.keys())}")
            return
            
        picks = team_data['picks']
        logger.info(f"Found {len(picks)} players in team")
        logger.debug(f"Team data structure: {json.dumps(team_data, indent=2, default=str)[:1000]}...")
        
        # Load bootstrap data
        elements, teams = load_bootstrap_data()
        if not elements or not teams:
            logger.error("Failed to load bootstrap data, cannot continue")
            return
        
        # Process each player
        for i, pick in enumerate(picks, 1):
            try:
                pid = pick.get('element')
                if not pid:
                    logger.warning(f"Pick {i}: No element ID found")
                    continue
                    
                logger.info(f"\nProcessing player {i} (ID: {pid}):")
                
                # Get player data
                player = elements.get(int(pid))
                if not player:
                    logger.warning(f"  No player data found for ID {pid}")
                    continue
                    
                # Log player details
                name = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
                team_id = player.get('team')
                team_name = teams.get(team_id, {}).get('name', f'Unknown (ID: {team_id})') if team_id is not None else 'Unknown'
                
                logger.info(f"  Name: {name}")
                logger.info(f"  Position: {player.get('element_type')} ({get_position_name(player.get('element_type'))})")
                logger.info(f"  Team: {team_name}")
                logger.info(f"  Cost: {player.get('now_cost', 0)/10}M")
                logger.info(f"  Selected by: {player.get('selected_by_percent')}%")
                logger.info(f"  Web name: {player.get('web_name')}")
                logger.info(f"  News: {player.get('news', 'No news')}")
                logger.info(f"  Status: {player.get('status')}")
                logger.info(f"  Chance of playing: {player.get('chance_of_playing_next_round')}%")
                
                # Check if this is a goalkeeper
                if player.get('element_type') == 1:
                    logger.info("  This is a goalkeeper!")
                    logger.info(f"  Saves: {player.get('saves')}")
                    logger.info(f"  Clean sheets: {player.get('clean_sheets')}")
                    logger.info(f"  Goals conceded: {player.get('goals_conceded')}")
                
            except Exception as e:
                logger.error(f"Error processing pick {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in debug_player_processing: {str(e)}", exc_info=True)

def get_position_name(position_id):
    """Convert position ID to position name."""
    position_map = {
        1: 'Goalkeeper',
        2: 'Defender',
        3: 'Midfielder',
        4: 'Forward'
    }
    return position_map.get(position_id, f'Unknown ({position_id})')

if __name__ == "__main__":
    logger.info("Starting player data debug script")
    try:
        debug_player_processing()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    logger.info("Debug script finished")
