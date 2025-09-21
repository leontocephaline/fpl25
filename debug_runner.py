import sys
import os

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("--- Starting debug runner ---")

try:
    print("Importing fpl_weekly_updater modules one by one...")
    
    print("Importing fpl_client...")
    from fpl_weekly_updater.apis.fpl_client import FPLClient
    print("Importing fpl_auth...")
    from fpl_weekly_updater.apis.fpl_auth import get_password, fetch_picks
    print("Importing fpl_browser_login...")
    from fpl_weekly_updater.apis.fpl_browser_login import login_and_get_cookie
    print("Importing perplexity_client...")
    from fpl_weekly_updater.apis.perplexity_client import PerplexityClient
    print("Importing betting_client...")
    from fpl_weekly_updater.apis.betting_client import BettingClient
    print("Importing player_scorer...")
    from fpl_weekly_updater.analysis.player_scorer import PlayerInputs, score_player
    print("Importing news_analyzer...")
    from fpl_weekly_updater.analysis.news_analyzer import analyze_players
    print("Importing transfer_optimizer...")
    from fpl_weekly_updater.analysis.transfer_optimizer import recommend_transfers
    print("Importing settings...")
    from fpl_weekly_updater.config.settings import load_settings
    print("Importing pdf_generator...")
    from fpl_weekly_updater.reporting.pdf_generator import generate_pdf
    print("All imports successful.")

    print("Importing run_weekly_update...")
    from fpl_weekly_updater.main import run_weekly_update
    print("Executing run_weekly_update()...")
    run_weekly_update()
    print("--- Debug runner finished ---")

except Exception as e:
    print(f"!!! An error occurred: {e} !!!")
    import traceback
    traceback.print_exc()
