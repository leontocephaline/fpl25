#!/usr/bin/env python3
"""
FPL Optimizer - Main execution script
Orchestrates the complete FPL optimization pipeline from data ingestion to team selection
"""

import argparse
import logging
from pathlib import Path
import yaml
from typing import Dict
import pandas as pd

# This assumes the script is run from the project root directory.
# Adjust if necessary, but this is a common convention.
from src.utils.config import Config, setup_logger
from src.data.fpl_api import FPLAPIClient
from src.data.data_processor import DataProcessor
from src.models.ml_predictor import MLPredictor
from src.optimization.milp_optimizer import MILPOptimizer
from src.strategy.premium_selector import PremiumSelector
from src.strategy.captain_picker import CaptainPicker
from src.strategy.risk_manager import RiskManager
from src.analysis.reporting import Reporting

class FPLOptimizationPipeline:
    """Main pipeline orchestrator for FPL optimization"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        # Use the log_level from config, with a default of 'INFO'
        self.logger = setup_logger("FPLOptimizer", self.config.get('log_level', 'INFO'))
        self.logger.info("Initializing FPLOptimizationPipeline...")

        # Initialize components
        self.api_client = FPLAPIClient(self.config)
        self.data_processor = DataProcessor(self.config)
        self.ml_predictor = MLPredictor(self.config)
        self.optimizer = MILPOptimizer(self.config)
        self.premium_selector = PremiumSelector(self.config)
        self.captain_picker = CaptainPicker(self.config)
        self.risk_manager = RiskManager(self.config)
        self.reporting = Reporting(self.config)
        self.logger.info("All pipeline components initialized.")

    def _generate_stand_in_recommendations(self, optimal_team: Dict, predictions: pd.DataFrame, num_stand_ins: int = 3) -> Dict:
        """Generates stand-in players for each player in the optimal squad."""
        stand_in_recs = {}
        squad_player_ids = {p['id'] for p in optimal_team.get('squad', [])}

        for player in optimal_team.get('squad', []):
            player_id = player['id']
            player_pos = player['position']
            player_price = player['now_cost']

            # Find candidates: same position, not in squad, similar price
            price_margin = self.config.get('stand_in_price_margin', 0.5) * 10  # Margin from config, converted
            candidates = predictions[
                (predictions['position'] == player_pos) &
                (~predictions['id'].isin(squad_player_ids)) &
                (predictions['now_cost'] >= player_price - price_margin) &
                (predictions['now_cost'] <= player_price + price_margin)
            ].copy()

            # Sort by predicted points and select top N
            candidates = candidates.sort_values(by='predicted_points', ascending=False)
            top_candidates = candidates.head(num_stand_ins).to_dict('records')
            stand_in_recs[player_id] = top_candidates
        
        return stand_in_recs

    def run_weekly_optimization(self, current_team_ids=None, free_transfers=1, bank=0.0, retrain=False):
        """Execute complete weekly optimization pipeline"""
        self.logger.info("Starting FPL optimization pipeline")
        self.logger.debug(f"Run params: current_team_ids={current_team_ids}, free_transfers={free_transfers}, bank={bank}, retrain={retrain}")

        try:
            # Step 1: Data Ingestion
            self.logger.info("Fetching FPL data...")
            players_data = self.api_client.get_bootstrap_static()
            fixtures_data = self.api_client.get_fixtures()

            # Filter out unavailable players based on FPL API status
            if 'players' in players_data:
                players_df = players_data['players']
                original_player_count = len(players_df)
                # Status 'a' means available. Others are 'i' (injured), 'd' (doubtful), 's' (suspended), 'u' (unavailable)
                available_players_df = players_df[players_df['status'] == 'a'].copy()
                removed_count = original_player_count - len(available_players_df)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} unavailable players (injured, suspended, etc.).")
                players_data['players'] = available_players_df

            # Load historical data for feature engineering
            historical_data = None
            try:
                # Assuming actuals.csv is in the data directory
                historical_data_path = Path('data/actuals.csv')
                if historical_data_path.exists():
                    historical_data = pd.read_csv(historical_data_path)
                    self.logger.info(f"Loaded historical data from {historical_data_path} for feature engineering.")
                else:
                    self.logger.warning(f"Historical data file not found at {historical_data_path}. Proceeding without lagged features.")
            except Exception as e:
                self.logger.error(f"Error loading historical data: {e}", exc_info=True)


            # Step 2: Data Processing and Feature Engineering
            self.logger.info("Processing data and engineering features...")
            processed_data, feature_lineage, cleaned_stats = self.data_processor.process_all_data(
                players_data, fixtures_data, historical_data=historical_data
            )

            # Step 3: ML Predictions
            self.logger.info("Generating ML predictions...")
            predictions, models_data = self.ml_predictor.predict_player_points(
                processed_data.copy(), 
                feature_lineage, 
                cleaned_stats, 
                self.config.get('optimization_horizon', 4), 
                retrain=retrain
            )

            # Step 4: Premium Player Selection
            self.logger.info("Identifying premium player targets...")
            premium_recommendations = self.premium_selector.identify_premiums(
                processed_data, predictions
            )

            # Step 5: MILP Optimization
            self.logger.info("Running MILP optimization...")
            optimal_team = self.optimizer.optimize_team(
                processed_data, predictions,
                current_team_ids=current_team_ids,
                free_transfers=free_transfers,
                bank=bank
            )

            # Step 6: Captaincy Selection
            self.logger.info("Selecting captain and vice-captain...")
            # Avoid dumping full dicts into logs; summarize instead
            self.logger.debug(
                "Inspecting optimal_team before captain selection: type=%s, keys=%s",
                type(optimal_team), list(optimal_team.keys()) if isinstance(optimal_team, dict) else None
            )
            # Pass the starting XI list (fallback to full squad if missing)
            starting_xi_list = optimal_team.get('starting_xi') or optimal_team.get('squad', [])
            captain_recommendations = self.captain_picker.select_captain(
                starting_xi_list, predictions
            )

            # Step 7: Risk Analysis
            self.logger.info("Performing risk analysis...")
            risk_analysis = self.risk_manager.assess_team_risk(optimal_team, processed_data)

            # Step 8: Reporting
            self.logger.info("Generating PDF report...")
            # The reporting module needs the raw features (X) to show SHAP values for players
            X = self.ml_predictor.prepare_features(processed_data.copy())

            
            # Compile recommendations into a single dictionary for the report
            recommendations = {
                'optimal_team': optimal_team,
                'captaincy': captain_recommendations,
                'premium_recommendations': premium_recommendations,
                'risk_analysis': risk_analysis
            }
            
            recommendations = self.reporting.generate_report(
                df=processed_data, 
                feature_lineage=feature_lineage, 
                models_data=models_data, 
                recommendations=recommendations, 
                X=X
            )
            report_path = recommendations.get('report_path')

            # Step 9: Generate Stand-in Recommendations
            self.logger.info("Generating stand-in recommendations...")
            stand_ins = self._generate_stand_in_recommendations(optimal_team, predictions)
            optimal_team['stand_ins'] = stand_ins

            self.logger.info(f"Optimization complete. Report generated at {report_path}")

            return {
                'optimal_team': optimal_team,
                'captain_recommendations': captain_recommendations,
                'risk_analysis': risk_analysis,
                'report_path': report_path
            }

        except Exception as e:
            self.logger.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FPL Optimization Pipeline.")
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the ML models.')
    parser.add_argument('--current-team', nargs='+', type=int, help='Current team player IDs')
    parser.add_argument('--free-transfers', type=int, default=1, help='Number of free transfers')
    parser.add_argument('--bank', type=float, default=0.0, help='Money in bank')
    parser.add_argument('--disable-shap', action='store_false', dest='enable_shap', help='Disable SHAP value calculations (faster).')
    args = parser.parse_args()

    pipeline = FPLOptimizationPipeline(config_path=args.config)
    results = pipeline.run_weekly_optimization(
        current_team_ids=args.current_team,
        free_transfers=args.free_transfers,
        bank=args.bank,
        retrain=args.retrain
    )

    print("\n=== FPL OPTIMIZATION RESULTS ===")
    print(f"Optimal Team Formation: {results['optimal_team']['formation']}")

    # Handle case where captaincy selection is skipped
    if results['captain_recommendations'] == 'SKIPPED':
        print("Captain: SKIPPED")
        print("Vice-Captain: SKIPPED")
    else:
        captain_name = results['captain_recommendations']['captain']['web_name'] if results['captain_recommendations']['captain'] else 'None'
        vice_captain_name = results['captain_recommendations']['vice_captain']['web_name'] if results['captain_recommendations']['vice_captain'] else 'None'
        print(f"Captain: {captain_name}")
        print(f"Vice-Captain: {vice_captain_name}")

    print(f"Expected Points: {results['optimal_team']['expected_points']:.2f}")
    print(f"Total Cost: £{results['optimal_team']['total_cost'] / 10:.1f}m")

    # Display team details
    if 'squad' in results['optimal_team']:
        # Create a set of starting XI player IDs for quick lookup
        starting_xi_ids = set(player['id'] for player in results['optimal_team'].get('starting_xi', []))

        # Separate squad into starting XI and bench
        starting_xi_players = [player for player in results['optimal_team']['squad'] if player['id'] in starting_xi_ids]
        bench_players = [player for player in results['optimal_team']['squad'] if player['id'] not in starting_xi_ids]

        # Sort players by position and then by points
        position_order = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

        starting_xi_players.sort(key=lambda x: (position_order.get(x['position'], 5), -x.get('predicted_points', 0)))
        bench_players.sort(key=lambda x: (position_order.get(x['position'], 5), -x.get('predicted_points', 0)))

        print("\nStarting XI:")
        for i, player in enumerate(starting_xi_players, 1):
            # Check if player is captain
            captain_marker = ""
            if results['captain_recommendations'] != 'SKIPPED' and results['captain_recommendations']['captain'] and results['captain_recommendations']['vice_captain']:
                if player['id'] == results['captain_recommendations']['captain']['id']:
                    captain_marker = " (C)"
                elif player['id'] == results['captain_recommendations']['vice_captain']['id']:
                    captain_marker = " (VC)"
            print(f"  {i}. {player['web_name']} ({player['position']}) - {player.get('team_name', 'Unknown')} - £{player['now_cost'] / 10:.1f}m - {player.get('predicted_points', 0):.1f} pts{captain_marker}")
            
            # Print stand-ins
            if 'stand_ins' in results['optimal_team'] and player['id'] in results['optimal_team']['stand_ins']:
                stand_ins = results['optimal_team']['stand_ins'][player['id']]
                if stand_ins:
                    print("    \033[94mStand-ins:\033[0m")
                    for si in stand_ins:
                        print(f"      - {si['web_name']} ({si['position']}) - £{si['now_cost'] / 10:.1f}m - {si.get('predicted_points', 0):.1f} pts")


        print("\nBench:")
        for i, player in enumerate(bench_players, 1):
            print(f"  {i}. {player['web_name']} ({player['position']}) - {player.get('team_name', 'Unknown')} - £{player['now_cost'] / 10:.1f}m - {player.get('predicted_points', 0):.1f} pts")
            
            # Print stand-ins
            if 'stand_ins' in results['optimal_team'] and player['id'] in results['optimal_team']['stand_ins']:
                stand_ins = results['optimal_team']['stand_ins'][player['id']]
                if stand_ins:
                    print("    \033[94mStand-ins:\033[0m")
                    for si in stand_ins:
                        print(f"      - {si['web_name']} ({si['position']}) - £{si['now_cost'] / 10:.1f}m - {si.get('predicted_points', 0):.1f} pts")

        # Show team player counts
        team_counts = {}
        for player in results['optimal_team']['squad']:
            team_name = player.get('team_name', 'Unknown')
            team_counts[team_name] = team_counts.get(team_name, 0) + 1

        print("\nTeam Player Counts (Max 3 per team):")
        for team, count in sorted(team_counts.items()):
            status = "(OK)" if count <= 3 else "(!!)"
            print(f"  {team}: {count} players {status}")

    # Additional formation analysis
    if 'squad' in results['optimal_team']:
        #gk_count = sum(1 for player in results['optimal_team']['squad'] if player['position'] == 'GK')
        print(f"\nSquad Composition: full squad of 15 players")
