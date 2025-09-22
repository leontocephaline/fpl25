# src/optimization/milp_optimizer.py - Core MILP optimization engine
"""
MILP Optimizer - Mixed Integer Linear Programming optimization for FPL team selection
Implements sophisticated optimization with formation constraints, premium limits, and transfer logic
"""

import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class MILPOptimizer:
    """MILP-based optimizer for FPL team selection"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load all optimization parameters from config
        self._load_config_params()

    def _load_config_params(self):
        """Load optimization parameters from the config object."""
        self.budget = self.config.get('optimization', {}).get('budget', 100.0)
        self.squad_size = self.config.get('optimization.squad_size')
        self.starting_xi_size = self.config.get('optimization.starting_xi_size')
        self.max_per_team = self.config.get('optimization.max_per_team')
        self.position_constraints = self.config.get('optimization.positions')
        self.bench_weights = self.config.get('optimization.bench_weights')
        self.formation_preferences = self.config.get('optimization.formations')
        self.premium_limit = self.config.get('strategy.premium_limit')
        self.premium_thresholds = self.config.get('strategy.premium_thresholds')

    def _get_position_name(self, element_type: int) -> str:
        """Map element_type to position name."""
        return {
            1: 'GKP',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'
        }.get(element_type, 'UNKNOWN')
    
    def optimize_team(self, players_df: pd.DataFrame, predictions_df: pd.DataFrame,
                     current_team_ids: Optional[List[int]] = None,
                     free_transfers: int = 1, bank: float = 0.0) -> Dict:
        """Optimize team selection with transfer constraints"""
        self.logger.info("Starting MILP team optimization")
        
        # Prepare optimization data
        opt_data = self._prepare_optimization_data(players_df, predictions_df)
        
        # Run optimization scenarios
        scenarios = {}
        
        # Scenario 1: No transfers (baseline)
        if current_team_ids:
            scenarios['no_transfers'] = self._optimize_lineup_only(
                opt_data, current_team_ids
            )
        
        # Scenario 2: Use free transfers
        scenarios['free_transfers'] = self._optimize_with_transfers(
            opt_data, current_team_ids, free_transfers, bank, transfer_cost=0
        )
        
        # Scenario 3: Take a hit (-4 points)
        if free_transfers < 2:
            scenarios['with_hit'] = self._optimize_with_transfers(
                opt_data, current_team_ids, free_transfers + 1, bank, transfer_cost=4
            )
        
        # Select best scenario
        best_scenario = self._select_best_scenario(scenarios)
        
        # Add formation analysis
        best_scenario = self._add_formation_analysis(best_scenario, opt_data)
        
        # Add captain recommendation
        best_scenario = self._add_captain_recommendation(best_scenario, predictions_df)
        
        self.logger.info("MILP optimization completed")
        return best_scenario
    
    def _prepare_optimization_data(self, players_df: pd.DataFrame, 
                                 predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for optimization"""
        # Merge player data with predictions
        opt_data = players_df.merge(
            predictions_df[['id', 'predicted_points', 'prediction_confidence']], 
            on='id', how='left'
        )
        
        # Fill missing predictions with form-based estimates
        opt_data['predicted_points'] = opt_data['predicted_points'].fillna(
            opt_data['form_numeric'] * self.config.optimization_horizon * 0.8
        )
        
        # Add optimization-specific features
        opt_data['value_score'] = opt_data['predicted_points'] / opt_data['now_cost']

        # Convert cost back to integer for optimizer
        opt_data['now_cost'] = (opt_data['now_cost'] * 10).astype(int)
        
        # The 'is_premium' flag is already calculated in the DataProcessor.
        # The line below was causing a crash and is redundant.
        # We rely on the `is_premium` column from the input DataFrame.
        
        # Position mapping
        opt_data['position'] = opt_data['element_type'].apply(self._get_position_name)
        
        # Ensure team_name is preserved (if available)
        if 'team_name' not in opt_data.columns and 'team' in opt_data.columns:
            # If team_name is missing but team ID is present, we'll add it back later
            pass
        
        return opt_data.sort_values('predicted_points', ascending=False)
    
    def _optimize_with_transfers(self, opt_data: pd.DataFrame, 
                               current_team_ids: Optional[List[int]],
                               available_transfers: int, bank: float,
                               transfer_cost: float = 0) -> Dict:
        """Optimize team with transfer constraints"""
        
        # Create optimization problem
        prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)
        
        # Decision variables
        players = opt_data['id'].tolist()
        
        # Squad selection variables (binary)
        squad_vars = pulp.LpVariable.dicts("squad", players, cat='Binary')
        
        # Starting XI variables (binary)
        starting_vars = pulp.LpVariable.dicts("starting", players, cat='Binary')
        
        # Captain variable (binary)
        captain_vars = pulp.LpVariable.dicts("captain", players, cat='Binary')
        
        # Transfer variables (if applicable)
        if current_team_ids:
            transfer_in_vars = pulp.LpVariable.dicts("transfer_in", players, cat='Binary')
            transfer_out_vars = pulp.LpVariable.dicts("transfer_out", players, cat='Binary')
        
        # Objective function: Maximize expected points
        objective = 0
        
        for player_id in players:
            player_data = opt_data[opt_data['id'] == player_id].iloc[0]
            predicted_points = player_data['predicted_points']
            
            # Starting XI points (full points)
            objective += predicted_points * starting_vars[player_id]
            
            # Bench points (weighted)
            # This is a simplified placeholder; a more detailed bench weight assignment
            # would require ordering the bench, which adds complexity. Using an average.
            # The first weight is for the sub-GK, others for outfield players.
            avg_bench_weight = (self.bench_weights[1] + self.bench_weights[2] + self.bench_weights[3]) / 3
            objective += predicted_points * avg_bench_weight * (squad_vars[player_id] - starting_vars[player_id])
            
            # Captain points (double points for captain)
            objective += predicted_points * captain_vars[player_id]
        
        # Subtract transfer cost
        objective -= transfer_cost
        
        prob += objective
        
        # Constraints
        self._add_squad_constraints(prob, opt_data, squad_vars, starting_vars, captain_vars)
        self._add_budget_constraint(prob, opt_data, squad_vars, bank)
        self._add_formation_constraints(prob, opt_data, starting_vars)
        self._add_premium_constraints(prob, opt_data, squad_vars)
        
        if current_team_ids:
            self._add_transfer_constraints(
                prob, opt_data, squad_vars, transfer_in_vars, transfer_out_vars,
                current_team_ids, available_transfers
            )
        
        # Solve optimization
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            return self._extract_solution(
                prob, opt_data, squad_vars, starting_vars, captain_vars,
                current_team_ids, available_transfers if current_team_ids else 0
            )
        else:
            self.logger.error(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
            return self._get_fallback_solution(opt_data)
    
    def _add_squad_constraints(self, prob, opt_data: pd.DataFrame, 
                             squad_vars, starting_vars, captain_vars):
        """Add basic squad constraints"""
        players = opt_data['id'].tolist()
        
        # Squad size constraints
        prob += pulp.lpSum([squad_vars[pid] for pid in players]) == self.squad_size
        prob += pulp.lpSum([starting_vars[pid] for pid in players]) == self.starting_xi_size
        
        # Position constraints for squad
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        for element_type, pos_name in position_map.items():
            pos_players = opt_data[opt_data['element_type'] == element_type]['id'].tolist()
            constraints = self.position_constraints[pos_name]
            
            # Squad count
            prob += pulp.lpSum(squad_vars[pid] for pid in pos_players) == constraints['squad_count']
            
            # Starting XI count (if specified)
            if 'starting_count' in constraints:
                prob += pulp.lpSum(starting_vars[pid] for pid in pos_players) == constraints['starting_count']
        
        # Team constraints (max players per team)
        for team_id in opt_data['team'].unique():
            team_players = opt_data[opt_data['team'] == team_id]['id'].tolist()
            prob += pulp.lpSum([squad_vars[pid] for pid in team_players]) <= self.max_per_team
        
        # Starting XI must be subset of squad
        for pid in players:
            prob += starting_vars[pid] <= squad_vars[pid]
        
        # Captain constraints
        prob += pulp.lpSum([captain_vars[pid] for pid in players]) == 1
        for pid in players:
            prob += captain_vars[pid] <= starting_vars[pid]
    
    def _add_budget_constraint(self, prob, opt_data: pd.DataFrame, squad_vars, bank: float):
        """Add budget constraint"""
        total_cost = 0
        for _, player in opt_data.iterrows():
            total_cost += player['now_cost'] * squad_vars[player['id']]
        
        # Budget is in millions, now_cost is in 0.1 millions. Scale budget by 10.
        # Also scale bank, assuming it's in millions.
        prob += total_cost <= self.budget * 10 + (bank * 10)
    
    def _add_formation_constraints(self, prob, opt_data: pd.DataFrame, starting_vars):
        """Add formation preference constraints"""
        # Flexible formation constraints
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        for element_type, pos_name in position_map.items():
            pos_players = opt_data[opt_data['element_type'] == element_type]['id'].tolist()
            constraints = self.position_constraints[pos_name]

            if 'starting_min' in constraints:
                prob += pulp.lpSum(starting_vars[pid] for pid in pos_players) >= constraints['starting_min']
            if 'starting_max' in constraints:
                prob += pulp.lpSum(starting_vars[pid] for pid in pos_players) <= constraints['starting_max']
    
    def _add_premium_constraints(self, prob, opt_data: pd.DataFrame, squad_vars):
        """Add premium player constraints"""
        premium_players = opt_data[opt_data['is_premium'] == True]['id'].tolist()
        
        # Maximum 2 premium players total
        prob += pulp.lpSum([squad_vars[pid] for pid in premium_players]) <= self.premium_limit
    
    def _add_transfer_constraints(self, prob, opt_data: pd.DataFrame, 
                                squad_vars, transfer_in_vars, transfer_out_vars,
                                current_team_ids: List[int], available_transfers: int):
        """Add transfer-related constraints"""
        players = opt_data['id'].tolist()
        
        # Filter current_team_ids to only include players present in opt_data
        valid_current_team_ids = [pid for pid in current_team_ids if pid in players]
        if len(valid_current_team_ids) != len(current_team_ids):
            self.logger.warning("Some players in the current team are not available in the game data and will be ignored.")

        # Adjust transfer balance for incomplete squads. If the squad is short, we must buy players.
        num_missing_players = self.squad_size - len(valid_current_team_ids)
        prob += pulp.lpSum(transfer_in_vars[pid] for pid in players) - pulp.lpSum(transfer_out_vars[pid] for pid in players) == num_missing_players

        # Transfer limit constraint
        prob += pulp.lpSum([transfer_in_vars[pid] for pid in players]) <= available_transfers

        # Current team constraint
        for pid in players:
            if pid in valid_current_team_ids:
                # Player is currently in team
                prob += squad_vars[pid] + transfer_out_vars[pid] >= 1
                prob += transfer_in_vars[pid] == 0
            else:
                # Player is not currently in team
                prob += squad_vars[pid] <= transfer_in_vars[pid]
                prob += transfer_out_vars[pid] == 0

        # Link squad selection to transfers
        if len(valid_current_team_ids) != len(current_team_ids):
            self.logger.warning("Some players in the current team are not available in the game data and will be ignored.")

        for pid in valid_current_team_ids:
            prob += squad_vars[pid] == 1 - transfer_out_vars[pid]
    
    def _extract_solution(self, prob, opt_data: pd.DataFrame, 
                         squad_vars, starting_vars, captain_vars,
                         current_team_ids: Optional[List[int]] = None,
                         transfers_used: int = 0) -> Dict:
        """Extract optimization solution"""
        
        # Get selected players
        selected_squad = []
        selected_starting = []
        captain_id = None
        
        for pid in opt_data['id']:
            if squad_vars[pid].value() == 1:
                player_data = opt_data[opt_data['id'] == pid].iloc[0]
                selected_squad.append(player_data.to_dict())
                
                if starting_vars[pid].value() == 1:
                    selected_starting.append(player_data.to_dict())
                
                if captain_vars[pid].value() == 1:
                    captain_id = pid
        
        # Calculate costs and points
        total_cost = sum(p['now_cost'] for p in selected_squad)
        expected_points = sum(p['predicted_points'] for p in selected_starting)
        expected_points += sum(p['predicted_points'] * w for p, w in 
                             zip(selected_squad[11:], self.bench_weights))
        
        # Add captain bonus
        if captain_id:
            captain_data = opt_data[opt_data['id'] == captain_id].iloc[0]
            expected_points += captain_data['predicted_points']  # Double points
        
        # Formation analysis
        formation = self._analyze_formation(selected_starting)
        
        return {
            'squad': selected_squad,
            'starting_xi': selected_starting,
            'captain_id': captain_id,
            'formation': formation,
            'total_cost': total_cost,
            'expected_points': expected_points - transfers_used * 4,  # Account for transfer costs
            'transfers_used': transfers_used,
            'optimization_status': 'optimal',
            'solver_time': prob.solutionTime if hasattr(prob, 'solutionTime') else 0
        }
    
    def _analyze_formation(self, starting_xi: List[Dict]) -> str:
        """Analyze the formation of the starting XI"""
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for player in starting_xi:
            pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            position = pos_map[player['element_type']]
            position_counts[position] += 1
        
        # Format as formation string (excluding GK)
        return f"{position_counts['DEF']}-{position_counts['MID']}-{position_counts['FWD']}"
    
    def _optimize_lineup_only(self, opt_data: pd.DataFrame, 
                            current_team_ids: List[int]) -> Dict:
        """Optimize only the starting lineup from current squad"""
        current_squad = opt_data[opt_data['id'].isin(current_team_ids)]
        
        if len(current_squad) != 15:
            self.logger.warning(f"Current squad has {len(current_squad)} players, expected 15")
            return self._get_fallback_solution(opt_data)
        
        # Simple lineup optimization using pulp
        prob = pulp.LpProblem("FPL_Lineup_Optimization", pulp.LpMaximize)
        
        # Starting XI variables
        starting_vars = pulp.LpVariable.dicts("starting", current_team_ids, cat='Binary')
        captain_vars = pulp.LpVariable.dicts("captain", current_team_ids, cat='Binary')
        
        # Objective: maximize expected points
        objective = 0
        for pid in current_team_ids:
            player_data = current_squad[current_squad['id'] == pid].iloc[0]
            predicted_points = player_data['predicted_points']
            
            objective += predicted_points * starting_vars[pid]
            objective += predicted_points * captain_vars[pid]  # Captain bonus
        
        prob += objective
        
        # Constraints
        prob += pulp.lpSum([starting_vars[pid] for pid in current_team_ids]) == 11
        prob += pulp.lpSum([captain_vars[pid] for pid in current_team_ids]) == 1
        
        # Position constraints
        for element_type in [1, 2, 3, 4]:
            type_players = current_squad[current_squad['element_type'] == element_type]['id'].tolist()
            if element_type == 1:  # GK
                prob += pulp.lpSum([starting_vars[pid] for pid in type_players]) == 1
            elif element_type == 2:  # DEF
                prob += pulp.lpSum([starting_vars[pid] for pid in type_players]) >= 3
            elif element_type == 3:  # MID
                prob += pulp.lpSum([starting_vars[pid] for pid in type_players]) >= 1
            elif element_type == 4:  # FWD
                prob += pulp.lpSum([starting_vars[pid] for pid in type_players]) >= 1
        
        # Captain must be in starting XI
        for pid in current_team_ids:
            prob += captain_vars[pid] <= starting_vars[pid]
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            # Extract lineup
            starting_xi = []
            captain_id = None
            
            for pid in current_team_ids:
                if starting_vars[pid].value() == 1:
                    player_data = current_squad[current_squad['id'] == pid].iloc[0]
                    starting_xi.append(player_data.to_dict())
                
                if captain_vars[pid].value() == 1:
                    captain_id = pid
            
            formation = self._analyze_formation(starting_xi)
            expected_points = sum(p['predicted_points'] for p in starting_xi)
            if captain_id:
                captain_data = current_squad[current_squad['id'] == captain_id].iloc[0]
                expected_points += captain_data['predicted_points']
            
            return {
                'squad': current_squad.to_dict('records'),
                'starting_xi': starting_xi,
                'captain_id': captain_id,
                'formation': formation,
                'total_cost': current_squad['now_cost'].sum(),
                'expected_points': expected_points,
                'transfers_used': 0,
                'optimization_status': 'optimal'
            }
        
        return self._get_fallback_solution(current_squad)
    
    def _select_best_scenario(self, scenarios: Dict) -> Dict:
        """Select the best optimization scenario"""
        if not scenarios:
            return {}
        
        # Compare scenarios by expected points
        best_scenario = None
        best_points = -float('inf')
        
        for scenario_name, scenario_data in scenarios.items():
            if 'expected_points' in scenario_data:
                if scenario_data['expected_points'] > best_points:
                    best_points = scenario_data['expected_points']
                    best_scenario = scenario_data
                    best_scenario['scenario'] = scenario_name
        
        return best_scenario or list(scenarios.values())[0]
    
    def _add_formation_analysis(self, solution: Dict, opt_data: pd.DataFrame) -> Dict:
        """Add detailed formation analysis"""
        if 'starting_xi' not in solution:
            return solution
        
        starting_xi = solution['starting_xi']
        formation_str = solution.get('formation', 'Unknown')
        
        # Add formation preference score
        formation_score = 1.0
        if formation_str == '3-5-2':
            formation_score = 1.0
        elif formation_str == '3-4-3':
            formation_score = 0.8
        elif formation_str in ['4-5-1', '4-4-2']:
            formation_score = 0.7
        else:
            formation_score = 0.5
        
        solution['formation_score'] = formation_score
        solution['formation_preference'] = 'High' if formation_score >= 0.8 else 'Medium' if formation_score >= 0.6 else 'Low'
        
        return solution
    
    def _add_captain_recommendation(self, solution: Dict, predictions_df: pd.DataFrame) -> Dict:
        """Add captain and vice-captain recommendations"""
        if 'starting_xi' not in solution or 'captain_id' not in solution:
            return solution
        
        starting_xi = solution['starting_xi']
        
        # Sort starting XI by predicted points
        starting_players = []
        for player in starting_xi:
            pred_data = predictions_df[predictions_df['id'] == player['id']]
            if not pred_data.empty:
                predicted_points = pred_data.iloc[0]['predicted_points']
            else:
                predicted_points = player.get('predicted_points', 0)
            
            starting_players.append({
                'id': player['id'],
                'name': player['web_name'],
                'predicted_points': predicted_points,
                'position': player.get('position', ''),
                'is_premium': player.get('is_premium', False)
            })
        
        starting_players.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Captain and vice-captain
        captain = starting_players[0] if starting_players else None
        vice_captain = starting_players[1] if len(starting_players) > 1 else None
        
        solution['captain_recommendation'] = {
            'captain': captain,
            'vice_captain': vice_captain,
            'alternatives': starting_players[2:5] if len(starting_players) > 2 else []
        }
        
        return solution
    
    def _get_fallback_solution(self, opt_data: pd.DataFrame) -> Dict:
        """Generate fallback solution when optimization fails"""
        self.logger.warning("Using fallback solution due to optimization failure")
        
        # Simple greedy selection by value
        selected_squad = []
        budget_remaining = self.budget
        
        # Position requirements
        position_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # GK, DEF, MID, FWD
        position_limits = {1: 2, 2: 5, 3: 5, 4: 3}
        
        # Sort by value score
        sorted_players = opt_data.sort_values('value_score', ascending=False)
        
        for _, player in sorted_players.iterrows():
            position = player['element_type']
            cost = player['now_cost']
            
            # Check constraints
            if (len(selected_squad) < 15 and 
                position_counts[position] < position_limits[position] and
                cost <= budget_remaining):
                
                selected_squad.append(player.to_dict())
                budget_remaining -= cost
                position_counts[position] += 1
        
        # Select starting XI (top 11 by predicted points)
        squad_df = pd.DataFrame(selected_squad)
        starting_xi = squad_df.nlargest(11, 'predicted_points').to_dict('records')
        
        return {
            'squad': selected_squad,
            'starting_xi': starting_xi,
            'captain_id': starting_xi[0]['id'] if starting_xi else None,
            'formation': 'Unknown',
            'total_cost': sum(p['now_cost'] for p in selected_squad),
            'expected_points': sum(p['predicted_points'] for p in starting_xi),
            'transfers_used': 0,
            'optimization_status': 'fallback'
        }

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization parameters and constraints"""
        return {
            'budget': self.budget,
            'squad_size': self.squad_size,
            'starting_xi_size': self.starting_xi_size,
            'bench_weights': self.bench_weights,
            'formation_preferences': self.formation_preferences,
            'premium_limit': 2,
            'max_per_team': 3
        }
