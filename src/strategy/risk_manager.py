"""
Risk Manager - Analyzes player risk factors for captain selection
"""

import pandas as pd
import logging
from typing import Dict, List


class RiskManager:
    """Analyzes player risk factors to inform captain selection"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_player_risks(self, players: List[Dict]) -> Dict:
        """Analyze risk factors for a list of players"""
        try:
            if not players:
                return {}
            
            risk_analysis = {}
            
            for player in players:
                player_id = player.get('id')
                if not player_id:
                    continue
                
                # Calculate risk score based on various factors
                risk_score = self._calculate_risk_score(player)
                
                risk_analysis[player_id] = {
                    'risk_score': risk_score,
                    'injury_risk': player.get('chance_of_playing_next_round', 100),
                    'form_stability': self._assess_form_stability(player),
                    'fixture_difficulty': player.get('fixture_difficulty_next_4', 3.0),
                    'minutes_consistency': self._assess_minutes_consistency(player)
                }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {str(e)}")
            return {}
    
    def assess_team_risk(self, team_data: Dict, processed_data: pd.DataFrame) -> Dict:
        """Assess overall team risk based on player risks and team composition"""
        try:
            if not team_data or 'starting_xi' not in team_data:
                return {'overall_risk': 'unknown', 'risk_factors': {}}
            
            starting_xi = team_data['starting_xi']
            if not starting_xi:
                return {'overall_risk': 'unknown', 'risk_factors': {}}
            
            # Analyze risks for starting XI players
            player_risks = self.analyze_player_risks(starting_xi)
            
            # Calculate overall team risk
            if player_risks:
                avg_risk_score = sum(risk['risk_score'] for risk in player_risks.values()) / len(player_risks)
                overall_risk = 'high' if avg_risk_score > 60 else 'medium' if avg_risk_score > 30 else 'low'
            else:
                avg_risk_score = 50.0
                overall_risk = 'unknown'
            
            return {
                'overall_risk': overall_risk,
                'average_risk_score': avg_risk_score,
                'player_risks': player_risks,
                'risk_factors': {
                    'high_risk_players': len([r for r in player_risks.values() if r['risk_score'] > 60]),
                    'medium_risk_players': len([r for r in player_risks.values() if 30 < r['risk_score'] <= 60]),
                    'low_risk_players': len([r for r in player_risks.values() if r['risk_score'] <= 30])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in team risk assessment: {str(e)}")
            return {'overall_risk': 'error', 'risk_factors': {}}
    
    def _calculate_risk_score(self, player: Dict) -> float:
        """Calculate overall risk score for a player (0-100, lower is better)"""
        risk_score = 50.0  # Base score
        
        # Injury risk factor (0-100, higher means higher injury risk)
        injury_risk = 100 - player.get('chance_of_playing_next_round', 100)
        risk_score += injury_risk * 0.3
        
        # Form instability factor
        form_stability = self._assess_form_stability(player)
        risk_score += (100 - form_stability) * 0.2
        
        # Fixture difficulty factor (1-5, higher is harder)
        fixture_difficulty = player.get('fixture_difficulty_next_4', 3.0)
        risk_score += (fixture_difficulty - 3.0) * 10
        
        # Minutes consistency factor
        minutes_consistency = self._assess_minutes_consistency(player)
        risk_score += (100 - minutes_consistency) * 0.2
        
        return max(0, min(100, risk_score))  # Clamp between 0-100
    
    def _assess_form_stability(self, player: Dict) -> float:
        """Assess form stability (0-100, higher is more stable)"""
        # This is a simplified assessment
        # In a real implementation, this would analyze form trends
        form = player.get('form', 0)
        points_per_game = player.get('points_per_game', 0)
        
        # Players with consistent points are more stable
        if points_per_game > 0:
            return min(100, 50 + (form / points_per_game) * 10)
        return 50
    
    def _assess_minutes_consistency(self, player: Dict) -> float:
        """Assess minutes consistency (0-100, higher is more consistent)"""
        # This is a simplified assessment
        # In a real implementation, this would analyze minutes history
        minutes = player.get('minutes', 0)
        
        # Players who play more minutes are more consistent
        return min(100, minutes / 90 * 20) if minutes > 0 else 0
