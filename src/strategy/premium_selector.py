# src/strategy/premium_selector.py - Premium player identification
"""
Premium Selector - Identifies and evaluates premium player options
Focuses on high-value players (£8.0m+) with emphasis on nailed minutes and set pieces
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class PremiumSelector:
    """Identifies optimal premium player selections"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Premium thresholds by position
        self.premium_thresholds = {
            1: 5.5,  # GK - rare premium goalkeepers
            2: 6.0,  # DEF - premium defenders
            3: 8.0,  # MID - premium midfielders
            4: 8.0   # FWD - premium forwards
        }
        
        # Premium evaluation weights
        self.evaluation_weights = {
            'predicted_points': 0.35,
            'minutes_certainty': 0.25,
            'set_piece_involvement': 0.20,
            'form_consistency': 0.20
        }
    
    def identify_premiums(self, players_df: pd.DataFrame, 
                         predictions_df: pd.DataFrame) -> Dict:
        """Identify and rank premium player options"""
        self.logger.info("Identifying premium player targets")
        
        # Merge data
        # Prepare predictions for merge by renaming player_id to id
        temp_preds_df = predictions_df[['player_id', 'predicted_points', 'prediction_confidence']].rename(
            columns={'player_id': 'id'}
        )
        df = players_df.merge(
            temp_preds_df, 
            on='id', 
            how='left'
        )
        
        # Identify premium players
        premium_players = self._filter_premium_players(df)
        
        if premium_players.empty:
            return {'premium_options': [], 'recommendations': {}}
        
        # Evaluate premium options
        premium_evaluations = self._evaluate_premium_players(premium_players)
        
        # Generate recommendations
        recommendations = self._generate_premium_recommendations(premium_evaluations)
        
        return {
            'premium_options': premium_evaluations.to_dict('records'),
            'recommendations': recommendations,
            'evaluation_criteria': self.evaluation_weights
        }
    
    def _filter_premium_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter players meeting premium criteria"""
        premium_mask = df.apply(
            lambda row: row['now_cost'] >= self.premium_thresholds.get(row['element_type'], 8.0),
            axis=1
        )
        
        # Focus on MID and FWD premiums as per strategy
        position_mask = df['element_type'].isin([3, 4])  # MID, FWD only
        
        premium_players = df[premium_mask & position_mask].copy()
        
        # Additional filters for quality
        quality_filters = (
            (premium_players['total_points'] > 50) |  # Proven performers
            (premium_players['form_numeric'] > 4)     # In-form players
        )
        
        return premium_players[quality_filters]
    
    def _evaluate_premium_players(self, premium_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate premium players across multiple criteria"""
        
        # 1. Predicted Points Score (normalized)
        max_predicted = premium_df['predicted_points'].max()
        premium_df['points_score'] = premium_df['predicted_points'] / max_predicted if max_predicted > 0 else 0
        
        # 2. Minutes Certainty Score
        premium_df['minutes_score'] = self._calculate_minutes_score(premium_df)
        
        # 3. Set Piece Involvement Score
        premium_df['set_piece_score'] = self._calculate_set_piece_score(premium_df)
        
        # 4. Form Consistency Score
        premium_df['consistency_score'] = self._calculate_consistency_score(premium_df)
        
        # Overall Premium Score
        premium_df['premium_score'] = (
            premium_df['points_score'] * self.evaluation_weights['predicted_points'] +
            premium_df['minutes_score'] * self.evaluation_weights['minutes_certainty'] +
            premium_df['set_piece_score'] * self.evaluation_weights['set_piece_involvement'] +
            premium_df['consistency_score'] * self.evaluation_weights['form_consistency']
        )
        
        # Add premium-specific metrics
        premium_df['points_per_million'] = premium_df['predicted_points'] / premium_df['now_cost']
        premium_df['ownership_differential'] = 50 - premium_df['selected_by_percent']  # Lower = more differential
        
        return premium_df.sort_values('premium_score', ascending=False)
    
    def _calculate_minutes_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate minutes certainty score"""
        # Based on minutes per game and nailed status
        minutes_per_game = df['minutes_per_game'].fillna(0)
        
        # Score based on minutes reliability
        score = np.where(minutes_per_game >= 80, 1.0,
                np.where(minutes_per_game >= 70, 0.8,
                np.where(minutes_per_game >= 60, 0.6,
                np.where(minutes_per_game >= 45, 0.4, 0.2))))
        
        # Bonus for being clearly nailed
        nailed_bonus = df['is_nailed'].fillna(False).astype(int) * 0.2
        
        return np.clip(score + nailed_bonus, 0, 1)
    
    def _calculate_set_piece_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate set piece involvement score"""
        # Penalties are most valuable
        penalty_score = np.where(df['penalties_taken'] > 0, 0.6, 0)
        
        # Free kicks and corners
        fk_score = np.where(df['direct_freekicks_taken'] > 0, 0.3, 0)
        corner_score = np.where(df['corners_and_indirect_freekicks_taken'] > 0, 0.2, 0)
        
        # Combine scores
        total_score = penalty_score + fk_score + corner_score
        
        return np.clip(total_score, 0, 1)
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate form consistency score"""
        # Form score (recent performance)
        form_score = np.clip(df['form_numeric'].fillna(0) / 10, 0, 1)
        
        # Consistency based on coefficient of variation (if available)
        # For now, use form as a proxy
        consistency_bonus = np.where(form_score > 0.5, 0.2, 0)
        
        return np.clip(form_score + consistency_bonus, 0, 1)
    
    def _generate_premium_recommendations(self, premium_df: pd.DataFrame) -> Dict:
        """Generate premium player recommendations"""
        
        if premium_df.empty:
            return {}
        
        # Top recommendations
        top_premiums = premium_df.head(5).to_dict('records')
        
        # Position-specific recommendations
        mid_premiums = premium_df[premium_df['element_type'] == 3].head(3)
        fwd_premiums = premium_df[premium_df['element_type'] == 4].head(3)
        
        # Value picks (high points per million)
        value_premiums = premium_df.nlargest(3, 'points_per_million')
        
        # Differential picks (low ownership)
        differential_premiums = premium_df[
            premium_df['selected_by_percent'] < 15
        ].head(3)
        
        # Nailed picks (high minutes certainty)
        nailed_premiums = premium_df.nlargest(3, 'minutes_score')
        
        return {
            'top_overall': [self._format_player_rec(p) for p in top_premiums],
            'midfield_options': [self._format_player_rec(p) for p in mid_premiums.to_dict('records')],
            'forward_options': [self._format_player_rec(p) for p in fwd_premiums.to_dict('records')],
            'value_picks': [self._format_player_rec(p) for p in value_premiums.to_dict('records')],
            'differentials': [self._format_player_rec(p) for p in differential_premiums.to_dict('records')],
            'nailed_options': [self._format_player_rec(p) for p in nailed_premiums.to_dict('records')]
        }
    
    def _format_player_rec(self, player: Dict) -> Dict:
        """Format player recommendation"""
        return {
            'id': player['id'],
            'name': player['web_name'],
            'team': player.get('team_name', ''),
            'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[player['element_type']],
            'cost': player['now_cost'],
            'predicted_points': round(player['predicted_points'], 1),
            'premium_score': round(player['premium_score'], 3),
            'points_per_million': round(player['points_per_million'], 2),
            'ownership': player['selected_by_percent'],
            'minutes_score': round(player['minutes_score'], 2),
            'set_piece_score': round(player['set_piece_score'], 2),
            'is_nailed': player.get('is_nailed', False),
            'form': player['form_numeric']
        }

# src/strategy/captain_picker.py - Captaincy optimization
"""
Captain Picker - Optimizes captain and vice-captain selection
Evaluates captaincy options based on expected points, fixture difficulty, and reliability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class CaptainPicker:
    """Optimizes captain and vice-captain selection"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Captaincy evaluation weights
        self.evaluation_weights = {
            'predicted_points': 0.40,
            'fixture_difficulty': 0.25,
            'form': 0.20,
            'reliability': 0.15
        }
        
        # Fixture difficulty multipliers
        self.fixture_multipliers = {
            1: 1.3,  # Very easy
            2: 1.1,  # Easy
            3: 1.0,  # Neutral
            4: 0.9,  # Hard
            5: 0.7   # Very hard
        }
    
    def select_captain(self, starting_xi: List[Dict], 
                      predictions_df: pd.DataFrame) -> Dict:
        """Select optimal captain and vice-captain"""
        self.logger.info("Selecting captain and vice-captain")
        
        if not starting_xi:
            return {'captain': None, 'vice_captain': None, 'alternatives': []}
        
        # Evaluate captaincy options
        captaincy_options = self._evaluate_captaincy_options(starting_xi, predictions_df)
        
        if captaincy_options.empty:
            return self._fallback_captaincy_selection(starting_xi)
        
        # Select captain and vice-captain
        captain = captaincy_options.iloc[0]
        vice_captain = captaincy_options.iloc[1] if len(captaincy_options) > 1 else captain
        
        # Additional alternatives
        alternatives = captaincy_options.iloc[2:5].to_dict('records') if len(captaincy_options) > 2 else []
        
        return {
            'captain': self._format_captaincy_rec(captain),
            'vice_captain': self._format_captaincy_rec(vice_captain),
            'alternatives': [self._format_captaincy_rec(alt) for alt in alternatives],
            'evaluation_summary': self._get_evaluation_summary(captaincy_options)
        }
    
    def _evaluate_captaincy_options(self, starting_xi: List[Dict], 
                                  predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all captaincy options in starting XI"""
        
        captaincy_data = []
        
        for player in starting_xi:
            player_id = player['id']
            
            # Get prediction data
            pred_data = predictions_df[predictions_df['id'] == player_id]
            predicted_points = pred_data.iloc[0]['predicted_points'] if not pred_data.empty else player.get('predicted_points', 0)
            
            # Evaluate captaincy potential
            evaluation = self._evaluate_single_captain_option(player, predicted_points)
            captaincy_data.append(evaluation)
        
        captaincy_df = pd.DataFrame(captaincy_data)
        
        # Calculate overall captaincy score
        captaincy_df['captaincy_score'] = (
            captaincy_df['points_score'] * self.evaluation_weights['predicted_points'] +
            captaincy_df['fixture_score'] * self.evaluation_weights['fixture_difficulty'] +
            captaincy_df['form_score'] * self.evaluation_weights['form'] +
            captaincy_df['reliability_score'] * self.evaluation_weights['reliability']
        )
        
        # Sort by captaincy score
        return captaincy_df.sort_values('captaincy_score', ascending=False)
    
    def _evaluate_single_captain_option(self, player: Dict, predicted_points: float) -> Dict:
        """Evaluate a single player as captain option"""
        
        # Base evaluation data
        evaluation = {
            'id': player['id'],
            'name': player['web_name'],
            'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[player['element_type']],
            'predicted_points': predicted_points,
            'cost': player['now_cost']
        }
        
        # 1. Points Score (normalized)
        evaluation['points_score'] = min(predicted_points / 10, 1.0)  # Normalize to 0-1
        
        # 2. Fixture Score
        fixture_difficulty = player.get('fixture_difficulty_next_4', 3.0)
        evaluation['fixture_score'] = self._calculate_fixture_score(fixture_difficulty)
        
        # 3. Form Score
        form = player.get('form_numeric', 0)
        evaluation['form_score'] = min(form / 8, 1.0)  # Normalize to 0-1
        
        # 4. Reliability Score
        evaluation['reliability_score'] = self._calculate_reliability_score(player)
        
        # Additional metadata
        evaluation['expected_captain_points'] = predicted_points * 2  # Captain gets double points
        evaluation['fixture_difficulty'] = fixture_difficulty
        evaluation['form'] = form
        evaluation['is_premium'] = player.get('is_premium', False)
        evaluation['ownership'] = player.get('selected_by_percent', 0)
        
        return evaluation
    
    def _calculate_fixture_score(self, fixture_difficulty: float) -> float:
        """Calculate fixture-based captaincy score"""
        # Lower difficulty = better captaincy option
        if fixture_difficulty <= 2:
            return 1.0  # Excellent fixtures
        elif fixture_difficulty <= 2.5:
            return 0.8  # Good fixtures
        elif fixture_difficulty <= 3.5:
            return 0.6  # Neutral fixtures
        elif fixture_difficulty <= 4:
            return 0.4  # Difficult fixtures
        else:
            return 0.2  # Very difficult fixtures
    
    def _calculate_reliability_score(self, player: Dict) -> float:
        """Calculate player reliability for captaincy"""
        score = 0.5  # Base score
        
        # Minutes reliability
        minutes_per_game = player.get('minutes_per_game', 0)
        if minutes_per_game >= 80:
            score += 0.3
        elif minutes_per_game >= 70:
            score += 0.2
        elif minutes_per_game >= 60:
            score += 0.1
        
        # Premium player bonus (usually more reliable)
        if player.get('is_premium', False):
            score += 0.1
        
        # Set piece taker bonus
        if player.get('penalties_taken', 0) > 0:
            score += 0.15
        if player.get('direct_freekicks_taken', 0) > 0:
            score += 0.05
        
        return min(score, 1.0)
    
    def _format_captaincy_rec(self, player_data) -> Dict:
        """Format captaincy recommendation"""
        if isinstance(player_data, pd.Series):
            player_data = player_data.to_dict()
        
        return {
            'id': player_data['id'],
            'name': player_data['name'],
            'position': player_data['position'],
            'predicted_points': round(player_data['predicted_points'], 1),
            'expected_captain_points': round(player_data['expected_captain_points'], 1),
            'captaincy_score': round(player_data.get('captaincy_score', 0), 3),
            'fixture_difficulty': player_data.get('fixture_difficulty', 3),
            'form': player_data.get('form', 0),
            'reliability': round(player_data.get('reliability_score', 0), 2),
            'is_premium': player_data.get('is_premium', False),
            'reasoning': self._generate_captaincy_reasoning(player_data)
        }
    
    def _generate_captaincy_reasoning(self, player_data: Dict) -> str:
        """Generate reasoning for captaincy selection"""
        reasons = []
        
        # Points potential
        if player_data.get('predicted_points', 0) > 8:
            reasons.append("High points potential")
        
        # Fixture quality
        fixture_diff = player_data.get('fixture_difficulty', 3)
        if fixture_diff <= 2.5:
            reasons.append("Favorable fixtures")
        elif fixture_diff >= 4:
            reasons.append("Difficult fixtures")
        
        # Form
        form = player_data.get('form', 0)
        if form > 6:
            reasons.append("Excellent form")
        elif form > 4:
            reasons.append("Good form")
        
        # Premium status
        if player_data.get('is_premium', False):
            reasons.append("Premium player")
        
        # Set pieces
        if player_data.get('penalties_taken', 0) > 0:
            reasons.append("Penalty taker")
        
        return "; ".join(reasons) if reasons else "Standard captaincy option"
    
    def _get_evaluation_summary(self, captaincy_df: pd.DataFrame) -> Dict:
        """Get summary of captaincy evaluation"""
        if captaincy_df.empty:
            return {}
        
        top_3 = captaincy_df.head(3)
        
        return {
            'top_captain_score': round(top_3.iloc[0]['captaincy_score'], 3),
            'score_gap': round(top_3.iloc[0]['captaincy_score'] - top_3.iloc[1]['captaincy_score'], 3) if len(top_3) > 1 else 0,
            'average_predicted_points': round(top_3['predicted_points'].mean(), 1),
            'fixture_quality': 'Good' if top_3['fixture_score'].mean() > 0.7 else 'Average' if top_3['fixture_score'].mean() > 0.5 else 'Poor',
            'premium_captain_available': any(top_3['is_premium']),
            'evaluation_weights': self.evaluation_weights
        }
    
    def _fallback_captaincy_selection(self, starting_xi: List[Dict]) -> Dict:
        """Fallback captaincy selection when evaluation fails"""
        if not starting_xi:
            return {'captain': None, 'vice_captain': None, 'alternatives': []}
        
        # Sort by predicted points
        sorted_players = sorted(starting_xi, 
                              key=lambda x: x.get('predicted_points', 0), 
                              reverse=True)
        
        captain = sorted_players[0]
        vice_captain = sorted_players[1] if len(sorted_players) > 1 else captain
        
        return {
            'captain': {
                'id': captain['id'],
                'name': captain['web_name'],
                'predicted_points': captain.get('predicted_points', 0),
                'reasoning': 'Highest predicted points (fallback)'
            },
            'vice_captain': {
                'id': vice_captain['id'],
                'name': vice_captain['web_name'],
                'predicted_points': vice_captain.get('predicted_points', 0),
                'reasoning': 'Second highest predicted points (fallback)'
            },
            'alternatives': []
        }

# src/strategy/risk_manager.py - Risk assessment and differential analysis
"""
Risk Manager - Assesses team risk and identifies differential opportunities
Analyzes ownership, price volatility, and rotation risks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class RiskManager:
    """Manages risk assessment and differential analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.ownership_thresholds = {
            'template': 50,      # >50% = template player
            'popular': 25,       # 25-50% = popular
            'moderate': 10,      # 10-25% = moderate
            'differential': 5    # <10% = differential
        }
        
        self.rotation_risk_factors = {
            'minutes_consistency': 0.3,
            'squad_depth': 0.2,
            'recent_starts': 0.3,
            'manager_rotation': 0.2
        }
    
    def assess_team_risk(self, team_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Comprehensive team risk assessment"""
        self.logger.info("Performing team risk assessment")
        
        if 'squad' not in team_data:
            return {'error': 'No squad data provided'}
        
        squad = team_data['squad']
        starting_xi = team_data.get('starting_xi', [])
        
        # Convert to DataFrame for analysis
        squad_df = pd.DataFrame(squad)
        
        # Perform risk analyses
        ownership_analysis = self._analyze_ownership_risk(squad_df)
        rotation_analysis = self._analyze_rotation_risk(squad_df)
        price_analysis = self._analyze_price_risk(squad_df)
        fixture_analysis = self._analyze_fixture_risk(squad_df)
        differential_analysis = self._identify_differentials(squad_df, players_df)
        
        # Overall risk assessment
        overall_risk = self._calculate_overall_risk(
            ownership_analysis, rotation_analysis, price_analysis, fixture_analysis
        )
        
        return {
            'overall_risk': overall_risk,
            'ownership_analysis': ownership_analysis,
            'rotation_analysis': rotation_analysis,
            'price_analysis': price_analysis,
            'fixture_analysis': fixture_analysis,
            'differential_analysis': differential_analysis,
            'risk_recommendations': self._generate_risk_recommendations(
                overall_risk, ownership_analysis, rotation_analysis
            )
        }
    
    def _analyze_ownership_risk(self, squad_df: pd.DataFrame) -> Dict:
        """Analyze ownership patterns and template risk"""
        
        ownership_categories = {
            'template': [],
            'popular': [],
            'moderate': [],
            'differential': []
        }
        
        for _, player in squad_df.iterrows():
            ownership = player.get('selected_by_percent', 0)
            
            if ownership > self.ownership_thresholds['template']:
                ownership_categories['template'].append(player['web_name'])
            elif ownership > self.ownership_thresholds['popular']:
                ownership_categories['popular'].append(player['web_name'])
            elif ownership > self.ownership_thresholds['moderate']:
                ownership_categories['moderate'].append(player['web_name'])
            else:
                ownership_categories['differential'].append(player['web_name'])
        
        # Calculate ownership metrics
        avg_ownership = squad_df['selected_by_percent'].mean()
        template_count = len(ownership_categories['template'])
        differential_count = len(ownership_categories['differential'])
        
        # Risk assessment
        if template_count > 8:
            ownership_risk = 'High'
            risk_reason = 'Too many template players'
        elif template_count > 6:
            ownership_risk = 'Medium'
            risk_reason = 'Moderate template exposure'
        else:
            ownership_risk = 'Low'
            risk_reason = 'Good ownership balance'
        
        return {
            'ownership_risk': ownership_risk,
            'risk_reason': risk_reason,
            'average_ownership': round(avg_ownership, 1),
            'template_players': ownership_categories['template'],
            'differential_players': ownership_categories['differential'],
            'ownership_distribution': {k: len(v) for k, v in ownership_categories.items()},
            'template_count': template_count,
            'differential_count': differential_count
        }
    
    def _analyze_rotation_risk(self, squad_df: pd.DataFrame) -> Dict:
        """Analyze rotation and minutes risk"""
        
        high_risk_players = []
        medium_risk_players = []
        low_risk_players = []
        
        for _, player in squad_df.iterrows():
            minutes_per_game = player.get('minutes_per_game', 0)
            is_nailed = player.get('is_nailed', False)
            
            # Determine rotation risk
            if minutes_per_game < 45 or not is_nailed:
                risk_level = 'High'
                high_risk_players.append({
                    'name': player['web_name'],
                    'minutes_per_game': minutes_per_game,
                    'is_nailed': is_nailed
                })
            elif minutes_per_game < 70:
                risk_level = 'Medium'
                medium_risk_players.append({
                    'name': player['web_name'],
                    'minutes_per_game': minutes_per_game,
                    'is_nailed': is_nailed
                })
            else:
                risk_level = 'Low'
                low_risk_players.append({
                    'name': player['web_name'],
                    'minutes_per_game': minutes_per_game,
                    'is_nailed': is_nailed
                })
        
        # Overall rotation risk
        high_risk_count = len(high_risk_players)
        if high_risk_count > 4:
            rotation_risk = 'High'
        elif high_risk_count > 2:
            rotation_risk = 'Medium'
        else:
            rotation_risk = 'Low'
        
        return {
            'rotation_risk': rotation_risk,
            'high_risk_players': high_risk_players,
            'medium_risk_players': medium_risk_players,
            'low_risk_players': low_risk_players,
            'high_risk_count': high_risk_count,
            'average_minutes': round(squad_df['minutes_per_game'].mean(), 1)
        }
    
    def _analyze_price_risk(self, squad_df: pd.DataFrame) -> Dict:
        """Analyze price change and budget risk"""
        
        # Price change analysis
        rising_players = []
        falling_players = []
        
        for _, player in squad_df.iterrows():
            price_change = player.get('cost_change_event', 0)
            
            if price_change > 0:
                rising_players.append({
                    'name': player['web_name'],
                    'price_change': price_change,
                    'current_price': player['now_cost']
                })
            elif price_change < 0:
                falling_players.append({
                    'name': player['web_name'],
                    'price_change': price_change,
                    'current_price': player['now_cost']
                })
        
        # Budget efficiency
        total_value = squad_df['now_cost'].sum()
        avg_cost = total_value / len(squad_df)
        
        # Risk assessment
        if len(falling_players) > 3:
            price_risk = 'High'
        elif len(falling_players) > 1:
            price_risk = 'Medium'
        else:
            price_risk = 'Low'
        
        return {
            'price_risk': price_risk,
            'rising_players': rising_players,
            'falling_players': falling_players,
            'total_squad_value': round(total_value, 1),
            'average_player_cost': round(avg_cost, 1),
            'budget_utilization': round((total_value / 100) * 100, 1)  # As percentage of £100m
        }
    
    def _analyze_fixture_risk(self, squad_df: pd.DataFrame) -> Dict:
        """Analyze fixture difficulty risk"""
        
        difficult_fixtures = []
        easy_fixtures = []
        
        for _, player in squad_df.iterrows():
            fixture_difficulty = player.get('fixture_difficulty_next_4', 3.0)
            
            if fixture_difficulty >= 4:
                difficult_fixtures.append({
                    'name': player['web_name'],
                    'team': player.get('team_name', ''),
                    'fixture_difficulty': fixture_difficulty
                })
            elif fixture_difficulty <= 2:
                easy_fixtures.append({
                    'name': player['web_name'],
                    'team': player.get('team_name', ''),
                    'fixture_difficulty': fixture_difficulty
                })
        
        avg_fixture_difficulty = squad_df['fixture_difficulty_next_4'].mean()
        
        # Risk assessment
        if avg_fixture_difficulty >= 3.5:
            fixture_risk = 'High'
        elif avg_fixture_difficulty >= 3.2:
            fixture_risk = 'Medium'
        else:
            fixture_risk = 'Low'
        
        return {
            'fixture_risk': fixture_risk,
            'average_fixture_difficulty': round(avg_fixture_difficulty, 2),
            'difficult_fixtures': difficult_fixtures,
            'easy_fixtures': easy_fixtures,
            'fixture_balance': 'Good' if 2.5 <= avg_fixture_difficulty <= 3.2 else 'Unbalanced'
        }
    
    def _identify_differentials(self, squad_df: pd.DataFrame, 
                              all_players_df: pd.DataFrame) -> Dict:
        """Identify differential opportunities"""
        
        # Current differentials in squad
        squad_differentials = squad_df[
            squad_df['selected_by_percent'] < self.ownership_thresholds['differential']
        ]
        
        # Potential differentials not in squad
        potential_diffs = all_players_df[
            (all_players_df['selected_by_percent'] < self.ownership_thresholds['differential']) &
            (all_players_df['predicted_points'] > 20) &  # Minimum points threshold
            (~all_players_df['id'].isin(squad_df['id']))
        ].sort_values('predicted_points', ascending=False).head(10)
        
        return {
            'current_differentials': [
                {
                    'name': player['web_name'],
                    'ownership': player['selected_by_percent'],
                    'predicted_points': player.get('predicted_points', 0),
                    'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[player['element_type']]
                }
                for _, player in squad_differentials.iterrows()
            ],
            'potential_differentials': [
                {
                    'name': player['web_name'],
                    'ownership': player['selected_by_percent'],
                    'predicted_points': player.get('predicted_points', 0),
                    'cost': player['now_cost'],
                    'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[player['element_type']]
                }
                for _, player in potential_diffs.iterrows()
            ],
            'differential_count': len(squad_differentials),
            'differential_strategy': 'Aggressive' if len(squad_differentials) > 4 else 'Conservative' if len(squad_differentials) < 2 else 'Balanced'
        }
    
    def _calculate_overall_risk(self, ownership_analysis: Dict, 
                              rotation_analysis: Dict, price_analysis: Dict,
                              fixture_analysis: Dict) -> Dict:
        """Calculate overall team risk score"""
        
        # Risk scoring (0-3 scale)
        risk_scores = {
            'ownership': {'Low': 0, 'Medium': 1, 'High': 2}.get(ownership_analysis['ownership_risk'], 1),
            'rotation': {'Low': 0, 'Medium': 1, 'High': 3}.get(rotation_analysis['rotation_risk'], 1),
            'price': {'Low': 0, 'Medium': 1, 'High': 2}.get(price_analysis['price_risk'], 1),
            'fixture': {'Low': 0, 'Medium': 1, 'High': 1}.get(fixture_analysis['fixture_risk'], 1)
        }
        
        # Weighted overall score
        weights = {'ownership': 0.2, 'rotation': 0.4, 'price': 0.2, 'fixture': 0.2}
        overall_score = sum(risk_scores[key] * weights[key] for key in risk_scores)
        
        # Risk level
        if overall_score < 0.5:
            risk_level = 'Low'
        elif overall_score < 1.5:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'risk_level': risk_level,
            'risk_score': round(overall_score, 2),
            'risk_components': risk_scores,
            'primary_concerns': self._identify_primary_concerns(risk_scores)
        }
    
    def _identify_primary_concerns(self, risk_scores: Dict) -> List[str]:
        """Identify primary risk concerns"""
        concerns = []
        
        if risk_scores['rotation'] >= 2:
            concerns.append('High rotation risk - consider more nailed players')
        if risk_scores['ownership'] >= 2:
            concerns.append('Template-heavy squad - limited differential upside')
        if risk_scores['price'] >= 2:
            concerns.append('Price volatility risk - players likely to drop')
        if risk_scores['fixture'] >= 1:
            concerns.append('Fixture difficulty - tough upcoming matches')
        
        return concerns
    
    def _generate_risk_recommendations(self, overall_risk: Dict,
                                     ownership_analysis: Dict,
                                     rotation_analysis: Dict) -> List[str]:
        """Generate actionable risk recommendations"""
        recommendations = []
        
        # Rotation risk recommendations
        if rotation_analysis['high_risk_count'] > 2:
            recommendations.append("Consider transferring out high rotation risk players")
            recommendations.append("Prioritize nailed players in upcoming transfers")
        
        # Ownership recommendations
        if ownership_analysis['differential_count'] < 2:
            recommendations.append("Consider adding 1-2 differential players for upside")
        elif ownership_analysis['template_count'] > 8:
            recommendations.append("Squad is too template-heavy - consider unique picks")
        
        # Overall recommendations
        if overall_risk['risk_level'] == 'High':
            recommendations.append("High-risk squad - consider conservative approach")
        elif overall_risk['risk_level'] == 'Low':
            recommendations.append("Low-risk squad - opportunity for more aggressive plays")
        
        return recommendations[:5]  # Limit to top 5 recommendations
