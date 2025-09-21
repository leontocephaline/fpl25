"""
Captain Picker - Selects captain and vice-captain from the starting XI
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple


class CaptainPicker:
    """Selects captain and vice-captain from the starting XI based on predicted points"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select_captain(self, starting_xi: List[Dict], predictions: pd.DataFrame) -> Dict:
        """Select captain and vice-captain from the starting XI"""
        try:
            if not starting_xi:
                self.logger.warning("No players in starting XI for captain selection")
                return {
                    'captain': None,
                    'vice_captain': None,
                    'selection_method': 'none'
                }
            
            # Sort players by predicted points (descending)
            sorted_players = sorted(
                starting_xi, 
                key=lambda x: x.get('predicted_points', 0), 
                reverse=True
            )
            
            # Select top two players as captain and vice-captain
            captain = sorted_players[0] if len(sorted_players) > 0 else None
            vice_captain = sorted_players[1] if len(sorted_players) > 1 else None
            
            self.logger.info(f"Selected {captain['web_name'] if captain else 'None'} as captain, "
                           f"{vice_captain['web_name'] if vice_captain else 'None'} as vice-captain")
            
            return {
                'captain': captain,
                'vice_captain': vice_captain,
                'selection_method': 'highest_predicted_points'
            }
            
        except Exception as e:
            self.logger.error(f"Error in captain selection: {str(e)}")
            return {
                'captain': None,
                'vice_captain': None,
                'selection_method': 'error'
            }
