#!/usr/bin/env python3

import requests

# Get FPL data
data = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()

# Find Arsenal team ID
arsenal_team_id = None
for team in data['teams']:
    if team['name'] == 'Arsenal':
        arsenal_team_id = team['id']
        break

print("=== ARSENAL DEFENDERS ===")
arsenal_defenders = [p for p in data['elements'] if p['element_type'] == 2 and p['team'] == arsenal_team_id]
for player in arsenal_defenders:
    availability = player.get('chance_of_playing_next_round', 100) or 100
    print(f"ID {player['id']}: {player['web_name']} ({player['first_name']} {player['second_name']}) - Available: {availability}%")

print("\n=== ALL GABRIEL PLAYERS ===")
gabriel_players = [p for p in data['elements'] if 'gabriel' in p['web_name'].lower() or 'gabriel' in p['first_name'].lower() or 'gabriel' in p['second_name'].lower()]
for player in gabriel_players:
    team_name = data['teams'][player['team'] - 1]['name']
    position = ['GKP', 'DEF', 'MID', 'FWD'][player['element_type'] - 1]
    availability = player.get('chance_of_playing_next_round', 100) or 100
    print(f"ID {player['id']}: {player['web_name']} ({player['first_name']} {player['second_name']}) - {team_name} {position} - Available: {availability}%")
