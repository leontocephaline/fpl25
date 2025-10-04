from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PlayerInputs:
    player_id: int
    name: str
    position: str  # GK, DEF, MID, FWD
    team: str
    # Betting probabilities in [0,1]
    prob_goal: Optional[float] = None
    prob_assist: Optional[float] = None
    prob_clean_sheet: Optional[float] = None
    prob_card: Optional[float] = None
    # Historical metrics (per 90)
    xg_per90: Optional[float] = None
    xa_per90: Optional[float] = None
    xpts_per90: Optional[float] = None
    minutes_per_game: Optional[float] = None
    # Fixture & form
    fixture_difficulty: Optional[float] = None  # 0.7..1.3 scaling per config
    form_trend: Optional[float] = None  # normalized -1..1
    # News and status
    start_probability: Optional[int] = None  # 0..100
    injury_status: Optional[str] = None  # 'fit', 'doubtful', 'injured', 'suspended', etc.
    expected_return: Optional[int] = None  # Gameweek number of expected return
    confidence: float = 1.0  # Confidence in the analysis (0.0-1.0)


WEIGHTS = {
    "betting": 0.50,
    "historical": 0.30,
    "minutes": 0.10,
    "fixture": 0.05,
    "form": 0.05,
}


def safe(v: Optional[float], default: float = 0.0) -> float:
    return float(v) if v is not None else default


def convert_confidence(confidence) -> float:
    """Convert confidence string to float value"""
    if isinstance(confidence, (int, float)):
        return float(confidence)
    elif isinstance(confidence, str):
        confidence_lower = confidence.lower()
        if confidence_lower in ['low', 'l']:
            return 0.3
        elif confidence_lower in ['medium', 'med', 'm']:
            return 0.6
        elif confidence_lower in ['high', 'h']:
            return 0.9
        else:
            # Try to parse as float string
            try:
                return float(confidence)
            except ValueError:
                return 0.5  # Default fallback
    else:
        return 0.5  # Default fallback


def score_player(p: PlayerInputs) -> Dict[str, float]:
    """
    Returns a dict with component scores and total.
    All components normalized to a 0-10 scale.
    Implements fallback: if all betting inputs are missing, re-distribute
    the betting weight proportionally across other components.
    """
    # Apply injury status penalty
    injury_penalty = 1.0
    if p.injury_status == 'injured':
        injury_penalty = 0.2  # 80% penalty for injured players
    elif p.injury_status == 'doubtful':
        injury_penalty = 0.5  # 50% penalty for doubtful players
    elif p.injury_status == 'suspended':
        injury_penalty = 0.1  # 90% penalty for suspended players
    
    # Apply confidence multiplier
    confidence_val = convert_confidence(p.confidence)
    confidence_multiplier = 0.5 + (confidence_val * 0.5)  # Maps 0..1 to 0.5..1.0
    
    # Adjust start probability based on injury status and confidence
    base_start_prob = safe(p.start_probability, 100.0)
    if p.injury_status in ['injured', 'doubtful', 'suspended']:
        base_start_prob *= injury_penalty
    
    # Start probability: multiply total by start probability as availability
    start_multiplier = (base_start_prob / 100.0) * confidence_multiplier

    # Betting component: rough mapping to xPts contribution (scaled to 0-10)
    bet_goal = safe(p.prob_goal, 0) * 2.0  # Max 2.0 points per goal
    bet_assist = safe(p.prob_assist, 0) * 1.5  # Max 1.5 points per assist
    bet_cs = safe(p.prob_clean_sheet, 0) * (1.5 if p.position in ("GK", "DEF") else 0.5)  # Higher for defenders/GKs
    bet_card = -safe(p.prob_card, 0) * 0.5  # Penalty for likely bookings
    betting_component = (bet_goal + bet_assist + bet_cs + bet_card) * confidence_multiplier

    # Historical component: xG + xA + xPts per 90 as a baseline (scaled to 0-10)
    historical_component = (
        (safe(p.xg_per90, 0) * 3.0) +  # Max 3.0
        (safe(p.xa_per90, 0) * 2.0) +  # Max 2.0
        (safe(p.xpts_per90, 0) * 2.0)   # Max 2.0
    ) * confidence_multiplier

    # Minutes consistency: scaled 0-2.0 based on minutes played (capped at 90)
    minutes_component = min(safe(p.minutes_per_game, 0) / 90.0, 1.0) * 2.0 * confidence_multiplier

    # Fixture difficulty: -1.0 to 1.0 range (hard to easy)
    fixture_component = (safe(p.fixture_difficulty, 1.0) - 1.0) * 2.0  # Scale to -1.0 to 1.0

    # Form trend: -2.0 to 2.0 range (poor to excellent)
    form_component = safe(p.form_trend, 0) * 2.0 * confidence_multiplier

    # Weight fallback if betting unavailable
    betting_available = any(
        v is not None for v in (p.prob_goal, p.prob_assist, p.prob_clean_sheet, p.prob_card)
    )
    if betting_available:
        weights = WEIGHTS
    else:
        # redistribute betting weight proportionally to the others
        remaining = {k: v for k, v in WEIGHTS.items() if k != "betting"}
        total_rem = sum(remaining.values())
        weights = {k: (v / total_rem) for k, v in remaining.items()}

    # Baseline appearance component (pre-multiplier): ensures a non-zero base
    baseline_component = 0.5 * confidence_multiplier

    # Calculate component scores with weights
    components = {
        "betting": betting_component * weights["betting"] if betting_available else 0.0,
        "historical": historical_component * weights["historical"],
        "minutes": minutes_component * weights["minutes"],
        "fixture": fixture_component * weights["fixture"],
        "form": form_component * weights["form"]
    }
    
    # Calculate total score
    if betting_available:
        total = sum(components.values())
    else:
        # Redistribute betting weight to other components
        betting_weight = weights.get("betting", 0.0)
        total_weight = sum(weights.values()) - betting_weight
        if total_weight > 0:
            scale = 1.0 + (betting_weight / total_weight) if betting_weight > 0 else 1.0
            total = sum(v * scale for k, v in components.items() if k != "betting")
        else:
            total = sum(components.values())
    
    # Apply start probability multiplier (availability)
    total *= start_multiplier
    
    # Scale to 0-10 range and apply bounds
    total = max(0.0, min(10.0, total))
    
    # Prepare detailed breakdown for debugging/analysis
    breakdown = {
        "total": round(total, 2),
        "components": {
            "betting": round(betting_component, 2) if betting_available else None,
            "historical": round(historical_component, 2),
            "minutes": round(minutes_component, 2),
            "fixture": round(fixture_component, 2),
            "form": round(form_component, 2)
        },
        "modifiers": {
            "start_multiplier": round(start_multiplier, 2),
            "injury_penalty": round(injury_penalty, 2),
            "confidence": round(confidence_val, 2)
        }
    }
    
    # For backward compatibility, include flat structure as well
    result = {"total": round(total, 2)}
    result.update(breakdown)
    
    return result
