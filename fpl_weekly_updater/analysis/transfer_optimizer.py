from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TransferRecommendation:
    out_player_id: Optional[int]
    in_player_id: Optional[int]
    expected_points_delta: float
    rationale: str


def recommend_transfers(
    squad_player_ids: List[int],
    scored_players: Dict[int, Dict[str, float]],
    budget: float,
    free_transfers: int = 1,
    starters: Optional[List[int]] = None,
    hit_cost: int = 4,
    prices: Optional[Dict[int, float]] = None,
    bank: Optional[float] = None,
    # Market-aware inputs
    market_scores: Optional[Dict[int, float]] = None,
    pos_map: Optional[Dict[int, str]] = None,
    names: Optional[Dict[int, str]] = None,
    availabilities: Optional[Dict[int, int]] = None,
    max_suggestions: int = 3,
) -> List[Dict]:
    """
    Simple heuristic for initial wiring:
    - Consider only bench->starter upgrades within current squad (no market additions yet).
    - Compute gain = total(IN) - total(OUT).
    - Apply hit only if number of transfers exceeds free_transfers.
    - Return up to 2 best upgrades with positive net.

    Returns list of dicts: {out, in, gain, hit, net, why}
    """
    starters = starters or []
    starters_set = set(starters)
    if not squad_player_ids:
        return []

    # Split starters vs bench
    bench = [pid for pid in squad_player_ids if pid not in starters_set]
    starters_list = [pid for pid in squad_player_ids if pid in starters_set]

    # Score lookup helper
    def total(pid: int) -> float:
        return float((scored_players.get(pid) or {}).get("total", 0.0))

    # If we have market_scores, search market for upgrades within budget
    proposals: List[Dict] = []
    if market_scores and pos_map and prices is not None:
        # sort starters prioritizing low availability, then low score
        def starter_sort_key(pid: int):
            sp = (availabilities or {}).get(pid)
            if sp is None:
                sp = 100
            return (sp, total(pid))
        starters_sorted = sorted(starters_list, key=starter_sort_key)
        owned = set(squad_player_ids)
        current_bank = float(bank or 0.0)
        pos_used: Dict[str, int] = {}
        for out_pid in starters_sorted:
            out_pos = pos_map.get(out_pid, "") if pos_map else ""
            # enforce one suggestion per position unless this player is low availability (<50)
            sp = (availabilities or {}).get(out_pid)
            if sp is None:
                sp = 100
            if pos_used.get(out_pos, 0) >= 1 and sp >= 50:
                continue
            out_price = float((prices or {}).get(out_pid, 0.0))
            out_score = float(market_scores.get(out_pid, total(out_pid)))
            best_candidate = None
            best_gain = 0.0
            # iterate all market players matching position
            for cand_pid, cand_score in market_scores.items():
                if cand_pid == out_pid:
                    continue
                if cand_pid in owned:
                    # allow swapping in someone you already own only if it's bench -> starter,
                    # but here we target market upgrades, so skip owned.
                    continue
                if pos_map.get(cand_pid) != out_pos:
                    continue
                
                # Exclude players with very low start probability (injured/unavailable)
                cand_availability = (availabilities or {}).get(cand_pid, 100)
                if cand_availability <= 10:  # Skip players with <=10% start probability
                    continue
                cand_price = float((prices or {}).get(cand_pid, None) or 0.0)
                # affordability: price(IN) <= price(OUT) + bank
                if cand_price > out_price + current_bank:
                    continue
                gain = cand_score - out_score
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = (cand_pid, cand_price, cand_score)
            if best_candidate and best_gain > 0:
                in_pid, in_price, in_score = best_candidate
                proposals.append({
                    "out": out_pid,
                    "in": in_pid,
                    "gain": round(best_gain, 2),
                    "out_name": (names or {}).get(out_pid),
                    "in_name": (names or {}).get(in_pid),
                    "out_price": round(out_price, 1),
                    "in_price": round(in_price, 1),
                })
                pos_used[out_pos] = pos_used.get(out_pos, 0) + 1
                if len(proposals) >= max_suggestions:
                    break
    else:
        # Fallback: bench->starter swaps within squad only
        bench_sorted = sorted(bench, key=total, reverse=True)
        starters_sorted = sorted(starters_list, key=total)
        for out_pid, in_pid in zip(starters_sorted, bench_sorted):
            if total(in_pid) <= total(out_pid):
                continue
            gain = total(in_pid) - total(out_pid)
            proposals.append({
                "out": out_pid,
                "in": in_pid,
                "gain": round(gain, 2),
                "out_name": (names or {}).get(out_pid),
                "in_name": (names or {}).get(in_pid),
            })

    # Apply hits beyond free transfers and compute net
    results: List[Dict] = []
    for idx, p in enumerate(proposals[:max_suggestions], start=1):
        hit = 0
        if idx > free_transfers:
            hit = hit_cost
        net = round(p["gain"] - hit, 2)
        if net <= 0:
            continue
        p.update({
            "hit": hit,
            "net": net,
            "why": f"gain={p['gain']:.2f}, hit={hit}, net={net:.2f}",
        })
        results.append(p)

    return results
