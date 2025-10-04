from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


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
    market_scores: Optional[Dict[int, float]] = None,
    pos_map: Optional[Dict[int, str]] = None,
    names: Optional[Dict[int, str]] = None,
    availabilities: Optional[Dict[int, int]] = None,
    max_suggestions: int = 3,
    protected_player_ids: Optional[Iterable[int]] = None,
    suggestions_per_position: int = 2,
    include_bench_out_candidates: bool = True,
    bench_availability_threshold: int = 50,
    require_positive_net: bool = True,
) -> List[Dict]:
    """
    Simple heuristic for initial wiring:
    - Consider only bench->starter upgrades within current squad (no market additions yet).
    - Compute gain = total(IN) - total(OUT).
    - Apply hit only if number of transfers exceeds free_transfers.
    - Return up to ``max_suggestions`` best upgrades with positive net.

    Args:
        squad_player_ids: All player IDs currently owned.
        scored_players: Mapping from player ID to score dict containing ``total``.
        budget: Squad budget in millions (currently unused, placeholder for future logic).
        free_transfers: Number of free transfers available.
        starters: Optional list of starting XI IDs.
        hit_cost: Points deducted per extra transfer beyond free transfers.
        prices: Mapping of player ID to price in millions.
        bank: Available bank in millions.
        market_scores: Optional market-wide scores for evaluating replacements.
        pos_map: Mapping of player ID to position code (GK/DEF/MID/FWD).
        names: Mapping of player ID to player display name.
        availabilities: Mapping of player ID to start probability (0-100).
        max_suggestions: Maximum number of recommendations to return.
        protected_player_ids: Iterable of player IDs that should never be suggested as ``out``.

    Returns:
        A list of recommendation dicts sorted by descending net gain.
    """
    starters = starters or []
    starters_set = set(starters)
    protected_set: Set[int] = set(protected_player_ids or [])
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
        # Always choose OUT candidates from bench only (user rule)
        def bench_sort_key(pid: int):
            sp = (availabilities or {}).get(pid)
            if sp is None:
                sp = 100
            return (sp, total(pid))
        outs = sorted(bench, key=bench_sort_key)
        owned = set(squad_player_ids)
        current_bank = float(bank or 0.0)
        pos_used: Dict[str, int] = {}
        for out_pid in outs:
            if out_pid in protected_set:
                continue
            out_pos = pos_map.get(out_pid, "") if pos_map else ""
            # enforce one suggestion per position unless this player is low availability (<50)
            sp = (availabilities or {}).get(out_pid)
            if sp is None:
                sp = 100
            if pos_used.get(out_pos, 0) >= suggestions_per_position and sp >= 50:
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
            if best_candidate and (best_gain > 0 or not require_positive_net):
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
    else:
        # Fallback: bench->starter swaps within squad only
        bench_sorted = sorted(bench, key=total, reverse=True)
        starters_sorted = sorted(starters_list, key=total)
        for out_pid, in_pid in zip(starters_sorted, bench_sorted):
            if out_pid in protected_set:
                continue
            if total(in_pid) <= total(out_pid) and require_positive_net:
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
    proposals.sort(key=lambda item: item.get("gain", 0.0), reverse=True)

    enriched: List[Dict] = []
    for idx, p in enumerate(proposals, start=1):
        hit = 0
        if idx > free_transfers:
            hit = hit_cost
        net = round(p.get("gain", 0.0) - hit, 2)
        q = dict(p)
        q.update({
            "hit": hit,
            "net": net,
            "why": f"gain={p.get('gain', 0.0):.2f}, hit={hit}, net={net:.2f}",
        })
        enriched.append(q)

    # Prefer positive net first
    positives = [p for p in enriched if p["net"] > 0]
    results: List[Dict] = []
    for p in positives:
        results.append(p)
        if len(results) >= max_suggestions:
            break

    if len(results) < max_suggestions and not require_positive_net:
        # Fill remaining with next best non-positive net
        nonpos = [p for p in enriched if p["net"] <= 0]
        for p in nonpos:
            p = dict(p)
            p["why"] = p.get("why", "") + " | non-positive net"
            results.append(p)
            if len(results) >= max_suggestions:
                break

    return results
