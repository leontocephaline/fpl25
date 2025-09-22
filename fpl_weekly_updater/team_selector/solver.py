from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Note: Avoid heavy dependencies; use simple greedy selection with constraints.

Positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def _price_millions(elem: Dict) -> float:
    try:
        now_cost = elem.get('now_cost')  # tenths of a million
        return float(now_cost) / 10.0 if now_cost is not None else 0.0
    except Exception:
        return 0.0


def _ownership(elem: Dict) -> Optional[float]:
    try:
        val = elem.get('selected_by_percent')
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace('%', '').strip())
    except Exception:
        return None


def _name(elem: Dict) -> str:
    return f"{elem.get('first_name', '')} {elem.get('second_name', '')}".strip()


def select_initial_squad(
    *,
    bootstrap: Dict,
    budget_millions: float = 100.0,
    formation: str = "3-5-2",
    lock_names: List[str] | None = None,
    exclude_names: List[str] | None = None,
    max_from_team: int = 3,
) -> Dict:
    """
    Build an initial 15-player squad within budget and per-team limits using a simple greedy heuristic.

    The selector prioritizes lower price and higher expected points proxy (EP next) while respecting:
    - 2 GK, 5 DEF, 5 MID, 3 FWD
    - Max 3 from same PL team
    - Total spend <= budget_millions

    Args:
        bootstrap: Full FPL bootstrap response.
        budget_millions: Budget in millions.
        formation: Starting XI formation for info only (e.g., '3-5-2').
        lock_names: Players to force into the squad by display name (case-insensitive).
        exclude_names: Players to exclude.
        max_from_team: Max players per real team.

    Returns:
        Dict with keys: 'squad' (list of rows), 'total_cost', 'by_team', 'by_position'.
    """
    elements: List[Dict] = bootstrap.get('elements', [])
    teams: List[Dict] = bootstrap.get('teams', [])
    team_short = {t['id']: t.get('short_name') for t in teams}

    lock_set = {s.lower() for s in (lock_names or [])}
    excl_set = {s.lower() for s in (exclude_names or [])}

    # Filter candidates by availability (status 'a' is available); use FPL status as authoritative
    candidates: List[Dict] = []
    for el in elements:
        status = (el.get('status') or 'a').lower()
        if status != 'a':
            continue
        name = _name(el)
        if name.lower() in excl_set:
            continue
        el['_name'] = name
        el['_team_short'] = team_short.get(el.get('team'), 'UNK')
        el['_pos'] = Positions.get(el.get('element_type'), 'UNK')
        el['_price'] = _price_millions(el)
        # EP proxy; fallback to points_per_game or form
        ep_next = el.get('ep_next')
        try:
            el['_ep'] = float(ep_next) if ep_next is not None else float(el.get('points_per_game') or 0)
        except Exception:
            try:
                el['_ep'] = float(el.get('form') or 0)
            except Exception:
                el['_ep'] = 0.0
        el['_own'] = _ownership(el)
        candidates.append(el)

    # Put locks first; then by EP per price (value)
    def value_key(el: Dict) -> Tuple[float, float]:
        price = el['_price'] or 0.0001
        return (el['_ep'] / price if price > 0 else 0.0, el['_ep'])

    locks: List[Dict] = [el for el in candidates if el['_name'].lower() in lock_set]
    rest: List[Dict] = [el for el in candidates if el['_name'].lower() not in lock_set]
    rest.sort(key=value_key, reverse=True)
    locks.sort(key=value_key, reverse=True)

    target_counts = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    picked: List[Dict] = []
    by_team: Dict[int, int] = {}
    by_pos: Dict[str, int] = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    budget_left = float(budget_millions)

    def try_add(el: Dict) -> bool:
        nonlocal budget_left
        pos = el['_pos']
        team_id = el.get('team')
        price = el['_price']
        if by_pos.get(pos, 0) >= target_counts[pos]:
            return False
        if by_team.get(team_id, 0) >= max_from_team:
            return False
        if budget_left - price < -1e-6:
            return False
        picked.append(el)
        by_pos[pos] = by_pos.get(pos, 0) + 1
        by_team[team_id] = by_team.get(team_id, 0) + 1
        budget_left -= price
        return True

    # Take locks first (best value locks first)
    for el in locks:
        try_add(el)

    # Then fill the rest greedily
    for el in rest:
        if len(picked) >= 15:
            break
        try_add(el)

    # If under-filled due to constraints, relax by allowing doubtful status players as last resort (should rarely happen)
    if len(picked) < 15:
        backup_pool = [el for el in elements if (el.get('status') or 'a').lower() in {'a','d'} and _name(el).lower() not in excl_set]
        backup_pool.sort(key=lambda x: (float(x.get('ep_next') or 0), float(x.get('points_per_game') or 0)), reverse=True)
        for el in backup_pool:
            el['_name'] = _name(el)
            el['_team_short'] = team_short.get(el.get('team'), 'UNK')
            el['_pos'] = Positions.get(el.get('element_type'), 'UNK')
            el['_price'] = _price_millions(el)
            if try_add(el):
                if len(picked) >= 15:
                    break

    total_cost = sum(_price_millions(el) for el in picked)
    squad_rows = [{
        'id': el.get('id'),
        'name': el['_name'],
        'position': el['_pos'],
        'team': el['_team_short'],
        'price': round(_price_millions(el), 1),
        'ownership': el.get('selected_by_percent')
    } for el in picked]

    return {
        'squad': squad_rows,
        'total_cost': round(total_cost, 1),
        'budget_left': round(budget_left, 1),
        'by_team': by_team,
        'by_position': by_pos,
        'formation': formation,
    }
