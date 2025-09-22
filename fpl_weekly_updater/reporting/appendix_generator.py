from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from xml.sax.saxutils import escape


def _wrap_by_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    words = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        tentative = (" ".join(cur + [w])).strip()
        if stringWidth(tentative, font_name, font_size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            # hard break long words
            if stringWidth(w, font_name, font_size) > max_width:
                chunk = ""
                for ch in w:
                    if stringWidth((chunk + ch), font_name, font_size) <= max_width:
                        chunk += ch
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = [chunk] if chunk else []
            else:
                cur = [w]
    if cur:
        lines.append(" ".join(cur))
    if not lines:
        lines = [""]
    return lines


def _draw_paragraph(can: canvas.Canvas, text: str, left: float, y: float, width: float, italic: bool = False) -> float:
    """Draw a wrapped paragraph and return the new y position."""
    if not text:
        return y
    safe_text = escape(str(text))
    styles = getSampleStyleSheet()
    base = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica-Oblique' if italic else 'Helvetica', fontSize=9, leading=11)
    p = Paragraph(safe_text, base)
    # crude paginate if needed
    w, h = p.wrap(width, y - 2 * cm)
    p.drawOn(can, left, y - h)
    return y - h - 0.2 * cm


def _player_name(elem: Dict) -> str:
    return f"{elem.get('first_name', '')} {elem.get('second_name', '')}".strip()


def _team_short(teams: List[Dict], team_id: Optional[int]) -> str:
    if not team_id:
        return "TBD"
    rec = next((t for t in teams if t.get('id') == team_id), None)
    return rec.get('short_name', 'TBD') if rec else 'TBD'


def _next_fixture_for_team(fixtures: List[Dict], team_id: int) -> Optional[Dict]:
    for fx in fixtures:
        if fx.get('finished') or fx.get('finished_provisional'):
            continue
        if fx.get('team_h') == team_id or fx.get('team_a') == team_id:
            return fx
    return None


def _fixture_line(fx: Dict, team_id: int, teams: List[Dict]) -> str:
    if not fx:
        return "No upcoming fixture"
    home = fx.get('team_h')
    away = fx.get('team_a')
    side = 'H' if team_id == home else 'A'
    opp_id = away if team_id == home else home
    opp = _team_short(teams, opp_id)
    fdr = fx.get('team_h_difficulty') if team_id == home else fx.get('team_a_difficulty')
    return f"{opp} ({side}) • FDR {fdr}"


def generate_appendix_pdf(
    output_dir: Path,
    team_summary: Dict,
    player_scores: Dict[int, Dict[str, float]] | None,
    elements: Dict[int, Dict],
    starters: List[int],
    subs: List[int],
    bootstrap: Dict,
    fixtures: List[Dict],
    differential_threshold: float = 10.0,
) -> Path:
    """Generate the Appendix PDF containing fixtures, differentials, bench order and risk flags.

    Args:
        output_dir: Directory to write the PDF into.
        team_summary: Team summary (name, rank, etc.).
        player_scores: Optional per-player predicted score dict {pid: {"total": float}}.
        elements: Bootstrap elements dict keyed by id.
        starters: List of 11 player ids.
        subs: List of bench player ids.
        bootstrap: Full bootstrap data (includes teams for short names).
        fixtures: List of fixture dicts for the target gameweek.
        differential_threshold: Ownership percent under which pick is considered a differential.

    Returns:
        Path to the created appendix PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"fpl_weekly_appendix_{ts}.pdf"

    can = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    left = 2 * cm
    right = 2 * cm
    text_width = width - left - right

    can.setFont("Helvetica-Bold", 16)
    can.drawString(left, y, "FPL Weekly Appendix")
    y -= 1.0 * cm

    can.setFont("Helvetica", 10)
    can.drawString(left, y, f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    y -= 0.7 * cm
    can.drawString(left, y, f"Team: {team_summary.get('Team Name', 'My Team')}")
    y -= 0.7 * cm

    teams = bootstrap.get('teams', []) if isinstance(bootstrap, dict) else []

    # Section: Fixtures for next GW (for my players)
    can.setFont("Helvetica-Bold", 12)
    can.drawString(left, y, "Upcoming Fixtures (Next GW)")
    y -= 0.6 * cm
    can.setFont("Helvetica", 10)

    def _render_player_line(pid: int) -> None:
        nonlocal y
        elem = elements.get(pid, {})
        name = _player_name(elem)
        team_id = elem.get('team')
        fx = _next_fixture_for_team(fixtures, team_id) if team_id else None
        fx_text = _fixture_line(fx, team_id, teams) if fx else "No upcoming fixture"
        ep_next = elem.get('ep_next')  # FPL's expected points
        score = player_scores.get(pid, {}).get('total') if player_scores else None
        pts_text = f" | Pred: {score:.2f}" if isinstance(score, (int, float)) else (f" | EP: {ep_next}" if ep_next else "")
        chance = elem.get('chance_of_playing_next_round')
        status = elem.get('status', 'a').lower()
        status_map = { 'a': 'Available', 'u': 'Unavailable', 'd': 'Doubtful', 's': 'Suspended', 'i': 'Injured', 'n': 'Not in squad' }
        status_text = status_map.get(status, status)
        tail = f" | {status_text}"
        if chance is not None and chance < 100:
            tail += f" ({int(chance)}%)"
        line = f"{name} — {fx_text}{pts_text}{tail}"
        can.drawString(left, y, line)
        y -= 0.45 * cm
        if y < 3 * cm:
            can.showPage(); y = height - 2 * cm; can.setFont("Helvetica", 10)

    for pid in starters:
        _render_player_line(pid)
    if subs:
        y -= 0.2 * cm
        can.setFont("Helvetica-Oblique", 10)
        can.drawString(left, y, "Bench:")
        y -= 0.4 * cm
        can.setFont("Helvetica", 10)
        for pid in subs:
            _render_player_line(pid)

    # Section: Differentials among my squad
    y -= 0.3 * cm
    if y < 3 * cm:
        can.showPage(); y = height - 2 * cm
    can.setFont("Helvetica-Bold", 12)
    can.drawString(left, y, f"Differentials (ownership < {differential_threshold:.1f}%)")
    y -= 0.6 * cm
    can.setFont("Helvetica", 10)

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

    my_squad = list(starters) + list(subs)
    diffs: List[Tuple[str, float]] = []
    for pid in my_squad:
        elem = elements.get(pid, {})
        own = _ownership(elem)
        if own is not None and own < differential_threshold:
            diffs.append((_player_name(elem), own))
    if diffs:
        diffs.sort(key=lambda x: x[1])
        for name, own in diffs:
            can.drawString(left, y, f"{name}: {own:.1f}%")
            y -= 0.4 * cm
            if y < 3 * cm:
                can.showPage(); y = height - 2 * cm; can.setFont("Helvetica", 10)
    else:
        can.drawString(left, y, "No differentials found under threshold.")
        y -= 0.5 * cm

    # Section: Risk flags
    y -= 0.2 * cm
    if y < 3 * cm:
        can.showPage(); y = height - 2 * cm
    can.setFont("Helvetica-Bold", 12)
    can.drawString(left, y, "Risk Flags (non-available players)")
    y -= 0.6 * cm
    can.setFont("Helvetica", 10)

    any_risk = False
    for pid in my_squad:
        elem = elements.get(pid, {})
        status = (elem.get('status') or 'a').lower()
        if status in {'d','i','s','u','n'}:
            any_risk = True
            name = _player_name(elem)
            chance = elem.get('chance_of_playing_next_round')
            msg = f"{name}: status={status.upper()}"
            if chance is not None:
                msg += f", chance={int(chance)}%"
            can.drawString(left, y, msg)
            y -= 0.4 * cm
            if y < 3 * cm:
                can.showPage(); y = height - 2 * cm; can.setFont("Helvetica", 10)
    if not any_risk:
        can.drawString(left, y, "No risk flags – all players available.")
        y -= 0.5 * cm

    can.showPage()
    can.save()
    return out_path
