from __future__ import annotations

import datetime as dt
from pathlib import Path
import re
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import textwrap
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape


def generate_pdf(
    output_dir: Path,
    team_summary: Dict,
    player_scores: Dict[int, Dict[str, float]],
    news_summaries: Dict[str, Dict],
    transfers: List[Dict],
    name_to_pid: Dict[str, int] | None = None,
    pid_to_name: Dict[int, str] | None = None,
    elements: Dict[int, Dict] | None = None,
    my_team: Dict | None = None,
    starters: List[int] | None = None,
    subs: List[int] | None = None,
    bootstrap: Dict | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"fpl_weekly_update_{ts}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    left_margin = 2 * cm
    right_margin = 2 * cm
    max_text_width = width - left_margin - right_margin

    def wrap_by_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
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
                # Long single word fallback: hard-break if needed
                if stringWidth(w, font_name, font_size) > max_width:
                    # crude chunking by characters
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

    def page_break_reset_body_font():
        nonlocal y
        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica", 10)

    # Define paragraph styles for wrapped text
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=11
    )
    body_italic_style = ParagraphStyle(
        'BodyItalic', parent=body_style, fontName='Helvetica-Oblique'
    )

    def draw_paragraph(text: str, x_offset: float = 0.0, italic: bool = False):
        """Draw a paragraph with automatic wrapping and page breaks.

        Args:
            text: The text to render.
            x_offset: Additional left indent relative to left_margin.
            italic: Whether to use italic paragraph style.
        """
        nonlocal y
        if not text:
            return
        # Escape any XML-reserved characters to avoid Paragraph parse issues
        safe_text = escape(str(text))
        style = body_italic_style if italic else body_style
        max_width = (width - left_margin - right_margin) - x_offset
        while safe_text:
            p = Paragraph(safe_text, style)
            avail_h = y - 2 * cm
            w, h = p.wrap(max_width, avail_h)
            if h <= avail_h:
                p.drawOn(c, left_margin + x_offset, y - h)
                y -= h + 0.2 * cm
                break
            else:
                # Not enough space on this page; page break and retry
                page_break_reset_body_font()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "FPL Weekly Update")
    y -= 1 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    y -= 1 * cm

    # Team Summary Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Team Summary")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    
    # Show optimized lineup if we have the data
    if starters is not None and subs is not None and elements:
        # Starting 11
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left_margin, y, "Optimized Starting 11:")
        y -= 0.7 * cm
        c.setFont("Helvetica", 10)
        
        # Show optimized starting 11 (minimal info only)
        for player_id in starters:
            player = elements.get(player_id, {})
            player_name = f"{player.get('first_name', '')} {player.get('second_name', '')}"
            position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player.get('element_type', 0), 'UNK')

            # Team short name
            team_name = 'TBD'
            team_id = player.get('team')
            if team_id and isinstance(team_id, int):
                team_data = next((t for t in bootstrap.get('teams', []) if t.get('id') == team_id), {})
                team_name = team_data.get('short_name', 'TBD')

            # Availability (FPL status code)
            status = (player.get('status') or 'a').lower()
            status_text = {
                'a': 'Available', 'u': 'Unavailable', 'd': 'Doubtful', 's': 'Suspended', 'i': 'Injured', 'n': 'Not in squad'
            }.get(status, f'Unknown ({status})')

            # Score
            score = player_scores.get(player_id, {}).get('total', 0)

            # Start probability: prefer Perplexity news if present, else FPL chance
            news = news_summaries.get(player_name, {}) if news_summaries else {}
            prob = news.get('start_probability') or news.get('start_prob')
            if prob is None:
                chance = player.get('chance_of_playing_next_round')
                try:
                    prob = int(chance) if chance is not None else None
                except Exception:
                    prob = None

            # Minimal single-line display
            start_str = f" • Start: {int(prob)}%" if isinstance(prob, (int, float)) else ""
            c.drawString(left_margin, y, f"{position} - {player_name} ({team_name}): {status_text} (Score: {score:.1f}){start_str}")
            y -= 0.5 * cm
        
        # Subs
        if subs:  # Only show subs section if there are any subs
            y -= 0.3 * cm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(left_margin, y, "Substitutes:")
            y -= 0.7 * cm
            c.setFont("Helvetica", 10)
            
            # Show optimized substitutes
            for player_id in subs:
                player = elements.get(player_id, {})
                player_name = f"{player.get('first_name', '')} {player.get('second_name', '')}"
                position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player.get('element_type', 0), 'UNK')
                
                # Get team information for subs
                team_name = 'TBD'
                team_id = player.get('team')
                if team_id and isinstance(team_id, int):
                    # Get team data from bootstrap
                    team_data = next((t for t in bootstrap.get('teams', []) if t.get('id') == team_id), {})
                    team_name = team_data.get('short_name', 'TBD')
                
                # Get player status with more detailed information for subs (from FPL code)
                status = player.get('status', 'a').lower()
                status_text = {
                    'a': 'Available',
                    'u': 'Unavailable',
                    'd': 'Doubtful',
                    's': 'Suspended',
                    'i': 'Injured',
                    'n': 'Not in squad'
                }.get(status, f'Unknown ({status})')
                
                # Get player score for subs
                score = player_scores.get(player_id, {}).get('total', 0)
                
                # Add chance of playing next round if available
                chance = player.get('chance_of_playing_next_round')
                if chance is not None and chance < 100:
                    status_text += f" ({chance}% chance)"
                
                c.drawString(left_margin, y, f"{position} - {player_name} ({team_name}): {status_text} (Score: {score:.1f})")
                y -= 0.5 * cm
    
    # Original team summary
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Team Stats:")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for k, v in (team_summary or {}).items():
        # Wrap long values (e.g., players list)
        prefix = f"- {k}: "
        text = f"{v}" if v is not None else ""
        # Compute available width for first line (prefix eats width)
        first_line_max = max_text_width - stringWidth(prefix, "Helvetica", 10)
        lines = wrap_by_width(str(text), "Helvetica", 10, max_text_width)
        # First line includes the key prefix; recompute if needed
        first_payload = lines[0]
        if stringWidth(first_payload, "Helvetica", 10) > first_line_max:
            # re-wrap with reduced width for first line
            first_wrapped = wrap_by_width(str(text), "Helvetica", 10, first_line_max)
            first_payload = first_wrapped[0]
            rest = first_wrapped[1:] + lines[1:]
        else:
            rest = lines[1:]
        c.drawString(left_margin, y, prefix + first_payload)
        y -= 0.5 * cm
        for line in rest:
            c.drawString(left_margin + 1.2 * cm, y, line)
            y -= 0.5 * cm
            if y < 3 * cm:
                page_break_reset_body_font()
        if y < 3 * cm:
            page_break_reset_body_font()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Player Scores")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for pid, comps in (player_scores or {}).items():
        nm = (pid_to_name or {}).get(pid)
        label = f"Player {pid}" if not nm else f"{nm} ({pid})"
        c.drawString(left_margin, y, f"{label}: total={comps.get('total', 0):.2f}")
        y -= 0.4 * cm
        if y < 3 * cm:
            page_break_reset_body_font()


    # Add space before captain section
    y -= 0.5 * cm
    
    # Captain and vice-captain selection
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y, "Captain & Vice-Captain")
    y -= 1.0 * cm
    
    # Get top 5 players by score for captain selection
    if player_scores and pid_to_name:
        sorted_players = sorted(
            [(pid, score.get('total', 0)) for pid, score in player_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 players
        
        # Captain section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, "Captain")
        c.setFont("Helvetica", 10)
        y -= 0.6 * cm
        
        if sorted_players:
            captain_id, captain_score = sorted_players[0]
            captain_name = pid_to_name.get(captain_id, f"Player {captain_id}")
            c.drawString(left_margin + 0.5 * cm, y, f"{captain_name}")
            c.drawRightString(width - right_margin, y, f"Score: {captain_score:.2f}")
        y -= 0.8 * cm
        
        # Vice-Captain section with space
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, "Vice-Captain")
        c.setFont("Helvetica", 10)
        y -= 0.6 * cm
        
        if len(sorted_players) > 1:
            vc_id, vc_score = sorted_players[1]
            vc_name = pid_to_name.get(vc_id, f"Player {vc_id}")
            c.drawString(left_margin + 0.5 * cm, y, f"{vc_name}")
            c.drawRightString(width - right_margin, y, f"Score: {vc_score:.2f}")
        y -= 1.2 * cm
        
        # Add divider line
        c.line(left_margin, y, width - right_margin, y)
        y -= 0.8 * cm
    
    # Move player news onto a fresh page for readability
    page_break_reset_body_font()

    # Show player news and updates
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Player News & Updates")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    
    player_news_entries: List[Dict[str, Any]] = []
    added_names: set[str] = set()

    def collect_news_entry(pid: int | None, name: str, raw_info: Dict | None) -> None:
        nonlocal player_news_entries
        if not name:
            return
        info_dict = raw_info if isinstance(raw_info, dict) else {}
        info_copy: Dict[str, Any] = {k: v for k, v in info_dict.items() if v is not None}

        score = 0.0
        if pid is not None and player_scores:
            score = float((player_scores.get(pid) or {}).get('total', 0.0))

        fpl_elem = elements.get(pid, {}) if elements and pid is not None else {}
        fpl_code = (fpl_elem.get('status') or '').lower() if fpl_elem else ''
        fpl_chance_val = fpl_elem.get('chance_of_playing_next_round') if fpl_elem else None
        try:
            fpl_chance = int(fpl_chance_val) if fpl_chance_val is not None else None
        except Exception:
            fpl_chance = None

        heading_status = (info_copy.get('status') or fpl_code or 'unknown').lower()

        news_text = (info_copy.get('news') or info_copy.get('summary') or '').strip()
        if '```' in news_text or '``' in news_text:
            news_text = re.sub(r'`+\s*json\s*{[^`]+}\s*`+', '', news_text, flags=re.DOTALL | re.IGNORECASE)
            news_text = re.sub(r'`+\s*{[^`]+}\s*`+', '', news_text, flags=re.DOTALL)
            news_text = news_text.strip()
            if not news_text:
                summary_raw = (info_copy.get('summary') or 'No recent updates available.').strip()
                summary_raw = re.sub(r'`+\s*(?:json)?\s*{[^`]+}\s*`+', '', summary_raw, flags=re.DOTALL | re.IGNORECASE)
                news_text = summary_raw.strip()

        prob_val = info_copy.get('start_probability') or info_copy.get('start_prob')
        if prob_val is None:
            prob_val = fpl_chance
        prob_int = None
        if prob_val is not None:
            try:
                prob_int = int(prob_val)
                if prob_int <= 0:
                    prob_int = None
            except Exception:
                prob_int = None

        if not news_text:
            news_text = "No recent updates available from trusted sources in the past week."

        if prob_int is not None and prob_int < 100:
            news_text = f"({prob_int}% chance) {news_text}"
        elif prob_int is None and fpl_chance is not None and fpl_chance < 100:
            news_text = f"(FPL {fpl_chance}% chance) {news_text}"

        injury = info_copy.get('injury_status')
        if isinstance(injury, str) and injury.lower() not in ['none', 'available', 'fit', 'null'] and injury.strip():
            news_text = f"{injury.strip().capitalize()}. {news_text}"

        player_news_entries.append({
            'name': name,
            'pid': pid,
            'score': score,
            'news': news_text,
            'status': heading_status,
            'info': info_copy,
            'prob_int': prob_int,
            'fpl_chance': fpl_chance
        })
        added_names.add(name)

    # Prioritise starters and subs
    team_player_ids: List[int] = []
    if starters:
        team_player_ids.extend(starters)
    if subs:
        team_player_ids.extend(subs)

    ordered_unique_ids = list(dict.fromkeys(team_player_ids))
    for pid in ordered_unique_ids:
        name = pid_to_name.get(pid, f"Player {pid}") if pid_to_name else f"Player {pid}"
        info = (news_summaries or {}).get(name)
        collect_news_entry(pid, name, info)

    # Include any remaining news entries not already covered
    if news_summaries:
        for name, info in news_summaries.items():
            if name in added_names:
                continue
            pid = name_to_pid.get(name) if name_to_pid else None
            collect_news_entry(pid, name, info)

    if not player_news_entries:
        c.drawString(left_margin, y, "No significant player news or updates available this week.")
        y -= 0.5 * cm
    else:
        player_news_entries.sort(key=lambda x: x['score'], reverse=True)

        for entry in player_news_entries:
            if y < 3 * cm:
                page_break_reset_body_font()
                y = height - 2 * cm

            raw_status = (entry['status'] or '').lower()
            status_text = {
                'fit': 'Available',
                'available': 'Available',
                'a': 'Available',
                'no recent update': 'No Recent Update',
                'unknown': 'Unknown',
                'injured': 'Injured',
                'i': 'Injured',
                'doubtful': 'Doubtful',
                'd': 'Doubtful',
                'suspended': 'Suspended',
                's': 'Suspended',
                'unavailable': 'Unavailable',
                'u': 'Unavailable'
            }.get(raw_status, entry['status'].capitalize() if entry['status'] else 'Unknown')

            c.setFont("Helvetica-Bold", 10)
            c.drawString(left_margin, y, f"{entry['name']} ({status_text}):")
            y -= 0.3 * cm

            draw_paragraph(entry['news'], x_offset=0.5 * cm, italic=False)

            info = entry.get('info', {})
            if isinstance(info, dict) and info:
                bullets = []
                status_b = info.get('status') or entry['status']
                if status_b:
                    s_raw = str(status_b).lower()
                    status_map = {
                        'fit': 'Available','available': 'Available','a': 'Available',
                        'no recent update': 'No Recent Update','unknown': 'Unknown',
                        'injured': 'Injured','i': 'Injured','doubtful': 'Doubtful','d': 'Doubtful',
                        'suspended': 'Suspended','s': 'Suspended','unavailable': 'Unavailable','u': 'Unavailable'
                    }
                    bullets.append(f"• Status: {status_map.get(s_raw, status_b)}")
                prob_val = entry.get('prob_int')
                if prob_val is not None and prob_val > 0:
                    bullets.append(f"• Start Probability: {prob_val}%")
                elif (info.get('start_probability') or info.get('start_prob')) is not None:
                    try:
                        sp_i = int(info.get('start_probability') or info.get('start_prob'))
                        if sp_i > 0:
                            bullets.append(f"• Start Probability: {sp_i}%")
                    except Exception:
                        bullets.append(f"• Start Probability: {info.get('start_probability') or info.get('start_prob')}")
                chance_val = entry.get('fpl_chance')
                if chance_val is not None and chance_val < 100:
                    bullets.append(f"• Chance (FPL): {chance_val}%")
                inj = info.get('injury_status')
                if inj is not None:
                    txt = str(inj).strip()
                    if txt and txt.lower() not in {'none','null','available','fit'}:
                        bullets.append(f"• Injury Details: {txt}")
                exp = info.get('expected_return')
                if exp is not None:
                    exp_txt = str(exp).strip()
                    if exp_txt:
                        bullets.append(f"• Expected Return: {exp_txt}")
                conf = info.get('confidence')
                if conf is not None:
                    try:
                        conf_f = float(conf)
                        if conf_f > 0:
                            bullets.append(f"• Confidence: {conf_f:.2f}")
                    except Exception:
                        ctxt = str(conf).strip()
                        if ctxt:
                            bullets.append(f"• Confidence: {ctxt}")
                rec = info.get('transfer_recommendation')
                if rec:
                    bullets.append(f"• Transfer Recommendation: {rec}")
                for bl in bullets:
                    draw_paragraph(bl, x_offset=0.5 * cm, italic=False)

            if entry['score'] > 0:
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(left_margin + 0.5 * cm, y, f"[Projected points: {entry['score']:.2f}]")
                y -= 0.4 * cm

            y -= 0.2 * cm

    # Show transfer recommendations if any
    if transfers:
        y -= 0.5 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, "Recommended Transfers:")
        y -= 0.7 * cm
        c.setFont("Helvetica", 10)
        
        for i, t in enumerate(transfers, 1):
            out_name = t.get('out_name', f"Player {t.get('out')}")
            in_name = t.get('in_name', f"Player {t.get('in')}")
            gain = t.get('gain', 0)
            hit = t.get('hit', 0)
            net = t.get('net', 0)
            
            # Get player positions and teams
            out_pos = ""
            in_pos = ""
            out_team = ""
            in_team = ""
            
            # Get outgoing player info
            if elements and 'out' in t and t['out'] in elements:
                out_player = elements[t['out']]
                out_pos_code = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(out_player.get('element_type', 0), 'UNK')
                out_pos = f"({out_pos_code})"
                out_team_id = out_player.get('team')
                if out_team_id and bootstrap and 'teams' in bootstrap:
                    out_team_data = next((team for team in bootstrap['teams'] 
                                        if team.get('id') == out_team_id), {})
                    out_team = out_team_data.get('short_name', '')
            
            # Get incoming player info
            if elements and 'in' in t and t['in'] in elements:
                in_player = elements[t['in']]
                in_pos_code = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(in_player.get('element_type', 0), 'UNK')
                in_pos = f"({in_pos_code})"
                in_team_id = in_player.get('team')
                if in_team_id and bootstrap and 'teams' in bootstrap:
                    in_team_data = next((team for team in bootstrap['teams'] 
                                       if team.get('id') == in_team_id), {})
                    in_team = in_team_data.get('short_name', '')
            
            # Format transfer line
            transfer_line = f"{i}. {out_pos} {out_name} ({out_team}) → {in_pos} {in_name} ({in_team})"
            
            # Format stats line
            stats_line = []
            if gain is not None:
                stats_line.append(f"Gain: {gain:.2f}")
            if hit is not None and hit > 0:
                stats_line.append(f"Hit: -{hit}")
            if net is not None:
                stats_line.append(f"Net: {net:+.2f}")
            
            # Draw transfer line
            c.drawString(left_margin, y, transfer_line)
            y -= 0.5 * cm
            
            # Draw stats line indented
            if stats_line:
                stats_text = " | ".join(stats_line)
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(left_margin + 0.5 * cm, y, stats_text)
                c.setFont("Helvetica", 10)
                y -= 0.5 * cm
            
            y -= 0.3 * cm  # Extra space between transfers
            
            if y < 3 * cm:
                page_break_reset_body_font()
    else:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, "No transfers recommended.")
        y -= 0.5 * cm

    c.showPage()
    c.save()
    return out_path
