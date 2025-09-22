from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

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
        
        # Show optimized starting 11
        for player_id in starters:
            player = elements.get(player_id, {})
            player_name = f"{player.get('first_name', '')} {player.get('second_name', '')}"
            position = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player.get('element_type', 0), 'UNK')
            
            # Get team information
            team_name = 'TBD'
            team_id = player.get('team')
            if team_id and isinstance(team_id, int):
                # Get team data from bootstrap
                team_data = next((t for t in bootstrap.get('teams', []) if t.get('id') == team_id), {})
                team_name = team_data.get('short_name', 'TBD')
            
            # Get player status with more detailed information (from FPL code)
            status = player.get('status', 'a').lower()
            status_text = {
                'a': 'Available',
                'u': 'Unavailable',
                'd': 'Doubtful',
                's': 'Suspended',
                'i': 'Injured',
                'n': 'Not in squad'
            }.get(status, f'Unknown ({status})')
            
            # Get player score
            score = player_scores.get(player_id, {}).get('total', 0)
            
            # Add chance of playing next round if available
            chance = player.get('chance_of_playing_next_round')
            if chance is not None and chance < 100:
                status_text += f" ({chance}% chance)"
                
            # Display player info
            c.drawString(left_margin, y, f"{position} - {player_name} ({team_name}): {status_text} (Score: {score:.1f})")
            y -= 0.5 * cm
            
            # Add news if available
            news = news_summaries.get(player_name, {})
            news_text = ''
            
            # Check if we have a proper news summary
            if isinstance(news, dict) and 'summary' in news and news['summary']:
                news_text = news['summary']
                if ' | ' in news_text:
                    # Take the most relevant part before the separator
                    news_text = news_text.split(' | ')[0]
                
                # Remove any duplicate sentences
                sentences = news_text.split('. ')
                unique_sentences = []
                for s in sentences:
                    s = s.strip()
                    if s and s not in unique_sentences:
                        unique_sentences.append(s)
                news_text = '. '.join(unique_sentences)
                
                # Add probability if available
                prob = news.get('start_probability') or news.get('start_prob')
                if prob is not None:
                    try:
                        prob = int(prob)
                        if prob > 0:
                            news_text = f"({prob}% chance) {news_text}"
                    except (ValueError, TypeError):
                        pass
                
                # Truncate if too long
                if len(news_text) > 200:
                    news_text = news_text[:197] + '...'
            # If no proper news but we have a score, show that
            elif 'total' in news and news['total'] is not None:
                news_text = f"Score: {news['total']:.2f}"
                prob = news.get('start_probability') or news.get('start_prob')
                if prob is not None:
                    try:
                        prob = int(prob)
                        news_text = f"{prob}% chance | {news_text}"
                    except (ValueError, TypeError):
                        pass
            
            if news_text:
                # Render wrapped news text with automatic page breaks
                draw_paragraph(f"→ {news_text}", x_offset=0.5 * cm, italic=True)
                
            # Render structured bullet items if available
            if news_summaries:
                info = news_summaries.get(player_name, {})
                if isinstance(info, dict) and info:
                    bullets = []
                    status_b = info.get('status')
                    if status_b:
                        bullets.append(f"• Status: {status_b}")
                    sp = info.get('start_probability') or info.get('start_prob')
                    if sp is not None:
                        try:
                            sp_i = int(sp)
                            bullets.append(f"• Start Probability: {sp_i}%")
                        except Exception:
                            bullets.append(f"• Start Probability: {sp}")
                    # FPL authoritative chance if available
                    chance = player.get('chance_of_playing_next_round')
                    if chance is not None:
                        try:
                            chance_i = int(chance)
                            bullets.append(f"• Chance (FPL): {chance_i}%")
                        except Exception:
                            bullets.append(f"• Chance (FPL): {chance}")
                    inj = info.get('injury_status')
                    if inj is not None:
                        bullets.append(f"• Injury Details: {inj if inj else 'None'}")
                    exp = info.get('expected_return')
                    if exp is not None:
                        bullets.append(f"• Expected Return: {exp if exp else 'N/A'}")
                    conf = info.get('confidence')
                    if conf is not None:
                        # If numeric, format; if text, show as-is
                        try:
                            conf_f = float(conf)
                            bullets.append(f"• Confidence: {conf_f:.2f}")
                        except Exception:
                            bullets.append(f"• Confidence: {conf}")
                    rec = info.get('transfer_recommendation')
                    if rec:
                        bullets.append(f"• Transfer Recommendation: {rec}")
                    for bl in bullets:
                        draw_paragraph(bl, x_offset=0.5 * cm, italic=False)
            y -= 0.1 * cm
        
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
    
    # Show player news and updates
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Player News & Updates")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    
    if news_summaries:
        # Group news by player and sort by score
        player_news = []
        for name, info in news_summaries.items():
            if not info:
                continue
                
            # Get player ID and score if available
            pid = name_to_pid.get(name) if name_to_pid else None
            score = player_scores.get(pid, {}).get('total', 0) if pid and player_scores else 0
            
            # Get news text - prefer detailed news, fall back to summary
            news_text = info.get('news') or info.get('summary') or "No news available"
            
            # Add start probability if available
            prob = info.get('start_probability')
            if prob is not None:
                news_text = f"({prob}% chance) {news_text}"
                
            # Add injury status if available
            injury = info.get('injury_status')
            if injury and injury.lower() not in ['none', 'available', 'fit']:
                news_text = f"{injury.capitalize()}. {news_text}"
                
            player_news.append({
                'name': name,
                'pid': pid,
                'score': score,
                'news': news_text,
                'status': info.get('status', 'unknown')
            })
        
        # Sort by score (highest first)
        player_news.sort(key=lambda x: x['score'], reverse=True)
        
        # Display news for each player
        for p in player_news:
            if y < 3 * cm:
                page_break_reset_body_font()
                y = height - 2 * cm
                
            # Player name and status
            status_text = {
                'a': 'Available',
                'u': 'Unavailable',
                'd': 'Doubtful',
                's': 'Suspended',
                'i': 'Injured'
            }.get((p['status'] or '').lower(), str(p['status']).capitalize())

            c.setFont("Helvetica-Bold", 10)
            c.drawString(left_margin, y, f"{p['name']} ({status_text}):")
            y -= 0.3 * cm

            # News text with robust wrapping
            draw_paragraph(p['news'], x_offset=0.5 * cm, italic=False)

            # Structured bullet lines under each player's news if available
            info = news_summaries.get(p['name'], {}) if news_summaries else {}
            if isinstance(info, dict) and info:
                bullets = []
                status_b = info.get('status')
                if status_b:
                    bullets.append(f"• Status: {status_b}")
                sp = info.get('start_probability') or info.get('start_prob')
                if sp is not None:
                    try:
                        sp_i = int(sp)
                        bullets.append(f"• Start Probability: {sp_i}%")
                    except Exception:
                        bullets.append(f"• Start Probability: {sp}")
                # FPL authoritative chance if available (look up by player name → id → element)
                if name_to_pid and elements:
                    p_id = name_to_pid.get(p['name'])
                    if p_id is not None:
                        elem = elements.get(p_id, {})
                        chance = elem.get('chance_of_playing_next_round')
                        if chance is not None:
                            try:
                                chance_i = int(chance)
                                bullets.append(f"• Chance (FPL): {chance_i}%")
                            except Exception:
                                bullets.append(f"• Chance (FPL): {chance}")
                inj = info.get('injury_status')
                if inj is not None:
                    bullets.append(f"• Injury Details: {inj if inj else 'None'}")
                exp = info.get('expected_return')
                if exp is not None:
                    bullets.append(f"• Expected Return: {exp if exp else 'N/A'}")
                conf = info.get('confidence')
                if conf is not None:
                    try:
                        conf_f = float(conf)
                        bullets.append(f"• Confidence: {conf_f:.2f}")
                    except Exception:
                        bullets.append(f"• Confidence: {conf}")
                rec = info.get('transfer_recommendation')
                if rec:
                    bullets.append(f"• Transfer Recommendation: {rec}")
                for bl in bullets:
                    draw_paragraph(bl, x_offset=0.5 * cm, italic=False)

            # Add score if available
            if p['score'] > 0:
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(left_margin + 0.5 * cm, y, f"[Projected points: {p['score']:.2f}]")
                y -= 0.4 * cm
            
            y -= 0.2 * cm  # Space between players
    else:
        c.drawString(left_margin, y, "No player news available.")
        y -= 0.5 * cm

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
