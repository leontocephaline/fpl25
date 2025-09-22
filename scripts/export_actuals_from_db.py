from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def find_candidate_table(conn: sqlite3.Connection) -> Tuple[str, Dict[str, str]]:
    """Heuristically find the table with actual results and map its columns.

    Returns a tuple of (table_name, column_map) where column_map provides
    standardized keys: player_id, gameweek, actual_points, minutes, season.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]

    # Possible aliases for each required column
    col_aliases = {
        "player_id": ["player_id", "element", "player", "id"],
        "gameweek": ["gameweek", "gw", "round", "event"],
        "actual_points": ["actual_points", "total_points", "points", "score"],
        "minutes": ["minutes", "mins"],
        "season": ["season", "season_name", "season_tag"],
    }

    best: Optional[Tuple[str, Dict[str, str], int]] = None

    for t in tables:
        cursor.execute(f"PRAGMA table_info('{t}')")
        info = cursor.fetchall()
        names = {row[1].lower() for row in info}

        mapping: Dict[str, str] = {}
        score = 0
        for std_key, candidates in col_aliases.items():
            for cand in candidates:
                if cand.lower() in names:
                    mapping[std_key] = cand
                    score += 1
                    break
        # Require at least id, gameweek, points
        required = ["player_id", "gameweek", "actual_points"]
        if all(k in mapping for k in required):
            # Prefer tables that also have minutes
            if "minutes" in mapping:
                score += 1
            # Prefer tables that also have season
            if "season" in mapping:
                score += 1
            if best is None or score > best[2]:
                best = (t, mapping, score)

    if best is None:
        raise RuntimeError("Could not find a suitable table with player_id/gameweek/points")
    return best[0], best[1]


def export_actuals(
    db_path: Path,
    out_csv: Path,
    table: Optional[str] = None,
    season_fallback: Optional[str] = None,
) -> int:
    """Export actuals from SQLite DB to CSV with standardized columns.

    Columns: player_id, gameweek, actual_points, minutes, season
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        if table:
            # Build mapping from provided table
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info('{table}')")
            info = cursor.fetchall()
            if not info:
                raise RuntimeError(f"Table not found or empty: {table}")
            names = {row[1].lower() for row in info}
            # Build minimal mapping
            mapping: Dict[str, str] = {}
            def pick(*cands: str) -> Optional[str]:
                for c in cands:
                    if c.lower() in names:
                        return c
                return None
            mapping["player_id"] = pick("player_id", "element", "player", "id") or "id"
            mapping["gameweek"] = pick("gameweek", "gw", "round", "event") or "gameweek"
            mapping["actual_points"] = pick("actual_points", "total_points", "points", "score") or "total_points"
            mapping["minutes"] = pick("minutes", "mins") or "minutes"
            mapping["season"] = pick("season", "season_name", "season_tag") or None
            selected_table = table
        else:
            selected_table, mapping = find_candidate_table(conn)

        # Construct SQL
        cols_sql: List[str] = []
        cols_sql.append(f"{mapping['player_id']} AS player_id")
        cols_sql.append(f"{mapping['gameweek']} AS gameweek")
        cols_sql.append(f"{mapping['actual_points']} AS actual_points")
        if mapping.get("minutes"):
            cols_sql.append(f"{mapping['minutes']} AS minutes")
        else:
            cols_sql.append("CAST(NULL AS INTEGER) AS minutes")
        if mapping.get("season"):
            cols_sql.append(f"{mapping['season']} AS season")
        else:
            # Use fallback season if provided; else NULL
            if season_fallback:
                cols_sql.append(f"'{season_fallback}' AS season")
            else:
                cols_sql.append("CAST(NULL AS TEXT) AS season")

        sql = f"SELECT {', '.join(cols_sql)} FROM {selected_table}"
        df = pd.read_sql_query(sql, conn)

        # Normalize types
        for col in ("player_id", "gameweek"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        if "actual_points" in df.columns:
            df["actual_points"] = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0).astype(float)
        if "minutes" in df.columns:
            df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(float)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logging.getLogger(__name__).info("Exported %d rows from %s to %s", len(df), selected_table, out_csv)
        return len(df)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FPL actuals from SQLite DB to CSV")
    parser.add_argument("--db", required=True, help="Path to SQLite DB (e.g., data/fpl_history.db)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--table", help="Optional table name to read from")
    parser.add_argument("--season", help="Fallback season tag if not present in DB (e.g., 2024-25)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        n = export_actuals(Path(args.db), Path(args.out), table=args.table, season_fallback=args.season)
        print(f"Exported {n} rows to {args.out}")
        return 0
    except Exception as e:
        logging.exception("Failed to export actuals: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
