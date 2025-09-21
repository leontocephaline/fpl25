#!/usr/bin/env python3
"""
Safely upsert Teams and Fixtures (including difficulty/strength columns) into SQLite
without altering players or gameweeks.

Usage:
  python scripts/update_fixtures_teams.py --season 2024-25
  python scripts/update_fixtures_teams.py --season 2023-24 --season 2024-25

Notes:
- This script only updates `teams` and `fixtures` tables via UPSERT.
- Primary keys:
  - teams: (team_id, season)
  - fixtures: fixture_id
- Columns included follow src/data/data_importer.py but use ON CONFLICT upserts.
- It is idempotent and safe to re-run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List, Tuple

import pandas as pd
import requests

from src.data.historical_db import FPLHistoricalDB

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


@dataclass
class Fetcher:
    user_agent: str = "FPL-Backtesting-Updater/1.0"

    def get_csv(self, url: str) -> pd.DataFrame:
        sess = requests.Session()
        sess.headers.update({"User-Agent": self.user_agent})
        resp = sess.get(url, timeout=60)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))


def upsert_teams(db: FPLHistoricalDB, df: pd.DataFrame, season: str) -> int:
    # Rename and select columns consistent with teams table
    df = df.rename(columns={
        "id": "team_id",
        "name": "name",
        "short_name": "short_name",
    }).copy()
    df["season"] = season

    # Candidate columns (some may not exist in the source)
    possible_cols = [
        "team_id", "season", "name", "short_name",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    cols = [c for c in possible_cols if c in df.columns]
    if not cols:
        return 0

    # Build UPSERT statement (exclude PKs from update set)
    update_cols = [c for c in cols if c not in ("team_id", "season")]
    placeholders = ", ".join([":" + c for c in cols])
    set_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    sql = f"""
        INSERT INTO teams ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(team_id, season) DO UPDATE SET {set_clause}
    """

    data = df[cols].to_dict(orient="records")
    with db.conn:
        db.conn.executemany(sql, data)
    return len(data)


def upsert_fixtures(db: FPLHistoricalDB, df: pd.DataFrame, season: str) -> int:
    # Normalize consistent with fixtures table
    df = df.rename(columns={
        "id": "fixture_id",
        "event": "gw",
        "team_h": "team_h",
        "team_a": "team_a",
        "team_h_score": "team_h_score",
        "team_a_score": "team_a_score",
        "finished": "finished",
        "kickoff_time": "kickoff_time",
        # difficulties (if present)
        "team_h_difficulty": "team_h_difficulty",
        "team_a_difficulty": "team_a_difficulty",
    }).copy()
    df["season"] = season

    possible_cols = [
        "fixture_id", "season", "gw", "team_h", "team_a", "team_h_score", "team_a_score",
        "finished", "kickoff_time", "team_h_difficulty", "team_a_difficulty",
    ]
    cols = [c for c in possible_cols if c in df.columns]
    if "fixture_id" not in cols:
        return 0

    update_cols = [c for c in cols if c != "fixture_id"]
    placeholders = ", ".join([":" + c for c in cols])
    set_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    sql = f"""
        INSERT INTO fixtures ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(fixture_id) DO UPDATE SET {set_clause}
    """

    data = df[cols].to_dict(orient="records")
    with db.conn:
        db.conn.executemany(sql, data)
    return len(data)


def update_one_season(db: FPLHistoricalDB, fetcher: Fetcher, season: str) -> Tuple[int, int]:
    teams_url = f"{BASE_URL}/{season}/teams.csv"
    fixtures_url = f"{BASE_URL}/{season}/fixtures.csv"

    teams_df = fetcher.get_csv(teams_url)
    fixtures_df = fetcher.get_csv(fixtures_url)

    n_teams = upsert_teams(db, teams_df, season)
    n_fixtures = upsert_fixtures(db, fixtures_df, season)
    return n_teams, n_fixtures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upsert teams and fixtures (difficulty/strength) into SQLite")
    p.add_argument("--season", action="append", required=True, help="Season string (e.g., 2024-25). Can be repeated.")
    p.add_argument("--db-path", default=None, help="Optional custom DB path (defaults to data/fpl_history.db)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    db = FPLHistoricalDB(args.db_path) if args.db_path else FPLHistoricalDB()
    fetcher = Fetcher()

    total_t, total_f = 0, 0
    for season in args.season:
        n_t, n_f = update_one_season(db, fetcher, season)
        print(f"Season {season}: upserted teams={n_t}, fixtures={n_f}")
        total_t += n_t
        total_f += n_f

    print(f"Done. Total teams rows upserted: {total_t}; fixtures rows upserted: {total_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
