#!/usr/bin/env python3
"""Export LoL participant rows from Postgres for the TabFM demo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:
    raise SystemExit(
        "psycopg is not installed correctly. In this directory run:\n"
        "  python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install 'psycopg[binary]' python-dotenv pandas\n"
        f"\nOriginal error: {exc}"
    ) from exc

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "lol_participants.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CANDIDATES = (
    Path.cwd() / ".env",
    REPO_ROOT / "db details" / ".env",
    REPO_ROOT / ".env",
)

PARTICIPANT_SQL = """
SELECT
    mp.match_id,
    mp.player_id,
    mp.champion_id,
    c.champion_name,
    mp.team_id,
    mp.match_result,
    mp.position,
    m.duration_seconds
FROM match_participants mp
JOIN matches m ON m.match_id = mp.match_id
JOIN champions c ON c.champion_id = mp.champion_id
WHERE mp.match_result IN ('WIN', 'LOSS')
ORDER BY mp.match_id, mp.team_id, mp.champion_id
"""


def db_connect() -> psycopg.Connection:
    host = os.environ["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.environ["POSTGRES_PASSWORD"]
    dbname = os.getenv("POSTGRES_DB", "postgres")
    return psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        row_factory=dict_row,
    )


def enrich_with_teams(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for match_id, group in df.groupby("match_id", sort=False):
        by_team: dict[int, list[int]] = {100: [], 200: []}
        for row in group.itertuples(index=False):
            by_team[int(row.team_id)].append(int(row.champion_id))

        for row in group.itertuples(index=False):
            team_id = int(row.team_id)
            champ_id = int(row.champion_id)
            enemy_team = 200 if team_id == 100 else 100
            allies = sorted(c for c in by_team[team_id] if c != champ_id)
            enemies = sorted(by_team[enemy_team])
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": row.player_id,
                    "champion_id": champ_id,
                    "champion_name": row.champion_name,
                    "team_id": team_id,
                    "match_result": row.match_result,
                    "position": row.position,
                    "duration_seconds": int(row.duration_seconds),
                    "ally_champion_ids": ",".join(str(c) for c in allies),
                    "enemy_champion_ids": ",".join(str(c) for c in enemies),
                }
            )
    return pd.DataFrame(rows)


def fetch_participants(conn: psycopg.Connection) -> pd.DataFrame:
    raw = pd.DataFrame(conn.execute(PARTICIPANT_SQL).fetchall())
    if raw.empty:
        raise RuntimeError("No WIN/LOSS participants found in the database.")
    return enrich_with_teams(raw)


def load_env() -> None:
    for path in ENV_CANDIDATES:
        if path.is_file():
            load_dotenv(path)
            return
    load_dotenv()


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    missing = [k for k in ("POSTGRES_HOST", "POSTGRES_PASSWORD") if not os.getenv(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    with db_connect() as conn:
        participants = fetch_participants(conn)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    participants.to_csv(args.output, index=False)
    print(f"Wrote {len(participants)} rows to {args.output}")
    print(f"  matches: {participants['match_id'].nunique()}")
    print(f"  champions: {participants['champion_id'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
