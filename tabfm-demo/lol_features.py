"""Build TabFM input rows for full-draft match winner prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINNER_LABELS = ("TEAM_100", "TEAM_200")
POSITION_ORDER = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

# Mixed categorical draft picks + one numeric column — TabFM's expected input shape.
TABFM_FEATURE_COLUMNS: list[str] = [
    *[f"team100_{pos.lower()}" for pos in POSITION_ORDER],
    *[f"team200_{pos.lower()}" for pos in POSITION_ORDER],
    "duration_seconds",
]


def _champion_names_by_position(team_df: pd.DataFrame) -> list[str]:
    pos_map = {
        row.position: str(row.champion_name)
        for row in team_df.itertuples(index=False)
        if row.position in POSITION_ORDER
    }
    return [pos_map.get(pos, "") for pos in POSITION_ORDER]


def participants_to_drafts(participants: pd.DataFrame) -> pd.DataFrame:
    """One row per complete 5v5 match with both drafts and the winner."""
    rows: list[dict] = []
    for match_id, group in participants.groupby("match_id", sort=False):
        if len(group) != 10:
            continue
        winners = group.loc[group["match_result"] == "WIN", "team_id"].unique()
        if len(winners) != 1:
            continue
        winning_team = int(winners[0])

        team100 = group[group["team_id"] == 100]
        team200 = group[group["team_id"] == 200]
        if len(team100) != 5 or len(team200) != 5:
            continue

        team100_names = _champion_names_by_position(team100)
        team200_names = _champion_names_by_position(team200)
        if "" in team100_names or "" in team200_names:
            continue

        row: dict = {
            "match_id": match_id,
            "duration_seconds": int(group["duration_seconds"].iloc[0]),
            "winning_team": winning_team,
            "winner": "TEAM_100" if winning_team == 100 else "TEAM_200",
        }
        for pos, name in zip(POSITION_ORDER, team100_names, strict=True):
            row[f"team100_{pos.lower()}"] = name
        for pos, name in zip(POSITION_ORDER, team200_names, strict=True):
            row[f"team200_{pos.lower()}"] = name
        row["team100_champion_names"] = ",".join(team100_names)
        row["team200_champion_names"] = ",".join(team200_names)
        rows.append(row)
    return pd.DataFrame(rows)


def drafts_to_tabfm_frame(drafts: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return X/y in the shape TabFM expects: raw columns, no feature engineering."""
    X = drafts[TABFM_FEATURE_COLUMNS].copy()
    # Explicit dtypes so TabFM treats picks as categorical and duration as numeric.
    for col in TABFM_FEATURE_COLUMNS[:-1]:
        X[col] = X[col].astype("string")
    X["duration_seconds"] = X["duration_seconds"].astype("float64")
    y = drafts["winner"].to_numpy()
    return X, y


def train_test_split_by_match(
    participants: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drafts = participants_to_drafts(participants)
    match_ids = np.array(drafts["match_id"].unique())
    rng = np.random.default_rng(random_state)
    rng.shuffle(match_ids)
    split = int(len(match_ids) * (1 - test_size))
    train_ids = set(match_ids[:split])
    train = drafts[drafts["match_id"].isin(train_ids)].reset_index(drop=True)
    test = drafts[~drafts["match_id"].isin(train_ids)].reset_index(drop=True)
    return train, test
