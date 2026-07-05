"""Build TabFM feature rows for full-draft match winner prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# 14 duration buckets (seconds): features 2–15.
DURATION_BOUNDS = (
    0,
    1200,
    1320,
    1440,
    1560,
    1680,
    1800,
    1920,
    2040,
    2160,
    2280,
    2400,
    2580,
    2700,
    10_000_000,
)
DURATION_LABELS = (
    "00_20m",
    "20_22m",
    "22_24m",
    "24_26m",
    "26_28m",
    "28_30m",
    "30_32m",
    "32_34m",
    "34_36m",
    "36_38m",
    "38_40m",
    "40_43m",
    "43_45m",
    "45m_plus",
)

# Team 100 strength minus team 200 strength (positive → favors team 100).
FEATURE_NAMES: list[str] = [
    "overall_win_rate_diff",
    *[f"wr_duration_{label}_diff" for label in DURATION_LABELS],
    *[f"adv_vs_enemy_{i}_diff" for i in range(1, 6)],
    *[f"syn_with_ally_{i}_diff" for i in range(1, 5)],
]

WINNER_LABELS = ("TEAM_100", "TEAM_200")


def duration_bucket(seconds: int) -> int:
    for idx in range(len(DURATION_LABELS)):
        if DURATION_BOUNDS[idx] <= seconds < DURATION_BOUNDS[idx + 1]:
            return idx
    return len(DURATION_LABELS) - 1


def _win_rate(wins: int, games: int, default: float = 0.5) -> float:
    if games == 0:
        return default
    return wins / games


@dataclass
class ChampionStats:
    overall: dict[int, list[int]] = field(default_factory=dict)
    by_duration: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    vs_enemy: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    with_ally: dict[int, dict[int, list[int]]] = field(default_factory=dict)

    def _bump(self, table: dict, key: int, won: bool) -> None:
        wins, games = table.setdefault(key, [0, 0])
        games += 1
        if won:
            wins += 1
        table[key] = [wins, games]

    def _bump_nested(
        self,
        table: dict[int, dict[int, list[int]]],
        hero_id: int,
        other_id: int,
        won: bool,
    ) -> None:
        hero_table = table.setdefault(hero_id, {})
        self._bump(hero_table, other_id, won)

    def observe(
        self,
        champion_id: int,
        won: bool,
        duration_seconds: int,
        enemy_ids: Iterable[int],
        ally_ids: Iterable[int],
    ) -> None:
        self._bump(self.overall, champion_id, won)
        bucket = duration_bucket(duration_seconds)
        hero_duration = self.by_duration.setdefault(champion_id, {})
        self._bump(hero_duration, bucket, won)
        for enemy_id in enemy_ids:
            self._bump_nested(self.vs_enemy, champion_id, enemy_id, won)
        for ally_id in ally_ids:
            self._bump_nested(self.with_ally, champion_id, ally_id, won)


def build_champion_stats(participants: pd.DataFrame) -> ChampionStats:
    stats = ChampionStats()
    for row in participants.itertuples(index=False):
        won = row.match_result == "WIN"
        enemies = [int(x) for x in row.enemy_champion_ids.split(",") if x]
        allies = [int(x) for x in row.ally_champion_ids.split(",") if x]
        stats.observe(
            int(row.champion_id),
            won,
            int(row.duration_seconds),
            enemies,
            allies,
        )
    return stats


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

        team100 = group[group["team_id"] == 100].sort_values("champion_id")
        team200 = group[group["team_id"] == 200].sort_values("champion_id")
        if len(team100) != 5 or len(team200) != 5:
            continue

        rows.append(
            {
                "match_id": match_id,
                "duration_seconds": int(group["duration_seconds"].iloc[0]),
                "winning_team": winning_team,
                "winner": "TEAM_100" if winning_team == 100 else "TEAM_200",
                "team100_champion_ids": ",".join(str(c) for c in team100["champion_id"]),
                "team200_champion_ids": ",".join(str(c) for c in team200["champion_id"]),
                "team100_champion_names": ",".join(team100["champion_name"]),
                "team200_champion_names": ",".join(team200["champion_name"]),
            }
        )
    return pd.DataFrame(rows)


def _team_draft_features(
    team_champion_ids: list[int],
    enemy_champion_ids: list[int],
    stats: ChampionStats,
) -> dict[str, float]:
    """Draft features for one side: mean hero WR, counters vs each enemy, ally synergy."""
    overall_wrs = [_win_rate(*stats.overall.get(c, [0, 0])) for c in team_champion_ids]
    features: dict[str, float] = {"overall_win_rate": float(np.mean(overall_wrs))}

    for idx, label in enumerate(DURATION_LABELS):
        wrs = [
            _win_rate(*stats.by_duration.get(c, {}).get(idx, [0, 0]))
            for c in team_champion_ids
        ]
        features[f"wr_duration_{label}"] = float(np.mean(wrs))

    enemies = sorted(enemy_champion_ids)[:5]
    while len(enemies) < 5:
        enemies.append(0)
    for i, enemy_id in enumerate(enemies):
        advs = []
        for hero_id in team_champion_ids:
            overall = _win_rate(*stats.overall.get(hero_id, [0, 0]))
            vs = _win_rate(*stats.vs_enemy.get(hero_id, {}).get(enemy_id, [0, 0]))
            advs.append(vs - overall)
        features[f"adv_vs_enemy_{i + 1}"] = float(np.mean(advs))

    for slot in range(4):
        syn_vals = []
        for hero_id in team_champion_ids:
            allies = sorted(c for c in team_champion_ids if c != hero_id)
            if slot >= len(allies):
                syn_vals.append(0.0)
                continue
            ally_id = allies[slot]
            overall = _win_rate(*stats.overall.get(hero_id, [0, 0]))
            syn = _win_rate(*stats.with_ally.get(hero_id, {}).get(ally_id, [0, 0])) - overall
            syn_vals.append(syn)
        features[f"syn_with_ally_{slot + 1}"] = float(np.mean(syn_vals))

    return features


def _parse_champion_ids(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x]


def draft_features(row, stats: ChampionStats) -> dict[str, float]:
    """Team 100 minus team 200 for each draft feature (positive favors team 100)."""
    team100 = _parse_champion_ids(row.team100_champion_ids)
    team200 = _parse_champion_ids(row.team200_champion_ids)
    f100 = _team_draft_features(team100, team200, stats)
    f200 = _team_draft_features(team200, team100, stats)

    base_keys = (
        ["overall_win_rate"]
        + [f"wr_duration_{label}" for label in DURATION_LABELS]
        + [f"adv_vs_enemy_{i}" for i in range(1, 6)]
        + [f"syn_with_ally_{i}" for i in range(1, 5)]
    )
    suffix = {
        "overall_win_rate": "overall_win_rate_diff",
        **{f"wr_duration_{label}": f"wr_duration_{label}_diff" for label in DURATION_LABELS},
        **{f"adv_vs_enemy_{i}": f"adv_vs_enemy_{i}_diff" for i in range(1, 6)},
        **{f"syn_with_ally_{i}": f"syn_with_ally_{i}_diff" for i in range(1, 5)},
    }
    return {suffix[key]: f100[key] - f200[key] for key in base_keys}


def drafts_to_feature_frame(
    drafts: pd.DataFrame,
    stats: ChampionStats,
) -> tuple[pd.DataFrame, np.ndarray]:
    features = [draft_features(row, stats) for row in drafts.itertuples(index=False)]
    X = pd.DataFrame(features, columns=FEATURE_NAMES)
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
