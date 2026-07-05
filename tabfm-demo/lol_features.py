"""Build TabFM feature rows for LoL draft/match outcome prediction."""

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

FEATURE_NAMES: list[str] = [
    "overall_win_rate",
    *[f"wr_duration_{label}" for label in DURATION_LABELS],
    *[f"adv_vs_enemy_{i}" for i in range(1, 6)],
    *[f"syn_with_ally_{i}" for i in range(1, 5)],
]


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


def _sorted_ids(raw: str, count: int) -> list[int]:
    ids = sorted(int(x) for x in raw.split(",") if x)
    if len(ids) < count:
        ids = ids + [0] * (count - len(ids))
    return ids[:count]


def row_features(row, stats: ChampionStats) -> dict[str, float]:
    hero_id = int(row.champion_id)
    overall_wr = _win_rate(*stats.overall.get(hero_id, [0, 0]))

    duration_stats = stats.by_duration.get(hero_id, {})
    duration_features = {
        f"wr_duration_{label}": _win_rate(*duration_stats.get(idx, [0, 0]))
        for idx, label in enumerate(DURATION_LABELS)
    }

    enemy_ids = _sorted_ids(row.enemy_champion_ids, 5)
    vs_enemy = stats.vs_enemy.get(hero_id, {})
    enemy_features = {
        f"adv_vs_enemy_{i + 1}": _win_rate(*vs_enemy.get(enemy_id, [0, 0])) - overall_wr
        for i, enemy_id in enumerate(enemy_ids)
    }

    ally_ids = _sorted_ids(row.ally_champion_ids, 4)
    with_ally = stats.with_ally.get(hero_id, {})
    ally_features = {
        f"syn_with_ally_{i + 1}": _win_rate(*with_ally.get(ally_id, [0, 0])) - overall_wr
        for i, ally_id in enumerate(ally_ids)
    }

    return {
        "overall_win_rate": overall_wr,
        **duration_features,
        **enemy_features,
        **ally_features,
    }


def participants_to_feature_frame(
    participants: pd.DataFrame,
    stats: ChampionStats,
) -> tuple[pd.DataFrame, np.ndarray]:
    features = [row_features(row, stats) for row in participants.itertuples(index=False)]
    X = pd.DataFrame(features, columns=FEATURE_NAMES)
    y = participants["match_result"].to_numpy()
    return X, y


def train_test_split_by_match(
    participants: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_ids = np.array(participants["match_id"].unique())
    rng = np.random.default_rng(random_state)
    rng.shuffle(match_ids)
    split = int(len(match_ids) * (1 - test_size))
    train_ids = set(match_ids[:split])
    train = participants[participants["match_id"].isin(train_ids)].reset_index(drop=True)
    test = participants[~participants["match_id"].isin(train_ids)].reset_index(drop=True)
    return train, test
