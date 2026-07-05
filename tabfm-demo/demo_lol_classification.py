#!/usr/bin/env python3
"""Zero-shot LoL draft winner prediction with TabFM (PyTorch).

One row per complete 5v5 draft. Given both teams' champion picks, predict
which team won (TEAM_100 or TEAM_200).

Features (24 numeric columns, team 100 minus team 200):
  1     mean overall win rate across the 5 picks
  2–15  mean win rate in each game-duration bucket
  16–20 mean counter edge vs each of the 5 enemy champions
  21–24 mean synergy with each of the 4 ally slots

Model: https://huggingface.co/google/tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from extract_lol_features import DEFAULT_OUTPUT, fetch_participants
from lol_features import (
    FEATURE_NAMES,
    build_champion_stats,
    drafts_to_feature_frame,
    participants_to_drafts,
    train_test_split_by_match,
)
from tabfm import TabFMClassifier
from tabfm_demo_common import (
    configure_logging,
    load_tabfm_model,
    log,
    log_torch_device,
    predownload_checkpoint,
    resolve_device,
    setup_hf_download,
)

MAX_TRAIN_ROWS = 800
MAX_TEST_ROWS = 2000
PREDICT_BATCH_SIZE = 8
TABFM_N_ESTIMATORS = 1


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def log_class_counts(name: str, labels: np.ndarray) -> None:
    classes, counts = np.unique(labels, return_counts=True)
    log(f"{name} class counts: {dict(zip(classes, [int(c) for c in counts]))}")


def load_participants() -> pd.DataFrame:
    csv_path = Path(os.getenv("LOL_PARTICIPANTS_CSV", DEFAULT_OUTPUT))
    if csv_path.is_file():
        log(f"Loading participants from {csv_path}")
        return pd.read_csv(csv_path)

    if os.getenv("POSTGRES_HOST") and os.getenv("POSTGRES_PASSWORD"):
        log("CSV not found — fetching participants from Postgres...")
        from extract_lol_features import db_connect

        with db_connect() as conn:
            return fetch_participants(conn)

    raise FileNotFoundError(
        f"No data at {csv_path} and Postgres env vars are not set. "
        "Run extract_lol_features.py first or set POSTGRES_* / LOL_PARTICIPANTS_CSV."
    )


def cap_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    max_rows: int,
    label: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if len(X) <= max_rows:
        return X, y, meta
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    log(f"Capping {label} from {len(X)} to {max_rows} rows")
    return X.iloc[idx].reset_index(drop=True), y[idx], meta.iloc[idx].reset_index(drop=True)


def predict_batched(
    clf: TabFMClassifier,
    X_test: pd.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    prob_chunks: list[np.ndarray] = []
    n = len(X_test)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X_test.iloc[start:end]
        prob_chunks.append(clf.predict_proba(chunk))
        log(f"  batch {start // batch_size + 1}/{(n + batch_size - 1) // batch_size}: rows {start}-{end - 1}")
    probs = np.vstack(prob_chunks)
    class_idx = np.argmax(probs[:, : clf.n_classes_], axis=1)
    preds = clf.y_encoder_.inverse_transform(class_idx.reshape(-1, 1)).flatten()
    return preds.astype(clf.classes_.dtype), probs


def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (
        Path.cwd() / ".env",
        repo_root / "db details" / ".env",
        repo_root / ".env",
    ):
        if path.is_file():
            load_dotenv(path)
            return
    load_dotenv()


def main() -> int:
    load_env()
    configure_logging()
    log("=== TabFM LoL draft winner demo ===")
    log("Task: given full 5v5 draft → predict TEAM_100 or TEAM_200 won")
    log_torch_device()
    device = resolve_device()
    log(f"Target device for inference: {device}")
    log("")

    token = setup_hf_download()
    log("")

    participants = load_participants()
    all_drafts = participants_to_drafts(participants)
    log(f"Complete 5v5 drafts: {len(all_drafts)} matches")

    train_drafts, test_drafts = train_test_split_by_match(participants)
    train_match_ids = set(train_drafts["match_id"])
    train_participants = participants[participants["match_id"].isin(train_match_ids)]
    stats = build_champion_stats(train_participants)

    max_train = env_int("MAX_TRAIN_ROWS", MAX_TRAIN_ROWS)
    max_test = env_int("MAX_TEST_ROWS", MAX_TEST_ROWS)
    batch_size = env_int("PREDICT_BATCH_SIZE", PREDICT_BATCH_SIZE)
    n_estimators = env_int("TABFM_N_ESTIMATORS", TABFM_N_ESTIMATORS)

    X_train, y_train = drafts_to_feature_frame(train_drafts, stats)
    X_test, y_test = drafts_to_feature_frame(test_drafts, stats)
    X_train, y_train, train_drafts = cap_rows(X_train, y_train, train_drafts, max_train, "training context")
    X_test, y_test, test_drafts = cap_rows(X_test, y_test, test_drafts, max_test, "test set")

    log(f"Training context: {len(X_train)} drafts, {len(FEATURE_NAMES)} features")
    log(f"Test drafts: {len(X_test)}")
    log_class_counts("Train", y_train)
    log_class_counts("Test", y_test)
    log("")

    checkpoint_dir = predownload_checkpoint("classification", token)
    model = load_tabfm_model("classification", device, checkpoint_dir)

    log("Wrapping model in TabFMClassifier...")
    clf = TabFMClassifier(model=model, n_estimators=n_estimators, batch_size=1)
    log(f"  n_estimators={n_estimators}, predict_batch_size={batch_size}")
    log("")

    log("Fitting (preprocessing only — foundation weights are NOT fine-tuned)...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    log(f"  fit complete in {time.time() - t0:.1f}s")
    log("")

    log(f"Predicting winners for {len(X_test)} held-out drafts (batch size {batch_size})...")
    t0 = time.time()
    preds, probs = predict_batched(clf, X_test, batch_size)
    log(f"  predict complete in {time.time() - t0:.1f}s")
    log_torch_device(model)
    log("")

    accuracy = (preds == y_test).mean()
    log(f"Draft winner accuracy: {accuracy:.1%}")
    log(f"Correct: {(preds == y_test).sum()} / {len(y_test)}")
    log("")

    log("Sample predictions:")
    for i in range(min(5, len(X_test))):
        row = test_drafts.iloc[i]
        log(
            f"  [{i}] {row['match_id']}\n"
            f"      team100: {row['team100_champion_names']}\n"
            f"      team200: {row['team200_champion_names']}\n"
            f"      overall_wr_diff={X_test.iloc[i]['overall_win_rate_diff']:.3f} "
            f"→ pred={preds[i]!r} (true={y_test[i]!r})"
        )

    log("")
    log("Class probabilities (first 3 test drafts):")
    for i in range(min(3, len(probs))):
        log(f"  row {i}: {probs[i]}")
    log("")
    log("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
