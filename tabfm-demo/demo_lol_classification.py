#!/usr/bin/env python3
"""Zero-shot LoL match outcome classification with TabFM (PyTorch).

Each row is one hero in one ranked match. Features (24 numeric columns):
  1     overall win rate
  2–15  win rate by game-duration bucket (14 intervals)
  16–20 relative advantage vs each of the 5 enemy champions
  21–24 relative synergy with each of the 4 allied champions

Target: WIN or LOSS for that participant.

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
    participants_to_feature_frame,
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


def cap_training_rows(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    max_rows: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X_train) <= max_rows:
        return X_train, y_train
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=max_rows, replace=False)
    log(f"Capping training context from {len(X_train)} to {max_rows} rows (TabFM memory limit)")
    return X_train.iloc[idx].reset_index(drop=True), y_train[idx]


def cap_test_rows(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_participants: pd.DataFrame,
    max_rows: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if len(X_test) <= max_rows:
        return X_test, y_test, test_participants
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_test), size=max_rows, replace=False)
    log(f"Capping test evaluation from {len(X_test)} to {max_rows} rows")
    return (
        X_test.iloc[idx].reset_index(drop=True),
        y_test[idx],
        test_participants.iloc[idx].reset_index(drop=True),
    )


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
    log("=== TabFM LoL classification demo ===")
    log_torch_device()
    device = resolve_device()
    log(f"Target device for inference: {device}")
    log("")

    token = setup_hf_download()
    log("")

    participants = load_participants()
    train_participants, test_participants = train_test_split_by_match(participants)
    stats = build_champion_stats(train_participants)

    max_train = env_int("MAX_TRAIN_ROWS", MAX_TRAIN_ROWS)
    max_test = env_int("MAX_TEST_ROWS", MAX_TEST_ROWS)
    batch_size = env_int("PREDICT_BATCH_SIZE", PREDICT_BATCH_SIZE)
    n_estimators = env_int("TABFM_N_ESTIMATORS", TABFM_N_ESTIMATORS)

    X_train, y_train = participants_to_feature_frame(train_participants, stats)
    X_test, y_test = participants_to_feature_frame(test_participants, stats)
    X_train, y_train = cap_training_rows(X_train, y_train, max_train)
    X_test, y_test, test_participants = cap_test_rows(
        X_test, y_test, test_participants, max_test
    )

    log(f"Training context: {len(X_train)} rows, {len(FEATURE_NAMES)} features")
    log(f"Test rows: {len(X_test)}")
    log(f"Train matches: {train_participants['match_id'].nunique()}")
    log(f"Test matches: {test_participants['match_id'].nunique()}")
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
    log("  stats computed from training matches; no weight updates")
    t0 = time.time()
    clf.fit(X_train, y_train)
    log(f"  fit complete in {time.time() - t0:.1f}s")
    log("")

    log(f"Predicting on {len(X_test)} held-out rows (batch size {batch_size})...")
    t0 = time.time()
    preds, probs = predict_batched(clf, X_test, batch_size)
    log(f"  predict complete in {time.time() - t0:.1f}s")
    log_torch_device(model)
    log("")

    accuracy = (preds == y_test).mean()
    log(f"Accuracy on held-out test set: {accuracy:.1%}")
    log(f"Correct: {(preds == y_test).sum()} / {len(y_test)}")
    log("")

    log("Sample predictions:")
    for i in range(min(5, len(X_test))):
        champ = test_participants.iloc[i]["champion_name"]
        log(
            f"  [{i}] {champ} overall_wr={X_test.iloc[i]['overall_win_rate']:.3f} "
            f"→ pred={preds[i]!r} (true={y_test[i]!r})"
        )

    log("")
    log("Class probabilities (first 3 test rows):")
    for i in range(min(3, len(probs))):
        log(f"  row {i}: {probs[i]}")
    log("")
    log("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
