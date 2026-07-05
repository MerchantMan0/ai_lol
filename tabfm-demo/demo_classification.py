#!/usr/bin/env python3
"""Zero-shot wine color classification with TabFM (PyTorch).

Uses the classic UCI wine dataset: predict red vs white from chemistry
measurements. No model fine-tuning — training rows are passed as context.

Model: https://huggingface.co/google/tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def log_torch_device() -> None:
    try:
        import torch
    except ImportError:
        log("PyTorch: not installed yet")
        return

    log(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    else:
        log("CUDA: not available (model will run on CPU — slower)")


def log_class_counts(name: str, labels: np.ndarray) -> None:
    classes, counts = np.unique(labels, return_counts=True)
    log(f"{name} class counts: {dict(zip(classes, counts))}")


from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0


FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

CLASS_NAMES = {
    0: "cultivar_1",
    1: "cultivar_2",
    2: "cultivar_3",
}


def load_wine_table() -> tuple[pd.DataFrame, np.ndarray]:
    wine = load_wine(as_frame=True)
    X = wine.data.rename(columns=dict(zip(wine.feature_names, FEATURE_NAMES)))
    y = wine.target.map(CLASS_NAMES).to_numpy()
    return X, y


def main() -> None:
    log("=== TabFM classification demo (wine cultivars) ===")
    log_torch_device()
    log("")

    log("Loading dataset...")
    X, y = load_wine_table()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    log(f"Training context: {len(X_train)} rows, {X_train.shape[1]} features")
    log(f"Test rows: {len(X_test)}")
    log(f"Classes: {sorted(set(y))}")
    log_class_counts("Train", y_train)
    log_class_counts("Test", y_test)
    log("")

    log("Downloading/loading TabFM classification weights from Hugging Face...")
    log("  repo: google/tabfm-1.0.0-pytorch (subfolder: classification)")
    log("  first run can take several minutes — download + weight load")
    t0 = time.time()
    model = tabfm_v1_0_0.load(model_type="classification")
    log(f"  weights loaded in {time.time() - t0:.1f}s")
    log_torch_device()

    log("Wrapping model in TabFMClassifier...")
    clf = TabFMClassifier(model=model)
    log("")

    log("Fitting (preprocessing only — foundation weights are NOT fine-tuned)...")
    log("  encoding categoricals, scaling numerics, storing training rows as context")
    t0 = time.time()
    clf.fit(X_train, y_train)
    log(f"  fit complete in {time.time() - t0:.1f}s")
    log("")

    log(f"Predicting on {len(X_test)} held-out test rows...")
    t0 = time.time()
    preds = clf.predict(X_test)
    log(f"  predict complete in {time.time() - t0:.1f}s")

    log("Computing class probabilities...")
    t0 = time.time()
    probs = clf.predict_proba(X_test)
    log(f"  predict_proba complete in {time.time() - t0:.1f}s")
    log_torch_device()
    log("")

    accuracy = (preds == y_test).mean()
    log(f"Accuracy on held-out test set: {accuracy:.1%}")
    log(f"Correct: {(preds == y_test).sum()} / {len(y_test)}")
    log("")

    log("Sample predictions:")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i]
        log(
            f"  [{i}] alcohol={row['alcohol']:.1f}, hue={row['hue']:.2f} "
            f"→ pred={preds[i]!r} (true={y_test[i]!r})"
        )

    log("")
    log("Class probabilities (first 3 test rows):")
    for i in range(min(3, len(probs))):
        log(f"  row {i}: {probs[i]}")
    log("")
    log("=== Done ===")


if __name__ == "__main__":
    main()
