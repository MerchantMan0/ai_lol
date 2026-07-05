#!/usr/bin/env python3
"""Zero-shot house price regression with TabFM (PyTorch).

Uses California housing: predict median home value from census features.
No model fine-tuning — training rows are passed as context.

Model: https://huggingface.co/google/tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, r2_score
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


from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0


def load_housing_table(sample_size: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop(columns=["MedHouseVal"])
    y = housing.frame["MedHouseVal"].to_numpy()

    # TabFM passes all training rows as context; cap size for memory.
    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    return X, y


def main() -> None:
    log("=== TabFM regression demo (California housing) ===")
    log_torch_device()
    log("")

    log("Loading dataset...")
    X, y = load_housing_table()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log(f"Training context: {len(X_train)} rows, {X_train.shape[1]} features")
    log(f"Test rows: {len(X_test)}")
    log(f"Target range: ${y.min():.0f}k – ${y.max():.0f}k (×100k)")
    log("")

    log("Downloading/loading TabFM regression weights from Hugging Face...")
    log("  repo: google/tabfm-1.0.0-pytorch (subfolder: regression)")
    log("  first run can take several minutes — download + weight load")
    t0 = time.time()
    model = tabfm_v1_0_0.load(model_type="regression")
    log(f"  weights loaded in {time.time() - t0:.1f}s")
    log_torch_device()

    log("Wrapping model in TabFMRegressor...")
    reg = TabFMRegressor(model=model)
    log("")

    log("Fitting (preprocessing only — foundation weights are NOT fine-tuned)...")
    t0 = time.time()
    reg.fit(X_train, y_train)
    log(f"  fit complete in {time.time() - t0:.1f}s")
    log("")

    log(f"Predicting on {len(X_test)} held-out test rows...")
    t0 = time.time()
    preds = reg.predict(X_test)
    log(f"  predict complete in {time.time() - t0:.1f}s")
    log_torch_device()
    log("")

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    log(f"MAE:  ${mae:.2f}k (×100k)")
    log(f"R²:   {r2:.3f}")
    log("")
    log("Sample predictions:")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i]
        log(
            f"  [{i}] MedInc={row['MedInc']:.2f}, AveRooms={row['AveRooms']:.1f} "
            f"→ pred=${preds[i]:.2f}k, true=${y_test[i]:.2f}k"
        )
    log("")
    log("=== Done ===")


if __name__ == "__main__":
    main()
