#!/usr/bin/env python3
"""Zero-shot house price regression with TabFM (PyTorch).

Uses California housing: predict median home value from census features.
No model fine-tuning — training rows are passed as context.

Model: https://huggingface.co/google/tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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
    X, y = load_housing_table()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training context: {len(X_train)} rows, {X_train.shape[1]} features")
    print(f"Test rows: {len(X_test)}")
    print(f"Target range: ${y.min():.0f}k – ${y.max():.0f}k (×100k)\n")

    print("Loading TabFM regression weights from Hugging Face...")
    model = tabfm_v1_0_0.load(model_type="regression")
    reg = TabFMRegressor(model=model)

    print("Fitting (preprocessing only — weights are not fine-tuned)...")
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\nMAE:  ${mae:.2f}k (×100k)")
    print(f"R²:   {r2:.3f}\n")
    print("Sample predictions:")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i]
        print(
            f"  MedInc={row['MedInc']:.2f}, AveRooms={row['AveRooms']:.1f} "
            f"→ pred=${preds[i]:.2f}k, true=${y_test[i]:.2f}k"
        )


if __name__ == "__main__":
    main()
