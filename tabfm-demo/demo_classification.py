#!/usr/bin/env python3
"""Zero-shot wine color classification with TabFM (PyTorch).

Uses the classic UCI wine dataset: predict red vs white from chemistry
measurements. No model fine-tuning — training rows are passed as context.

Model: https://huggingface.co/google/tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

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
    X, y = load_wine_table()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training context: {len(X_train)} rows, {X_train.shape[1]} features")
    print(f"Test rows: {len(X_test)}")
    print(f"Classes: {sorted(set(y))}\n")

    print("Loading TabFM classification weights from Hugging Face...")
    model = tabfm_v1_0_0.load(model_type="classification")
    clf = TabFMClassifier(model=model)

    print("Fitting (preprocessing only — weights are not fine-tuned)...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    accuracy = (preds == y_test).mean()

    print(f"\nAccuracy on held-out test set: {accuracy:.1%}\n")
    print("Sample predictions:")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i]
        print(
            f"  alcohol={row['alcohol']:.1f}, hue={row['hue']:.2f} "
            f"→ pred={preds[i]!r} (true={y_test[i]!r})"
        )

    print("\nClass probabilities (first 3 test rows):")
    print(probs[:3])


if __name__ == "__main__":
    main()
