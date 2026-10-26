from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
FEATURES = ["mean radius", "mean texture"]
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Part 1: Score -> probability -> class
# ----------------------------------------------------------------------
def sigmoid(z):
    """Stable sigmoid: same as 1 / (1 + exp(-z)) but never overflows."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


def part1_cases() -> None:
    cases = [
        ("A", -1.0, 0.8, 2.0, 0.50),
        ("B", -1.0, 0.8, 2.0, 0.70),
        ("C", -1.0, -0.8, 2.0, 0.50),
        ("D", 1.0, 0.0, 2.0, 0.50),
    ]
    print("Part 1: score, probability, and class")
    print(f"{'Case':<6}{'beta_0':>8}{'beta_1':>8}{'x':>6}{'threshold':>11}{'z':>8}{'p':>9}{'class':>7}")
    for name, b0, b1, x, threshold in cases:
        z = b0 + b1 * x
        p = float(sigmoid(z))
        label = int(p >= threshold)
        print(f"{name:<6}{b0:8.1f}{b1:8.1f}{x:6.1f}{threshold:11.2f}{z:8.2f}{p:9.4f}{label:7d}")
    print()


# ----------------------------------------------------------------------
# Part 2: Leakage-free data preparation
# ----------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    columns = [list(data.feature_names).index(name) for name in FEATURES]
    X = data.data[:, columns]
    y = (data.target == 0).astype(int)  # 1 = malignant (positive), 0 = benign (negative)

    print("Part 2: data")
    print(f"  Features (in order): {FEATURES}")
    print("  Target coding: 1 = malignant (positive class), 0 = benign (negative class)")
    print(f"  Rows: {X.shape[0]}")
    print(f"  Missing values per feature: {dict(zip(FEATURES, np.isnan(X).sum(axis=0).tolist()))}")
    print(f"  Class counts: benign (0) = {np.sum(y == 0)}, malignant (1) = {np.sum(y == 1)}")
    print()
    return X, y


def split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=0.25, stratify=y[dev_idx], random_state=RANDOM_STATE
    )

    print("  Partition sizes and class counts:")
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        print(f"    {name:<11}{len(idx):4d} rows | benign (0) = {np.sum(y[idx] == 0):3d} | malignant (1) = {np.sum(y[idx] == 1):3d}")

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    no_overlap = not (train_set & val_set or train_set & test_set or val_set & test_set)
    full_cover = len(train_set | val_set | test_set) == len(y)
    print(f"  No overlap between partitions: {no_overlap}")
    print(f"  Partitions cover every row:    {full_cover}")
    print()
    return train_idx, val_idx, test_idx


def scale_features(X, train_idx, val_idx, test_idx):
    scaler = StandardScaler().fit(X[train_idx])  # fit on training rows only
    print(f"  Scaler fitted on training rows only: mean = {scaler.mean_.round(4)}, std = {scaler.scale_.round(4)}")
    print()
    return scaler.transform(X[train_idx]), scaler.transform(X[val_idx]), scaler.transform(X[test_idx])


def main() -> None:
    part1_cases()

    X, y = load_data()
    train_idx, val_idx, test_idx = split_indices(y)
    X_train, X_val, X_test = scale_features(X, train_idx, val_idx, test_idx)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]


if __name__ == "__main__":
    main()