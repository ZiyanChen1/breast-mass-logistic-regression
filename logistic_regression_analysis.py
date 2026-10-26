from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"
FEATURES = ["mean radius", "mean texture"]
RANDOM_STATE = 42
LEARNING_RATE = 0.1
EPOCHS = 20000

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


# ----------------------------------------------------------------------
# Part 3: NumPy logistic regression (full-batch gradient descent)
# ----------------------------------------------------------------------
def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a leading column of ones so beta_0 is learned like the other coefficients."""
    return np.column_stack([np.ones(len(X)), X])


def bce_loss(X_aug: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Stable mean binary cross-entropy (log loss); never takes log(0)."""
    z = X_aug @ beta
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_numpy_logistic(X_train, y_train, learning_rate: float, epochs: int):
    X_aug = add_intercept(X_train)
    n = len(y_train)
    beta = np.zeros(X_aug.shape[1])              # beta_0, beta_1, beta_2 all start at 0
    losses = [bce_loss(X_aug, y_train, beta)]    # loss before training

    for _ in range(epochs):
        p = sigmoid(X_aug @ beta)                # predicted probability of malignant
        gradient = X_aug.T @ (p - y_train) / n   # gradient of mean cross-entropy
        beta = beta - learning_rate * gradient   # update all three coefficients together
        losses.append(bce_loss(X_aug, y_train, beta))

    final_gradient = X_aug.T @ (sigmoid(X_aug @ beta) - y_train) / n
    return beta, np.array(losses), float(np.linalg.norm(final_gradient))


def predict_proba_numpy(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(X) @ beta)


def report_training(beta, losses, gradient_norm) -> None:
    print("Part 3: NumPy logistic regression")
    print(f"  Settings: learning_rate = {LEARNING_RATE}, epochs = {EPOCHS}, initial coefficients = 0, no regularization")
    print(f"  Initial training loss: {losses[0]:.6f}")
    for epoch in [1000, 5000, 10000]:
        print(f"  Loss after epoch {epoch:5d}: {losses[epoch]:.6f}")
    print(f"  Final training loss:   {losses[-1]:.6f}")
    print(f"  Loss change over last 1,000 epochs: {losses[-1001] - losses[-1]:.2e}")
    print(f"  Final gradient size (norm):         {gradient_norm:.2e}")
    print("  Learned coefficients (standardized units: per 1 training standard deviation):")
    for name, value in zip(["beta_0 (intercept)", "beta_1 (mean radius)", "beta_2 (mean texture)"], beta):
        print(f"    {name:<23}{value:10.6f}")
    print()


def save_loss_plot(losses: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(losses)), losses, linewidth=2)
    plt.title("NumPy Logistic Regression: Training Loss by Epoch")
    plt.xlabel("Epoch (0 = before training)")
    plt.ylabel("Mean binary cross-entropy (training)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved loss plot to: {path}")
    print()


# ----------------------------------------------------------------------
# Part 4: scikit-learn reference model and majority baseline
# ----------------------------------------------------------------------
def fit_reference_models(X_train, y_train):
    sk_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000, tol=1e-9)
    sk_model.fit(X_train, y_train)          # fits its own intercept (no column of ones)
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)          # learns training class proportions only
    return sk_model, baseline


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    """Return P(class 1 = malignant), checking which predict_proba column is class 1."""
    column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, column]


def report_comparison(beta, sk_model, baseline, X_val) -> None:
    sk_beta = np.concatenate([sk_model.intercept_, sk_model.coef_[0]])
    print("Part 4: compare NumPy, scikit-learn, and baseline")
    print(f"  scikit-learn classes_ = {sk_model.classes_.tolist()} -> using column {list(sk_model.classes_).index(1)} for class 1 (malignant)")
    print(f"  {'Coefficient':<23}{'NumPy':>12}{'scikit-learn':>14}{'Abs diff':>12}")
    for name, a, b in zip(["beta_0 (intercept)", "beta_1 (mean radius)", "beta_2 (mean texture)"], beta, sk_beta):
        print(f"  {name:<23}{a:12.6f}{b:14.6f}{abs(a - b):12.2e}")

    p_numpy = predict_proba_numpy(X_val, beta)
    p_sklearn = predict_proba_positive(sk_model, X_val)
    print(f"  Mean absolute difference in validation probabilities: {np.mean(np.abs(p_numpy - p_sklearn)):.2e}")

    p_baseline = predict_proba_positive(baseline, X_val)[0]
    print(f"  Baseline predicts P(malignant) = {p_baseline:.6f} for every sample (training share of malignant)")
    print("  At threshold 0.50 the baseline therefore labels every sample benign (0).")
    print()


def main() -> None:
    part1_cases()

    X, y = load_data()
    train_idx, val_idx, test_idx = split_indices(y)
    X_train, X_val, X_test = scale_features(X, train_idx, val_idx, test_idx)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    beta, losses, gradient_norm = fit_numpy_logistic(X_train, y_train, LEARNING_RATE, EPOCHS)
    report_training(beta, losses, gradient_norm)
    save_loss_plot(losses, REPORT_DIR / "training_loss.png")

    sk_model, baseline = fit_reference_models(X_train, y_train)
    report_comparison(beta, sk_model, baseline, X_val)
    
if __name__ == "__main__":
    main()