# Report: Breast Mass Classification with Logistic Regression

**Author:** Ziyan Chen

**Script:** `logistic_regression_analysis.py`

**Scope:** Non-clinical analysis of a public benchmark dataset. Not intended for diagnostic use.

---

## 1. Score, Sigmoid, and Threshold

Logistic regression turns a score `z = beta_0 + beta_1 * x` into a probability `p = 1 / (1 + exp(-z))` and predicts class 1 if `p >= threshold`.

**Worked calculation for Case A:**

- z = -1.0 + 0.8 × 2.0 = 0.6
- p = 1 / (1 + e^(-0.6)) = 1 / (1 + 0.5488) = 0.6457
- 0.6457 ≥ 0.50, so the predicted class is 1

| Case | beta_0 | beta_1 | x | Threshold | z | p | Class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

- **A vs. B:** Only B's threshold rises to 0.70, so the class changes from 1 to 0. The threshold changes the decision, not the probability.
- **C:** A negative coefficient reverses the curve: probability falls as x increases.
- **D:** With beta_1 = 0, x has no effect: the probability is 0.7311 for every x, so the sigmoid curve is flat.

These coefficients are set by hand; the main model **learns** them with gradient descent.

## 2. Data, Positive Class, and Leakage Prevention

**Data:** I used the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn (Wolberg, Street, and Mangasarian, 1995; UCI Machine Learning Repository). Its features describe cell nuclei from breast mass images.

I used only **mean radius** and **mean texture**, and recoded the target as **1 = malignant (positive class)** and **0 = benign (negative class)**. Basic data checks show 569 rows, no missing values, 357 benign (62.7%) and 212 malignant (37.3%).

**Data split:** I split the data into 60% training, 20% validation, and 20% test. I set aside the test set first, then divided the rest into training and validation. Both splits were stratified (`random_state=42`): 341 training, 114 validation, and 114 test rows, each about 37% malignant, with no overlap. I fit the `StandardScaler` on training rows only and reused it for validation and test, so validation and test data never influence training. The test set was used once, at the end.

## 3. Training and Coefficients

I used full-batch gradient descent (learning rate 0.1, 20,000 epochs, coefficients starting at 0, no regularization). The training loss fell from 0.693147 (= ln 2, since every probability starts at 0.5) to 0.246834 and was flat by epoch 5,000, showing convergence.

Minimizing cross-entropy is the same as maximizing the likelihood of the observed labels. The likelihood multiplies the model's probability for each true label; the log turns this into a sum, and a minus sign turns "maximize" into "minimize."

| Coefficient (standardized units) | NumPy | scikit-learn | Absolute difference |
|---|---:|---:|---:|
| beta_0 (intercept) | -0.727172 | -0.727172 | 1.31e-09 |
| beta_1 (mean radius) | 3.656794 | 3.656794 | 1.77e-09 |
| beta_2 (mean texture) | 1.009872 | 1.009872 | 1.15e-09 |

The models agree almost exactly (mean validation-probability difference: 1.19e-10) because both minimize the same unregularized loss on the same data.

**Interpretation of beta_1:** Holding mean texture fixed, a one training-standard-deviation increase in mean radius (about 3.59 units) increases the **log-odds** of malignancy by 3.66. Since log-odds = ln(odds), this multiplies the odds by e^3.66 ≈ 38.7 (e ≈ 2.718). It is not a fixed probability change, and it shows association, not causation.

## 4. Baseline vs. Logistic Regression

The majority baseline gives every sample a 37.2% malignant probability, so at 0.50 it labels everyone benign. On the test set, it reaches 63.2% accuracy but finds **0 of 42** malignant samples. Its precision is reported as 0 because it never predicts malignant (`zero_division=0`). Accuracy alone is not enough.

**Test confusion matrix (NumPy model, threshold 0.50):**

| | Predicted benign (0) | Predicted malignant (1) |
|---|---:|---:|
| **Actual benign (0)** | 68 | 4 |
| **Actual malignant (1)** | 11 | 31 |

In plain language: the model caught 31 of 42 malignant samples and missed 11; of 72 benign samples, it cleared 68 and raised 4 false alarms.

## 5. Threshold Selection

On validation data, 0.60 had the highest F1 (0.8312 vs. 0.8193 at 0.50), so I selected and froze it. On the test set, it performed worse:

| Test set | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Default (0.50) | 4 | 11 | 0.8857 | 0.7381 | 0.8052 |
| Selected (0.60) | 3 | 14 | 0.9032 | 0.6667 | 0.7671 |

The higher threshold removed one false positive but added three false negatives. I kept 0.60, because changing it after seeing test results would be test leakage. A threshold does not change the probabilities, so log loss is the same (0.2557) for both policies.

With only 43 malignant validation samples, a few cases can shift F1, so the "best" validation threshold may not hold on new data. In practice, the threshold should reflect clinical cost (for example, a minimum required recall), not F1 alone.

## 6. Why This Is Not Ready for Clinical Use

A false negative (a malignant sample labeled benign) could delay treatment, which is far more serious than a false alarm, and the model still misses 11 of 42 malignant test samples. Other limitations:

- **Limited features:** only 2 of 30 available measurements.
- **Small, historical sample:** 569 samples from one 1990s source may not represent today's patients.
- **Unverified calibration:** a predicted 70% risk may not mean 70% of such cases are malignant.
- **No external validation:** not tested on other hospitals or time periods.

---

## Research Process

**1. Duplicate imports found in a fresh clone.** After pushing to GitHub, I reviewed a fresh clone of the repository and found that `LogisticRegression` and `train_test_split` were imported twice, because I had added code in stages. I removed the duplicates, reran the script, and confirmed that both CSV files were unchanged.

**2. Were 20,000 epochs too many?** I questioned whether the assigned epoch count was too large. I checked the loss at epochs 1,000, 5,000, and 10,000. The loss reached 0.246834 by epoch 5,000 and stayed the same, the change over the last 1,000 epochs was about 0, and the final gradient size was about 3e-15. Extra epochs did not change the result, so I kept the assigned settings.

**3. The selected threshold performed worse on test.** I expected 0.60 to also do better on the test set, but test F1 dropped. I looked at the validation results: the F1 gap between 0.50 and 0.60 came from only a few cases (4 fewer false positives, 2 more false negatives) among 43 malignant samples. A difference this small may not repeat on new data. I kept the frozen threshold and reported the result honestly.

**4. Design checks.** I confirmed that the `predict_proba` column for class 1 matches `classes_ = [0, 1]`, that the three partitions do not overlap, and that my NumPy coefficients match scikit-learn within 2e-09.

**5. Why a coarse threshold grid?** I considered whether finer thresholds (such as 0.25, 0.35, 0.45, …) would find a better cut-off than the required grid of 0.20 to 0.80 in steps of 0.10. Checking validation data only, F1 moved up and down between nearby thresholds, because one or two samples can change it. A finer grid would make it easier to pick a threshold that only fits this small validation set.

## Sources and Acknowledgements

- Wolberg, W. H., Street, W. N., and Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository. Loaded through `sklearn.datasets.load_breast_cancer()`.
- scikit-learn documentation for `LogisticRegression`, `DummyClassifier`, `StandardScaler`, `train_test_split`, and `sklearn.metrics`.
- MSBA 265 course materials: Classification Foundations lecture and Logistic Regression lecture.
