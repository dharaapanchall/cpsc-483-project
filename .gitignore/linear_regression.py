"""
CPSC 483 Project - Phase 3 (individual): Linear Regression
Author: Julie Yun

Predicts an actor's billing position (1 = top billed ... 15) in a Marvel
production from their previous appearances and character history.

Input : data_processed/train_scaled.csv, data_processed/test_scaled.csv
        (produced by preprocess.py)
Output: results/linear_regression/
          metrics.csv              - test-set metrics for every model
          cv_results.csv           - 5-fold cross-validation on the training set
          coefficients.csv         - learned coefficients (standardized features)
          pred_vs_actual.png       - predicted vs. actual billing
          coefficients.png         - which features push billing up / down
          residuals.png            - residuals vs. predicted
          error_by_history.png     - error for debut vs. returning actors

Run:  python linear_regression.py
Libraries: pandas, NumPy, scikit-learn, matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_validate

DATA = Path("data_processed")
OUT = Path("results/linear_regression")
OUT.mkdir(parents=True, exist_ok=True)

# Consistent color palette across all figures
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8c8b86", "#0b0b0b", "#52514e"
ORANGE = "#eb6834"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae6",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})

# 1. Load data
# billing_tier is the classification version of billing_position, so drop it from X
# to avoid leaking the answer.
train = pd.read_csv(DATA / "train_scaled.csv")
test = pd.read_csv(DATA / "test_scaled.csv")
TARGETS = ["billing_position", "billing_tier"]
X_train, y_train = train.drop(columns=TARGETS), train["billing_position"]
X_test, y_test = test.drop(columns=TARGETS), test["billing_position"]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Models
# Baseline: always predicts mean training billing (the model needs to beat this to be useful)
# OLS: main model; Ridge: several features are strongly correlated (best/mean/last billing),
# which makes plain OLS coefficients unstable, so Ridge is included for comparison.
models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge (alpha tuned by CV)": RidgeCV(alphas=np.logspace(-3, 3, 50)),
}


def to_tier(pos):
    """Map a billing position to tier: 0 = Lead (1-3), 1 = Supporting (4-8), 2 = Minor (9-15)."""
    return np.digitize(pos, [3.5, 8.5])


# 3. 5-fold CV on training set (validation without touching the test set)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, model in models.items():
    s = cross_validate(model, X_train, y_train, cv=cv,
                       scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    cv_rows.append({
        "model": name,
        "cv_MAE_mean": -s["test_neg_mean_absolute_error"].mean(),
        "cv_MAE_std": s["test_neg_mean_absolute_error"].std(),
        "cv_RMSE_mean": -s["test_neg_root_mean_squared_error"].mean(),
        "cv_R2_mean": s["test_r2"].mean(),
        "cv_R2_std": s["test_r2"].std(),
    })
cv_df = pd.DataFrame(cv_rows).round(3)
cv_df.to_csv(OUT / "cv_results.csv", index=False)
print("\n5-fold CV on training set:\n", cv_df.to_string(index=False))

# 4. Train on full training set, evaluate on test set
# Clip predictions to 1..15 since linear regression can go out of range.
# Tier accuracy converts position to Lead/Supporting/Minor so we can compare
# with teammates' classification models.
rows, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = np.clip(model.predict(X_test), 1, 15)
    preds[name] = p
    rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        "R2": r2_score(y_test, p),
        "within_3_spots": np.mean(np.abs(y_test - p) <= 3),
        "tier_accuracy": accuracy_score(to_tier(y_test), to_tier(p)),
    })
metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(OUT / "metrics.csv", index=False)
print("\nTest-set results:\n", metrics.to_string(index=False))
print(f"\nRidge chosen alpha: {models['Ridge (alpha tuned by CV)'].alpha_:.3f}")

# 5. Coefficients (features are standardized so magnitudes are comparable)
# Negative coefficient = lower position number = billed HIGHER (closer to #1)
lr = models["Linear Regression"]
ridge = models["Ridge (alpha tuned by CV)"]
coef = pd.DataFrame({"feature": X_train.columns,
                     "linear_regression": lr.coef_,
                     "ridge": ridge.coef_}).sort_values("ridge")
coef.round(4).to_csv(OUT / "coefficients.csv", index=False)
print("\nIntercept (LR):", round(lr.intercept_, 3))
print(coef.round(3).to_string(index=False))

# 6. Figures
p_lr = preds["Linear Regression"]
m_lr = metrics.set_index("model").loc["Linear Regression"]

# 6a. Predicted vs actual
fig, ax = plt.subplots(figsize=(5.5, 5))
jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(y_test))
ax.scatter(y_test + jitter, p_lr, s=22, color=BLUE, alpha=0.55,
           edgecolor="white", linewidth=0.5)
ax.plot([1, 15], [1, 15], color=GRAY, linewidth=1.5, linestyle="--")
ax.text(14.6, 14.1, "perfect prediction", color=MUTED, ha="right", fontsize=9)
ax.set(xlabel="Actual billing position (1 = top billed)",
       ylabel="Predicted billing position", xlim=(0, 16), ylim=(0, 16),
       title="Linear regression: predicted vs. actual (test set)")
ax.text(0.5, 15.3, f"MAE {m_lr.MAE:.2f}   RMSE {m_lr.RMSE:.2f}   R² {m_lr.R2:.2f}",
        color=INK, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "pred_vs_actual.png"); plt.close(fig)

# 6b. Coefficients - Ridge version plotted; OLS distorted by multicollinearity (both in .csv)
c = coef.set_index("feature")["ridge"].sort_values()
fig, ax = plt.subplots(figsize=(7, 6.5))
colors = [BLUE if v < 0 else ORANGE for v in c.values]
ax.barh(c.index, c.values, color=colors, height=0.6)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set(xlabel="Coefficient (change in billing position per 1 std. dev.)\n"
              "Blue = billed higher (closer to #1)     Orange = billed lower",
       title="Ridge-regularized linear regression coefficients")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "coefficients.png"); plt.close(fig)

# 6c. Residuals vs predicted
res = y_test - p_lr
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(p_lr, res, s=22, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.5)
ax.axhline(0, color=GRAY, linewidth=1.5, linestyle="--")
ax.set(xlabel="Predicted billing position", ylabel="Residual (actual − predicted)",
       title="Residuals vs. predicted (test set)")
fig.tight_layout(); fig.savefig(OUT / "residuals.png"); plt.close(fig)

# 6d. Error for debut vs returning actors
grp = np.where(test["actor_is_debut"] == 1, "Debut actor", "Returning actor")
err = pd.DataFrame({"group": grp, "abs_err": np.abs(res),
                    "base_err": np.abs(y_test - preds["Baseline (mean)"])})
g = err.groupby("group")[["abs_err", "base_err"]].mean()
n = err.group.value_counts()
fig, ax = plt.subplots(figsize=(5.5, 4))
x = np.arange(len(g))
ax.bar(x - 0.18, g.base_err, width=0.34, color=GRAY, label="Baseline (mean)")
ax.bar(x + 0.18, g.abs_err, width=0.34, color=BLUE, label="Linear regression")
for i, (b, a) in enumerate(zip(g.base_err, g.abs_err)):
    ax.text(i - 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.text(i + 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, [f"{k}\n(n={n[k]})" for k in g.index])
ax.set(ylabel="Mean absolute error (billing spots)", title="Test error by actor history")
ax.set_ylim(0, g.values.max() * 1.3)
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "error_by_history.png"); plt.close(fig)
print("\nError by group:\n", g.round(3))

print(f"\nSaved results and figures to {OUT}/")
