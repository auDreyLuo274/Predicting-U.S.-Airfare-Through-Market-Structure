# ══════════════════════════════════════════════════════════════
# Beyond Distance: Predicting U.S. Airfare Through Market Structure
# Datathon 2025
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FILE_PATH = 'airline_ticket_dataset.xlsx'

# ──────────────────────────────────────────────────────────────
# PART 1
# (Team Member: )
# ──────────────────────────────────────────────────────────────

# []


# ──────────────────────────────────────────────────────────────
# PART 2
# (Team Member: )
# ──────────────────────────────────────────────────────────────

# []


# ──────────────────────────────────────────────────────────────
# PART 3: SHAP Analysis & Feature Importance
# (Team Member: Audrey Luo)
# ──────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
file_path = '/Users/luozihan/Desktop/Airline Tickets/airline_ticket_dataset.xlsx'
df = pd.read_excel(file_path)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    Layer 1: Distance + Demand (baseline)
#    Layer 2: + Competition
#    Layer 3: + Hub characteristics (Full Model)
# ─────────────────────────────────────────────
features = [
    # Layer 1 — Baseline
    'nsmiles',
    'passengers',
    # Layer 2 — Competition
    'large_ms',
    'lf_ms',
    'fare_low',
    # Layer 3 — Hub characteristics (both endpoints)
    'TotalFaredPax_city1',
    'TotalPerLFMkts_city1',
    'TotalPerPrem_city1',
    'TotalFaredPax_city2',
    'TotalPerLFMkts_city2',
    'TotalPerPrem_city2',
]

X = df[features].fillna(0)
y = df['fare']

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 4. TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. MODEL EVALUATION
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n── Model Performance on Test Data ──")
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : ${mae:.2f}")
print(f"  RMSE : ${rmse:.2f}")

# ─────────────────────────────────────────────
# 6. SHAP VALUES  (unified old API — most stable)
# ─────────────────────────────────────────────
sample_size   = int(len(X_test) * 0.8)
X_test_sample = X_test.sample(n=sample_size, random_state=42)

print(f"\nCalculating SHAP values for {sample_size} random test samples...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

# ─────────────────────────────────────────────
# 7. PLOT 1 — Summary Plot (importance + direction)
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP: Feature Importance & Direction of Impact on Airfare", fontsize=14)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → shap_summary.png")

# ─────────────────────────────────────────────
# 8. PLOT 2 — Bar Plot (clean ranking for presentation)
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title("SHAP: Mean Absolute Feature Importance Ranking", fontsize=14)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → shap_bar.png")

# ─────────────────────────────────────────────
# 9. PLOT 3 — Dependence Plot (interaction effect)
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    "large_ms",
    shap_values,
    X_test_sample,
    interaction_index="nsmiles",
    show=False,
)
plt.title("SHAP Dependence: Market Concentration × Distance\n"
          "(colour = route distance — reveals monopoly premium on short routes)", fontsize=12)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 10. EXPORT MODEL DATA FOR THE WEB TOOL
# ─────────────────────────────────────────────
feature_stats = {
    feat: {
        "min":    float(X[feat].min()),
        "max":    float(X[feat].max()),
        "mean":   float(X[feat].mean()),
        "median": float(X[feat].median()),
    }
    for feat in features
}

mean_shap = dict(zip(features, np.abs(shap_values).mean(axis=0)))
feature_stats_out = {
    "feature_stats": feature_stats,
    "mean_shap": {k: float(v) for k, v in mean_shap.items()},
    "model_r2":   round(r2, 4),
    "model_mae":  round(mae, 2),
    "model_rmse": round(rmse, 2),
}

with open("model_stats.json", "w") as f:
    json.dump(feature_stats_out, f, indent=2)

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)