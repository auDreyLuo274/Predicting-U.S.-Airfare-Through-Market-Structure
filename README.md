# Predicting U.S. Airfare Through Market Structure
**Datathon 2025**

**[→ Live Demo: FareCheck Tool](https://airfare-predict.onrender.com/airfare_tool.html)**

---

## Overview

This project analyzes and predicts U.S. domestic airfare using the U.S. DOT Consumer Airfare Report (2021–2025 Q2). We use a hierarchical modeling approach — layering distance, competition, and hub characteristics — to isolate *why* fares deviate from what distance alone would predict. Beyond prediction accuracy, the goal is to surface structural drivers that determine fare variation across markets and challenge common assumptions about what makes a route expensive.

---

## Research Questions

1. Can route-level fares be accurately predicted using distance, demand, competition, and endpoint hub characteristics?
2. Which features contribute most to explaining fare variation across markets?
3. How much of a route's fare is *structural premium* — the portion above a distance-only cost baseline — and what drives it?

---

## Key Findings

### Model Performance

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Random Forest (Part 2, tuned) | 0.884 | — | — |
| **XGBoost — Full Model (Part 3)** | **0.9414** | **$11.39** | **$15.56** |

The XGBoost model outperforms the Random Forest baseline despite using only 11 structural features (no city or carrier identity dummies), demonstrating that market structure alone explains over 94% of fare variation.

---

### Finding 1 — Large Airports Are Actually Cheaper (Myth-Busted)

> **Part 1 (EDA, Carina Zhang)**

A common assumption is that flying through major hub airports is more expensive. The data shows the opposite:

- **Overall mean carrier concentration** (`large_ms`): **0.566**
- **Mean carrier concentration on major-hub routes** (routes between NYC, LA, SF, DC, Chicago, DFW, Atlanta, Miami, Denver): **0.433**

Routes connecting major hubs have *lower* carrier concentration than average — meaning more airlines competing for the same passengers. This competitive pressure is what keeps hub fares from rising. The *illusion* of expensive hub airports often comes from comparing premium-cabin average fares, not economy.

---

### Finding 2 — Short-Haul Routes Are the Most Expensive Per Mile (Myth-Busted)

> **Part 1 (EDA, Carina Zhang)**

It is often assumed that longer flights cost more. Per-mile, the reverse is true:

| Route Type | Distance Threshold | Avg. Fare per Mile |
|---|---|---|
| Short-haul | < 936 miles (median) | **$0.392 / mi** |
| Long-haul | ≥ 936 miles (median) | **$0.185 / mi** |

Short-haul passengers pay **2.1× more per mile** than long-haul passengers. Fixed per-flight costs — airport fees, minimum crew hours, gate time — are distributed over fewer miles on short routes, driving up per-mile fares regardless of competition.

---

### Finding 3 — Monopoly Premium Is Strongest on Short Routes

> **Part 3 (SHAP Analysis, Audrey Luo)**

SHAP dependence analysis of `large_ms` (dominant carrier market share) with `nsmiles` as the interaction index reveals that carrier concentration has the strongest upward effect on fares on **short-haul routes**. On longer routes, the monopoly premium is partially offset by the presence of more competing airlines drawn to high-volume transcontinental corridors.

---

### Finding 4 — Market Price Floor Is Downstream of Market Structure

> **Part 3 (Ablation Study, Audrey Luo)**

An ablation study trained an otherwise-identical XGBoost model with `fare_low` (the average fare paid by the most budget-conscious passengers — a proxy for the competitive price floor) removed. Model performance showed no meaningful degradation in R², MAE, or RMSE.

**Interpretation:** `fare_low` encodes no independent information beyond what the structural features already provide. The price floor on a route is itself determined by market structure — carrier concentration, LCC presence, hub scale — and removing it simply shifts importance back to those upstream drivers.

---

### Finding 5 — Feature Importance Ranking (SHAP)

> **Part 3 (Audrey Luo)**

Mean absolute SHAP values from the full XGBoost model:

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `fare_low` | 24.85 | Market price floor (downstream of competition) |
| 2 | `nsmiles` | 23.07 | Route distance drives base cost |
| 3 | `lf_ms` | 10.21 | Low-cost carrier presence suppresses fares |
| 4 | `TotalPerPrem_city1` | 7.82 | Origin hub premium-cabin concentration |
| 5 | `TotalPerPrem_city2` | 5.88 | Destination hub premium-cabin concentration |
| 6 | `passengers` | 5.64 | Route demand attracts competition |
| 7–11 | Hub scale, LCC routes | < 3.0 | Airport-level structural context |

`large_ms` ranks lower in mean |SHAP| than `lf_ms`, suggesting that **the presence of budget carriers** is a more consistent suppressor of fares than the absence of a dominant legacy carrier.

---

## Modeling Approach

### Part 1 — Exploratory Analysis (Carina Zhang)
Scatter plots of fare vs. distance, demand (log scale), market share, and price spread across segments. Linear regression models benchmarking each feature group in isolation, producing interpretable coefficient-level evidence before moving to ensemble methods.

### Part 2 — Random Forest Baseline (Edwin Zeng)
`RandomForestRegressor` with full categorical encoding of city and carrier identity via one-hot encoding. Hyperparameter tuning via `RandomizedSearchCV` (80 iterations, 5-fold CV). Permutation importance computed across all features including city/carrier dummies. OOB R²: **0.889**, Test R²: **0.884**.

### Part 3 — XGBoost + SHAP + Ablation (Audrey Luo)

Features organized in three layers to isolate each group's contribution:

| Layer | Features | Purpose |
|---|---|---|
| **1 — Baseline** | `nsmiles`, `passengers` | Distance + demand |
| **2 — Competition** | `large_ms`, `lf_ms`, `fare_low` | Carrier concentration + LCC presence |
| **3 — Full Model** | `TotalFaredPax`, `TotalPerLFMkts`, `TotalPerPrem` (both endpoints) | Hub characteristics |

**XGBoost hyperparameters:** 300 estimators, max depth 6, learning rate 0.05, subsample 0.8, colsample_bytree 0.8.

**SHAP analysis:** TreeExplainer used to produce summary plots, bar importance rankings, and a `large_ms × nsmiles` dependence plot revealing the interaction between monopoly power and route distance.

**Ablation study:** Identical model trained without `fare_low` to quantify its independent contribution — confirming it adds no signal beyond structural features.

---

## Repository Structure

```
├── main.ipynb              # Full analysis pipeline (Parts 1–3)
├── ablation_study.ipynb    # Standalone ablation study notebook
├── app.py                  # Flask backend — REST API for fare predictions
├── airfare_tool.html       # Consumer-facing web tool (FareCheck)
├── model.pkl               # Trained XGBoost model
├── model_stats.json        # Feature statistics + SHAP importance values
├── city_lookup.json        # City-level feature medians for inference
└── requirements.txt        # Python dependencies
```

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the prediction backend
python app.py

# 3. Open the web tool
open http://localhost:5001/airfare_tool.html
```

**API endpoints:**
- `GET  /cities`  — list of supported city markets
- `POST /predict` — `{"city1": "...", "city2": "..."}` → predicted fare + structural premium breakdown
- `GET  /stats`   — model metrics + SHAP importance values

---

## Data Source

U.S. Department of Transportation — [Consumer Airfare Report](https://www.transportation.gov/policy/aviation-policy/us-domestic-airline-fares-consumer-airfare-report), 2021 Q1 – 2025 Q2

> `fare` = average fare paid by fare-paying passengers across all cabin classes. Complimentary and loyalty award travel are excluded.

---

## Team

| Name | Section |
|---|---|
| Carina Zhang | Part 1 — EDA & Linear Models |
| Edwin Zeng | Part 2 — Random Forest Baseline |
| Audrey Luo | Part 3 — XGBoost, SHAP & Web Tool |
