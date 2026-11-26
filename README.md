# CPSC 483 – Predicting Billing Position in Marvel Productions

**Question:** Can we predict an actor's billing position in a Marvel production based on their previous appearances and character history?

**Dataset:** MCU & Marvel Media – Complete Dataset (Kaggle, vanshkumar007)
https://www.kaggle.com/datasets/vanshkumar007/mcu-and-marvel-media-complete-dataset

## How to run

```
pip install pandas numpy scikit-learn
python preprocess.py
```

Expects the raw Kaggle CSVs in `data_raw/`. Writes everything to `data_processed/`.

Models (run after preprocessing):

```
pip install matplotlib
python linear_regression.py      # Julie Yun - writes results/linear_regression/
```

Language: Python 3. Libraries: pandas, NumPy, scikit-learn.

## Outputs (`data_processed/`)

Preprocessing generates the following files in `data_processed/`:

| File | Description | Intended Use |
|---|---|---|
| `marvel_billing_clean.csv` | Cleaned dataset with engineered features, including title, actor, and character names | Data exploration and report visualizations |
| `train.csv` | Unscaled, one-hot encoded training data | Decision tree, random forest, gradient boosting |
| `test.csv` | Unscaled, one-hot encoded testing data | Decision tree, random forest, gradient boosting |
| `train_scaled.csv` | Standardized training data (scaler fit on training data only) | Logistic regression, KNN, SVM, neural networks |
| `test_scaled.csv` | Standardized testing data | Logistic regression, KNN, SVM, neural networks |
| `cleaning_log.txt` | Row counts removed or modified during preprocessing | Reporting data-cleaning decisions |

**Targets (pick one per model, drop the other from X):**
- `billing_tier` (classification): 0 = Lead (billed 1–3), 1 = Supporting (4–8), 2 = Minor (9–15)
- `billing_position` (regression): 1 = top billed … 15

```python
import pandas as pd
train = pd.read_csv("data_processed/train_scaled.csv")
test  = pd.read_csv("data_processed/test_scaled.csv")
X_train, y_train = train.drop(columns=["billing_tier", "billing_position"]), train["billing_tier"]
X_test,  y_test  = test.drop(columns=["billing_tier", "billing_position"]),  test["billing_tier"]
```
## Features

All historical features are computed **only from productions released in earlier years**, preventing data leakage from current or future productions.

| Feature | Meaning |
|---|---|
| `year` | Release year of the production. |
| `actor_prior_appearances` | Number of earlier Marvel productions the actor was billed in. |
| `actor_prior_mcu_appearances` | Number of earlier MCU productions the actor was billed in. |
| `actor_prior_lead_count` | Number of times the actor was billed in the top three in previous productions. |
| `actor_prior_best_billing` | Actor's best (lowest numerical) previous billing position. |
| `actor_prior_mean_billing` | Actor's average billing position across previous productions. |
| `actor_last_billing` | Actor's most recent previous billing position. |
| `actor_years_since_first` | Years since the actor's first Marvel appearance. |
| `actor_years_since_last` | Years since the actor's most recent Marvel appearance. |
| `actor_is_debut` | 1 if this is the actor's first Marvel appearance; otherwise 0. |
| `char_seen_before` | 1 if the character appeared in an earlier production, regardless of actor; otherwise 0. |
| `char_prior_appearances` | Number of earlier productions featuring the character. |
| `char_prior_best_billing` | Character's best historical billing position. |
| `actor_reprising_role` | 1 if the same actor previously portrayed this character; otherwise 0. |
| `actor_prior_same_char_count` | Number of earlier productions in which the actor portrayed the same character. |
| `is_voice_role` | Whether the character credit represents a voice role. |
| `is_uncredited` | Whether the actor's appearance is uncredited. |
| `is_tv_series` | Whether the production is a television series. |
| `is_animated` | Whether the production is animated. |
| `universe_*` | One-hot encoded Marvel universe: MCU, Pre-MCU, Fox / X-Men, or Sony / Spider-Man. |

**Note:** A billing value of `16` indicates no previous billing history (new actor or character). 
## Preprocessing notes

**Raw data.** We use `marvel_cast.csv` (1,628 rows — top-15 billed cast of 115 productions, 1,149 unique actors) and `marvel_master.csv` (161 productions × 50 columns, for universe/TV/animated flags). The ratings and RAG CSVs are production-level and identical for every actor in a production, so they don't help predict billing within a production — not used.

**Cleaning (1,628 → 1,440 rows, 115 → 101 productions):**
- **Duplicate productions (−30 rows):** Two title pairs share the same TMDB id (Spider-Man 1977 / The Amazing Spider-Man 1977 and Spider-Man 1978 / Spider-Man Strikes Back). Kept one of each so appearances aren't double-counted.
- **Mismatched productions (−158 rows, 12 productions):** 10 titles (Loki, Moon Knight, Ms. Marvel, She-Hulk, etc.) linked to behind-the-scenes docs — their "cast" is mostly crew credited as "Self" so the billing order isn't real. Two more (The Defenders 2017, What If...? 2023) linked to unrelated foreign films with the same name. Detected docs automatically: ≥50% of credits are "Self".
- **Wrong years:** Venom: The Last Dance and Kraven the Hunter were listed as 2025, corrected to 2024. The year matters because it controls which appearances count as "previous."
- **Missing character names:** 4 rows, kept and treated as having no character history. `type` is missing for some master rows, so we use `is_tv_series` instead (always populated).
- **Inconsistent character names:** "Tony Stark" vs "Tony Stark / Iron Man", "Bruce Banner / Hulk" vs "Bruce Banner / The Hulk", etc. Normalized to lowercase, stripped parentheticals and honorifics, split on "/" into alias sets — two credits match if they share any alias. Generic roles (guard, reporter, henchman…) are excluded from character history. "(voice)" and "(uncredited)" became binary features.

Final: **1,440 rows, 101 productions, 1,028 unique actors**, no duplicate (production, actor) pairs.

**Features.** All history is computed from strictly earlier years only (same-year productions don't count — we have year not date, so this is the safest cutoff). See the features table above.

**Dropped columns:**
- `popularity`: TMDB measures this today, so it's partly a result of the billing we're predicting — data leakage.
- `profile_path`, IDs, raw names: no predictive value.
- `is_mcu_canon`: redundant with `universe`.

**Targets.** `cast_order` → `billing_position` (1–15) for regression. For classification, `billing_tier`: Lead 1–3 (20.8%), Supporting 4–8 (34.0%), Minor 9–15 (45.2%). Classes are mildly imbalanced, so macro-F1 is a better metric than accuracy; consider `class_weight="balanced"`.

**Transformations:**
- `universe` one-hot encoded (4 columns).
- 12 numeric features standardized (z-score), scaler fit on train only.
- Binary/one-hot columns left as 0/1.
- No outliers removed — high values (e.g. 9 prior appearances = Samuel L. Jackson) are real.
- "No history" is encoded as billing position 16 (one worse than the worst real position), plus the `actor_is_debut` / `char_seen_before` flags.

**Train/test split.** Chronological — whole productions go to one side. Train: up to 2021 (1,143 rows, 79.4%), Test: 2022+ (297 rows, 20.6%). Class distribution is similar in both (Lead ~21% each). For tuning, do CV on train only.

**Limitations:**
- TMDB only lists top 15 billed, so the full cast isn't captured.
- The 12 removed productions (Loki, Ms. Marvel, etc.) are missing from history, so some actors' prior counts are undercounted.
- Character matching is heuristic — common short names like "Peter" or "Charles" might match across unrelated characters.
- ~72% of rows are debut actors, so most rows have no prior history.
