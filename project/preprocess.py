# CPSC 483 Project - Phase 2: Data Preprocessing
# Cleans marvel_cast.csv and engineers actor/character history features.
# Run: python preprocess.py  (outputs go to data_processed/)

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW = Path("data_raw")
OUT = Path("data_processed")
OUT.mkdir(exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# 1. Load
cast = pd.read_csv(RAW / "marvel_cast.csv")
master = pd.read_csv(RAW / "marvel_master.csv")
log(f"[load] marvel_cast: {cast.shape[0]} rows x {cast.shape[1]} cols, "
    f"{cast[['title', 'year']].drop_duplicates().shape[0]} productions, "
    f"{cast.actor_tmdb_id.nunique()} unique actors")
log(f"[load] marvel_master: {master.shape[0]} rows x {master.shape[1]} cols")

# 2. Remove duplicate productions
# Two title pairs share the same TMDB id (1977 Spider-Man pilot/series and 1978 film
# each listed under two names) - keeping both would double-count those appearances.
prods = cast[["title", "year", "tmdb_id"]].drop_duplicates()
dup_prods = prods[prods.duplicated("tmdb_id", keep="first")]
before = len(cast)
cast = cast.merge(dup_prods.assign(_dup=1), on=["title", "year", "tmdb_id"], how="left")
cast = cast[cast._dup.isna()].drop(columns="_dup")
log(f"[dedupe] removed duplicate productions {list(dup_prods.title + ' ' + dup_prods.year.astype(str))}: "
    f"{before - len(cast)} rows")

# 3. Remove mismatched productions
# The scraper matched wrong TMDB entries for some titles:
# - behind-the-scenes docs ("Assembled" etc.) have cast listed as "Self", so
#   billing order doesn't reflect the actual show's billing
# - some titles matched unrelated foreign films (The Defenders 2017 -> Russian film,
#   What If...? 2023 -> Filipino production)
# These don't describe the Marvel production so we remove them.
self_mask = cast.character.fillna("").str.match(r"^\s*(self|himself|herself)\b", case=False)
self_share = self_mask.groupby([cast.title, cast.year]).mean()
doc_titles = set(self_share[self_share >= 0.5].index)
foreign_titles = {("The Defenders", 2017), ("What If...?", 2023)}
bad = doc_titles | foreign_titles
before = len(cast)
key = list(zip(cast.title, cast.year))
cast = cast[[k not in bad for k in key]].copy()
log(f"[mismatch] removed {len(bad)} mis-matched productions "
    f"({sorted(t for t, _ in bad)}): {before - len(cast)} rows")

# 4. Correct inconsistent release years (verified against actual release dates)
year_fix = {"Venom: The Last Dance": 2024, "Kraven the Hunter": 2024}
for t, y in year_fix.items():
    n = (cast.title == t).sum()
    cast.loc[cast.title == t, "year"] = y
    log(f"[year] {t}: year corrected to {y} ({n} rows)")

# 5. Attach production-level info from master table (join on TMDB id)
m = master.dropna(subset=["tmdb_id"]).drop_duplicates("tmdb_id")
# ('type' is missing for some rows, so we use the complete is_tv_series flag instead)
m = m[["tmdb_id", "universe", "is_animated", "is_tv_series", "is_mcu_canon"]]
cast = cast.merge(m, on="tmdb_id", how="left")
assert cast.universe.notna().all(), "every production should match the master table"
log(f"[join] universe/type attached; universes: {cast.universe.value_counts().to_dict()}")

# Within-production duplicate actors (same actor listed twice) - check
d = cast.duplicated(["tmdb_id", "actor_tmdb_id"]).sum()
log(f"[check] duplicate (production, actor) rows after cleaning: {d}")
cast = cast.drop_duplicates(["tmdb_id", "actor_tmdb_id"], keep="first")

# 6. Character name cleaning
# Raw names are inconsistent ("Tony Stark" vs "Tony Stark / Iron Man", "(voice)", etc.)
# We flag voice/uncredited as features, strip parentheticals and honorifics, then split
# on "/" to build alias sets so we can match the same character across productions.
# Generic roles (guard, reporter, etc.) are excluded from character history.
# ~10 rows with missing character names are kept and treated as having no history.
log(f"[character] missing character names: {cast.character.isna().sum()} (kept, treated as no history)")
cast["character"] = cast["character"].fillna("")
cast["is_voice_role"] = cast.character.str.contains(r"\(voice\)", case=False).astype(int)
cast["is_uncredited"] = cast.character.str.contains(r"uncredited", case=False).astype(int)

GENERIC = re.compile(
    r"\b(guard|agent|reporter|henchman|thug|soldier|officer|cop|police|nurse|doctor|"
    r"scientist|kid|boy|girl|man|woman|narrator|self|waiter|waitress|bartender|driver|"
    r"pilot|student|teacher|technician|tech|newscaster|anchor|security|goon|"
    r"mercenary|prisoner|bystander|civilian|pedestrian|patron|crowd|dancer|"
    r"secretary|assistant|receptionist|clerk|announcer|host|vendor|customer)\b"
)
HONORIFIC = re.compile(
    r"^(the|dr\.?|doctor|prof\.?|professor|lt\.?|col\.?|colonel|lt\. col\.|"
    r"president|sen\.?|senator|general|gen\.?|mr\.?|mrs\.?|ms\.?|young|old)\s+"
)


def char_aliases(raw: str) -> frozenset:
    s = raw.lower()
    s = re.sub(r"\(.*?\)", " ", s)          # (voice), (uncredited), ...
    s = s.replace('"', " ").replace("'", " ")
    out = set()
    for part in s.split("/"):
        p = re.sub(r"\s+", " ", part).strip(" .-")
        for _ in range(3):                  # strip stacked honorifics
            p = HONORIFIC.sub("", p).strip()
        if len(p) < 3 or GENERIC.search(p) and len(p.split()) <= 2:
            continue
        out.add(p)
    return frozenset(out)


cast["char_aliases"] = cast.character.map(char_aliases)

# 7. Targets
# cast_order is 0-based, TMDB gives top 15 billed, so billing_position = cast_order + 1
# billing_tier: 0 = Lead (1-3), 1 = Supporting (4-8), 2 = Minor (9-15)
cast["billing_position"] = cast.cast_order + 1
cast["billing_tier"] = pd.cut(cast.billing_position, bins=[0, 3, 8, 15], labels=[0, 1, 2]).astype(int)

# 8. Feature engineering: actor & character history
# Only look at productions from strictly earlier years - same-year productions aren't
# counted as "previous" since we only have release year (not exact date), to avoid leakage.
cast = cast.sort_values(["year", "tmdb_id", "cast_order"]).reset_index(drop=True)
NO_HISTORY_BILLING = 16  # one worse than the lowest possible billing (15)

feat = []
for i, r in cast.iterrows():
    prev = cast[cast.year < r.year]
    a = prev[prev.actor_tmdb_id == r.actor_tmdb_id]
    if r.char_aliases:
        cmask = prev.char_aliases.map(lambda s: bool(s & r.char_aliases)).astype(bool)
        ch = prev.loc[cmask]
    else:
        ch = prev.iloc[0:0]
    ac = ch[ch.actor_tmdb_id == r.actor_tmdb_id]
    feat.append({
        # actor history
        "actor_prior_appearances": len(a),
        "actor_prior_mcu_appearances": int((a.universe == "MCU").sum()),
        "actor_prior_lead_count": int((a.billing_tier == 0).sum()),
        "actor_prior_best_billing": a.billing_position.min() if len(a) else NO_HISTORY_BILLING,
        "actor_prior_mean_billing": a.billing_position.mean() if len(a) else NO_HISTORY_BILLING,
        "actor_last_billing": a.sort_values("year").billing_position.iloc[-1] if len(a) else NO_HISTORY_BILLING,
        "actor_years_since_first": r.year - a.year.min() if len(a) else 0,
        "actor_years_since_last": r.year - a.year.max() if len(a) else 0,
        "actor_is_debut": int(len(a) == 0),
        # character history
        "char_seen_before": int(len(ch) > 0),
        "char_prior_appearances": ch.tmdb_id.nunique(),
        "char_prior_best_billing": ch.billing_position.min() if len(ch) else NO_HISTORY_BILLING,
        "actor_reprising_role": int(len(ac) > 0),
        "actor_prior_same_char_count": ac.tmdb_id.nunique(),
    })
cast = pd.concat([cast, pd.DataFrame(feat)], axis=1)

# 9. Select features, drop irrelevant or leaky columns
# Dropped: popularity (TMDB measures it today, partly a result of billing -> leakage),
# profile_path (URL, no signal), raw identifiers/text replaced by engineered features,
# cast_order (IS the target), is_mcu_canon (redundant with universe == "MCU").
# Ratings/box office from other files are production-level, identical for every actor
# in a production, so they can't explain within-production billing differences.
NUMERIC = [
    "year",
    "actor_prior_appearances", "actor_prior_mcu_appearances", "actor_prior_lead_count",
    "actor_prior_best_billing", "actor_prior_mean_billing", "actor_last_billing",
    "actor_years_since_first", "actor_years_since_last",
    "char_prior_appearances", "char_prior_best_billing", "actor_prior_same_char_count",
]
BINARY = ["actor_is_debut", "char_seen_before", "actor_reprising_role",
          "is_voice_role", "is_uncredited", "is_tv_series", "is_animated"]
CATEGORICAL = ["universe"]
TARGETS = ["billing_position", "billing_tier"]
ID_COLS = ["title", "year", "actor_name", "character"]  # kept in clean file for readability only

clean = cast[ID_COLS + [c for c in NUMERIC if c != "year"] + BINARY + CATEGORICAL + TARGETS].copy()
clean.to_csv(OUT / "marvel_billing_clean.csv", index=False)
log(f"[features] clean dataset: {clean.shape[0]} rows, "
    f"{cast[['title', 'year']].drop_duplicates().shape[0]} productions, "
    f"{cast.actor_tmdb_id.nunique()} unique actors")

# 10. One-hot encode universe
model_df = pd.get_dummies(cast[NUMERIC + BINARY + CATEGORICAL + TARGETS + ["title"]],
                          columns=CATEGORICAL, prefix="universe", dtype=int)

# 11. Train/test split - chronological by production
# Since the task is predicting from previous appearances, train on older productions
# and test on the newest (~20%). Whole productions stay on one side so no production
# is split between train and test.
test_share = model_df.groupby("year").size().sort_index(ascending=False).cumsum() / len(model_df)
test_start_year = (test_share - 0.20).abs().idxmin()   # year whose cutoff is closest to 80/20
train = model_df[model_df.year < test_start_year].drop(columns="title")
test = model_df[model_df.year >= test_start_year].drop(columns="title")
log(f"[split] train: years <= {test_start_year - 1}, {len(train)} rows "
    f"({len(train) / len(model_df):.1%}); test: years >= {test_start_year}, "
    f"{len(test)} rows ({len(test) / len(model_df):.1%})")
for name, d in [("train", train), ("test", test)]:
    dist = d.billing_tier.value_counts(normalize=True).sort_index().round(3).to_dict()
    log(f"[split] {name} billing_tier distribution (0=lead,1=supporting,2=minor): {dist}")

train.to_csv(OUT / "train.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

# 12. Standardize numeric features (z-score), fit on train only
# Distance/gradient-based models need this; tree models can use the unscaled files.
scaler = StandardScaler().fit(train[NUMERIC])
train_s, test_s = train.copy(), test.copy()
train_s[NUMERIC] = scaler.transform(train[NUMERIC])
test_s[NUMERIC] = scaler.transform(test[NUMERIC])
train_s.to_csv(OUT / "train_scaled.csv", index=False)
test_s.to_csv(OUT / "test_scaled.csv", index=False)
log(f"[scale] standardized {len(NUMERIC)} numeric features; binary/one-hot left as 0/1")

(OUT / "cleaning_log.txt").write_text("\n".join(log_lines) + "\n")
