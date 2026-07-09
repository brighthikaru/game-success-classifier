# Game Success Classifier

Can a video game's own release-time data predict whether critics will love it? This project trains and compares machine learning models to predict whether a game will be **critically acclaimed** (Metacritic score ≥ 75) using only characteristics genuinely known at or before release — genre, platform count, ESRB rating, tag diversity, and release timing.

Built as Mini Project 3 for the IOD Data Science & AI course.

## Business Question

The video game industry earns over $180B a year, with thousands of titles released annually. A strong Metacritic score drives storefront visibility and can determine a studio's commercial fate. So: **can we predict critical success before or at release, using only pre-release signals** — without leaking the answer through the score itself?

A game is labelled **Hit** (Metacritic ≥ 75) or **Not Hit** (< 75).

## Data

- **Source:** [RAWG Video Games Database API](https://rawg.io/apidocs), collected via the custom script in `src/collect_rawg_data.py`
- **Size:** 7,141 games after cleaning
- **Coverage:** 1980–2024 (skews toward older, already-reviewed titles — RAWG's coverage of 2022+ releases is thin)
- **Class balance:** 47.2% Hit / 52.8% Not Hit — close enough to balanced that no resampling was needed

The raw data was genuinely messy: genres, platforms, and tags arrived as pipe-separated strings in a single column (e.g. `"Action|RPG|Adventure"`), and ~33% of games had no ESRB rating recorded. All of this is cleaned and engineered into model-ready features in the notebook (Section 2).

**A note on data leakage, and why the results below look modest:** an earlier version of this project also used `rating_count` and `playtime` as features. Both are RAWG metrics that accumulate *after* a game releases — for an older title, `rating_count` can reflect decades of accumulated community engagement, not anything knowable at launch. `rating_count` alone dominated every other feature by more than 5×, which was the tell that it was leakage rather than a genuine signal. Both features were removed and the model retrained on features that are actually available at or before release: `metacritic` and community `rating` were already excluded for the same reason (they're near-restatements of the target). See Section 4 of the notebook for the full explanation.

## Approach

1. **Clean & engineer features** — parse pipe-separated columns, flag holiday releases, fill missing ESRB as "Not Rated"
2. **Feature selection** — exclude leakage-risk columns (`metacritic`, `rating`, `rating_count`, `playtime`); keep only signals genuinely available at or before release
3. **Train 4 models** — a majority-class baseline (sanity check), Logistic Regression, Random Forest, and a Stacking ensemble (Logistic Regression + Random Forest combined via a meta-learner)
4. **Evaluate** — Macro F1 (primary metric, since classes are only near-balanced) and ROC-AUC (secondary)
5. **Tune** — GridSearchCV with stratified 3-fold cross-validation on the best-performing model
6. **Explain** — feature importance (native Gini importance for the tree-based champion, with a permutation-importance fallback for ensembles that don't expose one) to identify which features the model actually relies on

## Results

| Model | Macro F1 | ROC-AUC |
|---|---|---|
| Baseline (majority class) | 0.345 | 0.500 |
| Logistic Regression | 0.546 | 0.566 |
| Stacking Classifier | 0.563 | 0.599 |
| **Random Forest (best, tuned)** | **0.570** | **0.615** |

All four real models beat the baseline, confirming there's genuine — if modest — predictive signal in features that are truly available before a game releases. Hyperparameter tuning of the champion model gave a small but real improvement (Macro F1: 0.564 → 0.570).

**These numbers are a lot lower than an earlier version of this project reported (Macro F1 0.690, ROC-AUC 0.747), and that's intentional.** That earlier result relied on `rating_count`, which turned out to be data leakage (see the note under **Data** above). Once it — and the similarly-leaky `playtime` — were removed, honest performance dropped substantially. That drop is itself the more interesting finding: it shows how much of the original score was inflated by information the model would never actually have at prediction time.

**What actually drives the (smaller, honest) signal:** with the champion now a Random Forest, feature importance is read directly off its native Gini importance rather than a permutation-based proxy. `release_year` ranks highest, with `primary_genre` and `n_platforms` close behind — no single feature dominates the way `rating_count` used to. That flatter, more diffuse picture is a more believable one for a set of features that are all genuinely knowable pre-release.

![Feature importance](outputs/feature_importance.png)

## Limitations

- Dataset skews toward older, already-reviewed games — the API returned limited data for 2022 onward
- No developer or publisher data was available through the API — likely a real driver of critical success that's missing entirely
- Metacritic itself isn't a perfect measure of quality — a game can score below 75 and still be deeply valued by its audience
- Model performance is modest (Macro F1 0.57) once post-release signals like `rating_co