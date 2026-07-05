# Game Success Classifier

Can a video game's own release-time data predict whether critics will love it? This project trains and compares machine learning models to predict whether a game will be **critically acclaimed** (Metacritic score ≥ 75) using only characteristics known at or before release — genre, platform, ESRB rating, playtime, release timing, and community engagement.

Built as Mini Project 3 for the IOD Data Science & AI course.

## Business Question

The video game industry earns over $180B a year, with thousands of titles released annually. A strong Metacritic score drives storefront visibility and can determine a studio's commercial fate. So: **can we predict critical success before or at release, using only pre-release signals** — without leaking the answer through the score itself?

A game is labelled **Hit** (Metacritic ≥ 75) or **Not Hit** (< 75).

## Data

- **Source:** [RAWG Video Games Database API](https://rawg.io/apidocs), collected via the custom script in `src/collect_rawg_data.py`
- **Size:** 7,141 games after cleaning
- **Coverage:** 1980–2024 (skews toward older, already-reviewed titles — RAWG's coverage of 2022+ releases is thin)
- **Class balance:** 47.2% Hit / 52.8% Not Hit — close enough to balanced that no resampling was needed

The raw data was genuinely messy: genres, platforms, and tags arrived as pipe-separated strings in a single column (e.g. `"Action|RPG|Adventure"`), ~33% of games had no ESRB rating recorded, and playtime was heavily right-skewed. All of this is cleaned and engineered into model-ready features in the notebook (Section 2), including a deliberate exclusion of `metacritic` and community `rating` from the feature set to avoid data leakage — including either would let the model "cheat" by reading the answer.

## Approach

1. **Clean & engineer features** — parse pipe-separated columns, log-transform playtime, flag holiday releases, fill missing ESRB as "Not Rated"
2. **Feature selection** — exclude leakage-risk columns (`metacritic`, `rating`); keep only pre-release-available signals
3. **Train 4 models** — a majority-class baseline (sanity check), Logistic Regression, Random Forest, and a Stacking ensemble (Logistic Regression + Random Forest combined via a meta-learner)
4. **Evaluate** — Macro F1 (primary metric, since classes are only near-balanced) and ROC-AUC (secondary)
5. **Tune** — GridSearchCV with stratified 3-fold cross-validation on the best-performing model
6. **Explain** — permutation importance to identify which features the champion model actually relies on

## Results

| Model | Macro F1 | ROC-AUC |
|---|---|---|
| Baseline (majority class) | 0.345 | 0.500 |
| Logistic Regression | 0.628 | 0.690 |
| Random Forest | 0.688 | 0.742 |
| **Stacking Classifier (best)** | **0.690** | **0.747** |

All three real models comfortably beat the baseline, confirming genuine predictive signal. Hyperparameter tuning of the champion model produced no meaningful improvement (Macro F1 moved from 0.6904 to 0.6894) — it was already close to its ceiling given the available features.

**The standout finding:** `rating_count` — essentially how much community engagement and buzz a game generates — is by far the strongest predictor. Permutation importance shows it accounts for more than **5× the impact of any other single feature**, well ahead of playtime, release year, and genre.

![Feature importance](outputs/feature_importance.png)

## Limitations

- Dataset skews toward older, already-reviewed games — the API returned limited data for 2022 onward
- No developer or publisher data was available through the API — likely a real driver of critical success that's missing entirely
- Metacritic itself isn't a perfect measure of quality — a game can score below 75 and still be deeply valued by its audience
- Model performance is moderate (Macro F1 0.69); there's more predictive signal available than these features alone capture

## Future Improvements

- Recollect a larger, date-balanced sample so recent releases aren't underrepresented
- Add developer/publisher fields and NLP features from game descriptions
- Explore multi-class prediction (Not Hit / Average / Good / Great)
- Package the model behind a simple app — pick genre, platform, and ESRB rating, get a prediction

## Repo Structure

```
game-success-classifier/
├── README.md
├── requirements.txt
├── src/
│   └── collect_rawg_data.py          # pulls data from the RAWG API
├── notebooks/
│   └── MP03_Game_Success_Classifier.ipynb   # full analysis, start to finish
├── data/
│   └── rawg_games.csv                # collected dataset (git-ignored)
└── outputs/
    └── *.png                          # charts saved out by the notebook
```

## Running This Yourself

```bash
git clone https://github.com/brighthikaru/game-success-classifier.git
cd game-success-classifier
pip install -r requirements.txt
jupyter notebook notebooks/MP03_Game_Success_Classifier.ipynb
```

To recollect fresh data instead of using the included CSV, set a `RAWG_API_KEY` in a local `.env` file and run `python src/collect_rawg_data.py`.

## Data Attribution

Data sourced from the [RAWG Video Games Database API](https://rawg.io) — used under its free tier with attribution.
