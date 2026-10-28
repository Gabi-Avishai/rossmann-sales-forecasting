# Rossmann Store Sales — Initial Plan

## Task

Predict daily `Sales` for each Rossmann store over a six-week future window
(`Date + Store -> Sales`). Official metric: **RMSPE** (Root Mean Square
Percentage Error), with zero-sales days excluded from scoring.

Source: [Rossmann_Store_Sales_Student_Assignment.docx](../../files_to_ignore/Rossmann_Store_Sales_Student_Assignment.docx)
Data: `train.csv`, `test.csv`, `store.csv`, `sample_submission.csv` in this
directory.

## Ground rules

- **`test.csv` stays unseen until the final evaluation.** All EDA, feature
  design, validation, and model comparison happen on `train.csv` alone,
  using a held-out time slice of `train.csv` as a proxy for the real test
  period. `test.csv` is touched only once, at the very end, to produce the
  final scored predictions — never to inform earlier decisions.
- Respect time order everywhere: no shuffled CV, no features computed with
  information that would not exist yet at forecast time.
- Prefer simple, explainable steps first; add complexity only when it earns
  its keep.

## Data facts gathered so far

- Train covers 2013-01-01 to 2015-07-31, 1,115 stores. Test covers exactly
  2015-08-01 to 2015-09-17 (six weeks), 856 of the 1,115 stores — so the
  internal holdout should mimic "last ~6 weeks, subset of stores" as
  closely as possible.
- `StateHoliday` has a mixed-type bug: both the string `'0'` and the
  integer `0` appear for "not a holiday" — needs cleaning before use.
- `Open` is 0 for ~17% of train rows. 54 rows have `Open==1` with
  `Sales==0` (anomalies, worth an EDA callout). Test has 11 `Open` nulls
  needing an imputation decision.
- `store.csv` has real missingness: `CompetitionDistance` (3),
  `CompetitionOpenSinceMonth/Year` (354), `Promo2SinceWeek/Year` and
  `PromoInterval` (544). Several of these fields are only "known" from a
  specific date onward, which is exactly the kind of leakage trap the
  assignment warns about.
- RMSPE isn't in sklearn — needs a custom function, plus a decision on
  whether to train on `log1p(Sales)` (standard for this competition).

## Phased plan

Each phase gets its own notebook (new file per experiment/variant), and
builds on saved artifacts (cleaned data, feature pipeline, predictions)
from the previous phase rather than re-deriving everything.

| # | Notebook | What it does | Interview-relevant concepts |
|---|---|---|---|
| 0 | `rossmann_eda.ipynb` | Clean `StateHoliday`, investigate closed-day/zero-sales anomalies, weekly & yearly seasonality, promo/holiday effects, per-`StoreType` behavior, competition-distance effects | Data cleaning judgment, seasonality decomposition, business-driven EDA |
| 1 | `rossmann_baseline.ipynb` | Time-based validation split (last ~6 weeks of `train.csv` held out, matching the real test window) + custom RMSPE. Baselines: global mean, per-store mean, per-store x day-of-week mean, "same day last year" | Why naive baselines matter; time-series CV done right (no shuffling, no future leakage) |
| 2 | `rossmann_linear.ipynb` | Feature engineering pass 1 (date parts, promo flags, holiday windows, competition-open flag, encoding for `StoreType`/`Assortment`) + regularized linear regression on `log1p(Sales)` | Feature engineering discipline, log-target rationale, why linear models underfit this problem (sets up the contrast with trees) |
| 3 | `rossmann_trees.ipynb` | Same features, plus lag/rolling features computed with correct as-of cutoffs. **Compare Random Forest vs. `HistGradientBoostingRegressor`** on the held-out slice, with train/val curves and RMSPE tracked per iteration for the boosting model | Bagging vs. boosting tradeoffs, feature importance, early stopping, avoiding leakage in lag features |
| 4 (stretch, later) | `rossmann_nn_sota.ipynb` | Try a state-of-the-art tabular network approach (e.g. entity embeddings for categoricals, feeding into an MLP or a modern tabular architecture such as TabNet/FT-Transformer, CPU-feasible size) | Embeddings vs. one-hot, when NNs help vs. hurt on tabular data, tradeoffs vs. tree ensembles |
| 5 | `rossmann_error_analysis_and_writeup.ipynb` | Compare all models on the common holdout, break down RMSPE by store type/promo/holiday/day-of-week, residual analysis. **Only here does `test.csv` get touched**, to produce final predictions from the chosen model. Written deliverables: approach summary, limitations, business use, production/monitoring notes (retraining cadence, drift checks, handling of new/closed stores) | Model comparison narrative, communicating results to non-technical stakeholders, productionization thinking |

Phase 4 (state-of-the-art networks) is deferred until after the tree-based
comparison in phase 3 is done — revisit scope for it at that point.
