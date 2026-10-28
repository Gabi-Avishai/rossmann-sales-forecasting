# Rossmann Store Sales Forecasting

Predict daily `Sales` for each of 1,115 Rossmann drug stores, roughly six weeks into
the future (`Date + Store -> Sales`). The task follows a course assignment brief, which is
not included in this repository. Data (`train.csv`, `test.csv`, `store.csv`,
`sample_submission.csv`) goes in this directory; see [Data and running](#data-and-running).
Full phase-by-phase plan and rationale: [Initial_Plan.md](Initial_Plan.md).

**Non-technical summary of results and business implications: [SUMMARY.md](SUMMARY.md).**
This README is the engineering log — what was tried, what worked, and what didn't.

## Framing

Store managers use this forecast for staffing and operational planning, so the model
needs to be honest about *when* it's wrong, not just accurate on average — a forecast
that's systematically too high before a holiday, say, costs real overstaffing. That
framing motivates a few choices that run through the whole project:

- **Official metric is RMSPE** (percentage error, zero-sales days excluded) — a
  percentage metric because a fixed absolute error means very different things for a
  small store and a flagship store; the leaderboard-style RMSE would let big stores
  dominate the score.
- **Only information available at forecast time is fair game.** Several `store.csv` /
  `train.csv` fields exist historically but wouldn't be known cleanly at forecast time
  for every store (e.g. `CompetitionOpenSince*`, `Promo2Since*`) — feature engineering
  has to respect that, not just impute blindly.
- **`test.csv` stays unseen until the final evaluation notebook.** All model
  development, validation, and comparison happens against a held-out slice of
  `train.csv` that mimics the real test window, so the final test-set score is a
  genuine check rather than something implicitly tuned against.

## Roadmap

| Phase | Goal | What | Status |
|---|---|---|---|
| 0 | Understand the data | EDA (`rossmann_eda.ipynb`) — seasonality, promos, holidays, store differences, competition, data-quality checks on both `train.csv` and `store.csv` | done — see [Phase 0 results](#phase-0-results-eda) |
| 1 | Validation + floor | Time-based holdout matching the ~6-week horizon, custom RMSPE, naive per-store/day-of-week/promo baselines (`rossmann_baseline.ipynb`) | done — see [Phase 1 results](#phase-1-results-validation--baselines) |
| 2 | Real features + linear model | Date/promo/holiday/competition features (leakage-safe), regularized linear regression on `log1p(Sales)` (`rossmann_linear.ipynb`) | done — see [Phase 2 results](#phase-2-results-feature-engineering--linear-model) |
| 3 | Tree ensembles | Same features + leakage-safe lag/rolling features. **Random Forest vs. `HistGradientBoostingRegressor`** compared head-to-head (`rossmann_trees.ipynb`) | done — see [Phase 3 results](#phase-3-results-tree-ensembles--lag-features) |
| 4 | State-of-the-art networks | Entity embeddings feeding a small MLP, tested for real rather than skipped — Phase 3's best model only beat Phase 1's baseline lookup table by ~4% relative RMSPE despite substantial engineering, which is reason enough to see whether a different modeling paradigm can separate itself further (`rossmann_nn_sota.ipynb`) | done — see [Phase 4 results](#phase-4-results-entity-embeddings--mlp) |
| 4.5 | Calibration + ensembling | RMSPE-oriented output shrinkage and a weighted geometric-mean ensemble of all four models, both fitted on the inner split only (`rossmann_calibration.ipynb`) | done — see [Phase 4.5 results](#phase-45-results-calibration--ensembling) |
| 4.6 | Feature rebuild | Train-like-you-serve forecast origins (fixing the `HorizonDay` defect), `SalesYearAgo` split, and known-schedule features — promo/holiday proximity, closure adjacency, post-blackout reopening. Run as an ablation (`rossmann_features_v2.py`, `rossmann_features_v2.ipynb`) | done — see [Phase 4.6 results](#phase-46-results-feature-rebuild) |
| 4.7 | LightGBM/CatBoost, tuned | `Store` finally usable as a native high-cardinality categorical, proper hyperparameter search, loss-weighting sweep, then recalibration and re-ensembling on the v2 feature set (`rossmann_gbm_tuned.ipynb`) | done — see [Phase 4.7 results](#phase-47-results-lightgbmcatboost-tuned) |
| 5 | Error analysis + writeup | Compare all models on the shared holdout, break down error by store type/promo/holiday/day-of-week. **`test.csv` is touched only here**, to produce final predictions. Business use, limitations, production/monitoring notes (`rossmann_final.ipynb`) | done — see [Phase 5 results](#phase-5-results-final-scoring--error-analysis) |

## Pitfalls (found or anticipated)

- **Store-selection confounds, not causal effects.** EDA found that "Sunday sells more"
  and "closer competition sells more" are both artifacts of *which* stores choose to
  open on Sundays / are located near competitors (structurally higher-traffic stores),
  not genuine effects of the day or the distance. Tree models can partially absorb this
  via interactions with store identity, but it's not a causal lever the business can pull.
- **A single historical data point is noisier than a multi-year average** — Phase 1
  found the naive "same day last year" baseline actually *underperforms* a plain
  `store x DayOfWeek x Promo` mean. Lag/lookback features in Phase 2/3 should be built
  to complement averaging, not replace it.
- **Leakage via `store.csv` "since" fields** — `CompetitionOpenSince*` / `Promo2Since*`
  describe when something started, which only becomes "known" from that date onward;
  naively joining them onto every historical row would leak future knowledge backward.
- **`test.csv` covers only 856 of the 1,115 stores** — resist the temptation to filter
  training/validation down to those stores, since that requires inspecting `test.csv`
  and breaks the "unseen until final" rule.
- **RMSPE ignores zero-sales days** — any custom loss/metric code must mask `Sales==0`
  rows exactly like the official metric, or comparisons across notebooks won't line up.
- **181 of 1,115 stores have an identical ~6-month reporting blackout** (2014-07-01 to
  2014-12-31) — not random missingness. Any 52-week lag/rolling feature in Phase 2/3
  needs an explicit fallback for these stores rather than a silent `NaN`.
- **`Customers` is not in `test.csv`** — it's historically informative (`Sales =
  Customers x BasketSize`) but must stay exploration-only; it can never be a model
  input, or Phase 5's final predictions would silently rely on a field that doesn't
  exist at forecast time. A *lagged historical aggregate* of `Customers` (e.g. a
  trailing rolling mean, as used for Phase 3's lag features) is a different, legitimate
  category — it's derived entirely from past training data, not from the row being
  predicted, so it works identically whether or not the target frame has a `Customers`
  column at all (verified directly by dropping the column and re-running the pipeline).
- **`HistGradientBoostingRegressor` caps native categorical cardinality at 255** —
  passing `Store` (1,115 values) crashes outright; leaving it as a raw ordinal integer
  instead doesn't crash but silently quantile-bins it into <=255 arbitrary buckets of
  adjacent, unrelated store IDs. Fixed in Phase 3 by target-encoding `Store` into a
  single `StoreMeanLogSales` continuous feature (fit on `train_fold` only), used for
  both tree models so the RF-vs-HGB comparison isn't confounded by an encoding
  difference.
- **Rolling/lag features need two different construction modes, not one.** Training
  rows are fit one at a time and can each use a per-row "as of yesterday" window, but a
  validation/test batch is a single forecast made from one data cutoff — every row in
  that batch must reuse the SAME frozen rolling window (computed as of the last known
  date), not a fresh one per day, or it leaks data a real 6-week-ahead forecast would
  never have. In v1 this left a real train/inference staleness mismatch: training rows saw
  yesterday's rolling mean, validation/test rows saw one up to 42 days stale, and
  `HorizonDay` — meant to signal that staleness — was hard-coded to `0` for every training
  row, making it constant and carrying zero learnable signal (measured importance exactly
  `0.000000` in both tree models). **Fixed in Phase 4.6** by tiling training rows into
  synthetic 42-day forecast blocks, each with its own origin, so training rows are stale
  by 1–42 days exactly as served rows are. Worth 4% relative RMSPE on its own.
- **A holdout carved from the middle of a time series cannot expose defects that live at
  its edge.** "Days until the next holiday" is undefined for rows near the end of the known
  calendar, so it falls back to a sentinel for **93%** of `test.csv` rows against **6%** of
  training rows — a serious train/serve mismatch. `val_fold` sits mid-panel with six further
  weeks of calendar after it, so its values are almost never censored and the defect was
  invisible there. It surfaced only in a train-vs-test feature-distribution check at final
  scoring. Right-censored "time until" features should be clipped at a horizon that is
  always observable (e.g. `min(days_until, 30)`), so "no event within 30 days" is a bucket
  training data contains in quantity.
- **Working around a library limit cost more than switching libraries.** `Store` (1,115
  levels) was target-encoded from Phase 3 onward purely to dodge
  `HistGradientBoostingRegressor`'s 255-category cap. Once LightGBM made a native
  high-cardinality categorical possible, that single change was worth **7.7%** relative
  RMSPE (0.1125 target-encoded → 0.1038 native), and adding the target encoding back on
  top bought nothing (0.1035, within noise). Four phases of modelling sat on top of a
  workaround that was quietly costing more than most of the feature engineering gained.
- **Check that a library's GPU path supports your data, not just your hardware.**
  LightGBM's GPU backend refuses this dataset outright — `bin size 1101 cannot run on
  GPU` — because `Store` has more categories than its GPU histogram allows. The GPU is
  unusable *precisely because of* the feature that motivated adopting the library. CatBoost's
  GPU works but buys only ~1.5x on this data size, against fits already measured in seconds.
- **Calibration was mostly a band-aid for the train/serve mismatch.** The optimal shrink
  factor was 0.951 on v1 features, worth ~8% relative RMSPE. After Phase 4.6 fixed the
  horizon mismatch it moved to 1.010, and by Phase 4.7 to 0.995 — worth ~0.6%. A
  correction that large is a signal to go looking for the bias's cause, not just to keep
  applying the correction.
- **Permutation importance is not a substitute for a drop test.** `SalesYearAgo` had a
  permutation importance of −0.185, the most negative of any feature, which reads as "this
  column is actively hurting." Removing it outright made RMSPE *worse* (0.1293 → 0.1341).
  The two measure different things — shuffling a feature under log-space MSE versus
  removing it under RMSPE — and with correlated features the shuffle can actively mislead.
  Confirm with the ablation you actually care about before deleting a feature.
- **The Rossmann promo calendar repeats on an exact 364-day cycle.** `P(promo 364 days ago
  == promo today) = 1.000`. That is why the 364-day lag (not 365) is the right one, and why
  smoothing it over neighbouring weeks backfires: the week-adjacent points carry the
  *opposite* promo state, and averaging them in dropped correlation with actual sales from
  0.927 to 0.786.
- **RMSPE is not minimized by the conditional mean, so every model needs an explicit
  output calibration step.** Because RMSPE weights each row by `1/y²`, low-sales days
  dominate the score and the optimal point prediction sits several percent *below* the
  conditional mean. Phase 1 discovered this (a flat 2% shrink improved the baseline from
  0.147 to 0.1448) and then it was silently dropped for three phases — Phases 2-4 all
  predicted `expm1(model output)` uncorrected, leaving ~8% relative RMSPE on the table
  for free. Fit the factor on held-out data the model didn't train on, never on the
  evaluation fold.
- **Mini-batch SGD validation curves are noisier than tree capacity curves.** Phase 3's
  RF/HGB sweeps (over `n_estimators`/`max_iter`) were smooth enough to read off a single
  hyperparameter value by eye. The Phase 4 neural net's per-epoch validation RMSPE
  oscillates instead of settling, so picking one epoch count that way would be
  arbitrary; the fix is to train for a generous fixed epoch budget and keep the
  best-validation-RMSPE checkpoint rather than the final epoch's weights.

## Phase 0 results (EDA)

`rossmann_eda.ipynb`. `train.csv` (2013-01-01 to 2015-07-31, 1,115 stores) has no
nulls/duplicates; `StateHoliday` is explicitly cast to `str` on read. `store.csv`
missingness is structural: `Promo2SinceWeek/Year`/`PromoInterval` are null in exactly
the 544 rows where `Promo2==0`, not randomly. **181 stores share an identical ~6-month
reporting gap** (2014-07-01 to 2014-12-31) — see Pitfalls above.

**Store identity dominates**: per-store mean sales range from 2,704 to 21,757 — bigger
than any single categorical effect measured. `Promo` gives a clean, direct ~39% lift
(8,228 vs 5,929 mean sales), decomposable into +21% customers and +14% basket size
(`Customers` itself is exploration-only, not a model input — see Pitfalls). December
sales spike ~25-30% above baseline (Christmas, clearest once the daily chart is
normalized by open-store count to remove a weekly sawtooth artifact), plus a mild
year-over-year upward trend. `Sales` is right-skewed (skew 1.59) vs. `log1p(Sales)`
near-symmetric (skew -0.64), supporting a log-scale target later.

**Confounds flagged, not trusted as causal**: only 33 of 1,115 stores ever open on
Sunday, and those same stores also out-sell the rest on weekdays (8,759 vs 7,121) — so
"Sunday sells more" is a store-identity effect, verified with numbers rather than taken
from the chart alone. All 17 `StoreType b` stores (a distinct, higher-traffic business
model) account for half of those 33. The same static-confound logic applies to
`CompetitionDistance` (closest-competitor quintile sells *more*, likely a marker of
busy urban locations, not competition helping) — though a separate before/after check
around actual competitor-*opening* events shows a ~3.6% relative sales decline versus
the organic trend, which looks like a genuine (if roughly measured) competitive effect,
unlike the static distance comparison.

## Phase 1 results (validation & baselines)

`rossmann_baseline.ipynb`. Holdout: last 42 days of `train.csv` (2015-06-20 to
2015-07-31, chosen from the assignment's stated "~six weeks" rather than by inspecting
`test.csv`), all 1,115 stores present on both sides.

| baseline | RMSPE |
|---|---|
| Global mean | 0.583 |
| Per-store mean | 0.330 |
| + DayOfWeek | 0.236 |
| **+ Promo (best)** | **0.147** |
| Same day last year (364-day lag) | 0.160 |

Baselines improve monotonically exactly as EDA predicted — store identity and Promo are
the two strongest simple signals. The 364-day-lag baseline underperforming the plain
averaged baseline was the one surprise (see Pitfalls above); its 25% coverage miss splits
roughly evenly between stores closed 364 days back and panel-gap-affected dates, but even
on genuine matches only, RMSPE is still ~0.161 — the noise is inherent to a single
historical data point, not just a coverage artifact. A flat 2% shrinkage of the best
baseline's arithmetic mean improves RMSPE to 0.1448, a concrete demonstration that RMSPE
isn't minimized by the same statistic as squared error (motivating `log1p(Sales)` from
Phase 2 onward). Error by `StoreType` is fairly flat (0.133–0.154), but by `DayOfWeek` it
ranges from 0.127 (Thursday) to 0.221 (Sunday) — likely tied to Sunday's much smaller
training sample. Split config, results, and row-level validation predictions are saved
(`phase1_split_config.json`, `phase1_baseline_results.csv`, `phase1_val_predictions.csv`)
so later phases reuse the identical holdout rather than re-deriving it.

**Bar to beat in Phase 2+: RMSPE 0.147** (0.1448 with the shrinkage refinement).

## Phase 2 results (feature engineering + linear model)

`rossmann_linear.ipynb`. Same holdout as Phase 1 (reloaded from `phase1_split_config.json`).
Feature logic lives in a shared `rossmann_features.py` module (fit/transform discipline —
fit on `train_fold` only) so Phase 3 and the final `test.csv` scoring reuse it unchanged.
Date-derived, promo, competition, and store features only — no lag/rolling yet. Ridge
regression (alpha=10, chosen from a flat plateau in an internal time-based CV, not by
tuning against `val_fold`) on `log1p(Sales)`, fit on `Open==1 & Sales>0`.

| model | RMSPE |
|---|---|
| Additive (main effects only) | 0.214 |
| **+ `Store x Promo` interaction (Phase 2 deliverable)** | **0.199** |
| + `StoreType x DayOfWeek` (27 extra columns, EDA-informed) | 0.185 |
| + full `Store x DayOfWeek x Promo` interaction (diagnostic) | 0.150 |
| Phase 1: `store_x_dow_x_promo_mean` | 0.147 |
| Phase 1: + shrinkage refinement | 0.1448 |

An ablation, not just a single model: the additive model underperforms Phase 1, as
expected, since it forces every store to share the same Promo lift and weekly shape.
Adding an explicit `Store x Promo` interaction recovers about a third of the gap. Adding
the *full* three-way interaction is structurally the same model as Phase 1's lookup
table and lands close to it (0.150 vs 0.1468/0.1448 — not exact, because only 12,298 of
15,610 possible combinations were seen in training, and fitting in log-space reconstructs
something closer to each cell's geometric mean than Phase 1's arithmetic mean). The
ablation traces almost the entire Phase 1 gap back to the missing interactions, rather
than asserting it.

Coefficients on time-varying features (`Promo`, `Month`, `DayOfWeek`, `Trend`,
`CompetitionOpen`) match EDA exactly — including a clean, independently-derived
confirmation that competition hurts sales once store identity is held fixed (the
`CompetitionOpen` coefficient is negative). Coefficients on store-*constant* features
(`StoreType`, `Assortment`, `LogCompetitionDistance`) aren't reliably interpretable,
since they're exactly collinear with the 1,115-column Store dummy set.

Sharpest failure mode: the deliverable model's Sunday RMSPE is **0.586** — more than 3x
every weekday — a stark, concrete demonstration that "Sunday" isn't a real day-of-week
effect but a store-identity effect (per EDA), which a model without a `Store x
DayOfWeek` term simply can't represent. A cheap, EDA-informed fix — `StoreType x
DayOfWeek`, just 27 columns, motivated directly by "`StoreType b` = the Sunday-openers"
— cuts Sunday RMSPE to 0.330 and overall RMSPE to 0.185, most of the full interaction's
benefit at a fraction of the cost. Fixing the rest without hand-crafting the interaction
is the real test for Phase 3's tree ensembles, not just beating 0.147.

The Ridge regularization path (train vs. inner-validation RMSPE across alpha) is flat
across 5 orders of magnitude — traced to `StandardScaler(with_mean=False)` making the
effective L2 penalty on Store dummies ~74 million times weaker than on `Trend`, so store
fixed effects stay close to unregularized (correct for a fixed-effects term) until alpha
is extreme. Train RMSPE sitting *above* inner-validation RMSPE (0.288 vs 0.216) is a
real, explained finding: a handful of extreme near-zero-Sales anomaly days survive the
`Sales>0` fit filter and dominate RMSPE's squared-percentage-error sum out of proportion
to their count — a generalizable caveat about this metric, not a bug.

Results and row-level predictions saved (`phase2_linear_results.csv`,
`phase2_val_predictions.csv`).

## Phase 3 results (tree ensembles + lag features)

`rossmann_trees.ipynb`. Same holdout as Phases 1-2. Leakage-safe lag/rolling features
added via `rossmann_features.py` (frozen-cutoff rolling means + `HorizonDay` for the
validation batch, per-row rolling means for training rows — see Pitfalls). Three models
on the identical final feature set, for a clean two-step attribution: Ridge (linear +
lags, one-hot `Store`), Random Forest, and `HistGradientBoostingRegressor` (both trees
using target-encoded `StoreMeanLogSales` in place of raw `Store` — see Pitfalls).

| model | RMSPE |
|---|---|
| Phase 2: additive + Store x Promo | 0.199 |
| Phase 2: + StoreType x DayOfWeek | 0.185 |
| Ridge + lag features | 0.176 |
| Random Forest | 0.146 |
| Phase 1: store_x_dow_x_promo_mean | 0.1468 |
| Phase 1: + shrinkage refinement | 0.1448 |
| **HistGradientBoostingRegressor (best overall)** | **0.141** |

**Two-step attribution confirmed**: Ridge+lags (0.176) beats both Phase 2 linear
results — lag information alone has real, independent value, isolated from any
interaction-handling difference. The best tree (HGB, 0.141) beats Ridge+lags by a
wider margin on top of that — automatic interaction-handling adds further value beyond
what lags alone provide. Both trees beat every prior phase; HGB is the best model of
the project so far, though only by ~4% relative RMSPE over Phase 1's shrinkage-refined
baseline despite substantial feature and model engineering in between — the
motivation for testing Phase 4 for real rather than treating it as an optional stretch.

**Headline result, cleanly confirmed**: Phase 2's Sunday failure mode (0.586 RMSPE) is
fixed by both trees via `StoreMeanLogSales x DayOfWeek` interactions learned
automatically from splits — Random Forest 0.183, HGB 0.159 (no longer even the worst
day; Saturday is worse). Ridge+lags improves Sunday to 0.426 but doesn't fix it,
confirming the fix specifically needed store x day-of-week interaction capacity, not
just more features.

Fit times were fast: RF ~50s (200 trees, constrained `max_depth=16`/
`min_samples_leaf=15`/`max_samples=0.25` to keep an 804k-row fit reasonable), HGB ~12s
(700 iterations) — both comfortable on a standard CPU. Train-vs-validation capacity
curves: Ridge and RF stay essentially flat (no overfitting risk in the range explored),
while HGB shows a genuine, clean overfitting signature past ~700 iterations. Train
RMSPE sitting above validation RMSPE recurs from Phase 2, for the same reason (rare
near-zero-Sales anomaly days dominating the training-period RMSPE). `StoreMeanLogSales`
is the dominant feature for both trees, followed by the rolling/lag features.

Results and row-level predictions saved (`phase3_trees_results.csv`,
`phase3_val_predictions.csv`).

**Bar to beat in Phase 4: RMSPE 0.141.**

## Phase 4 results (entity embeddings + MLP)

`rossmann_nn_sota.ipynb`. Same holdout and identical lag/rolling feature set as Phase 3
(reused via `rossmann_features.py`), so the comparison isolates the modeling paradigm,
not the inputs. Instead of one-hot (Ridge) or target-encoding (trees), `Store` and the
other categoricals get their own learned embedding tables (`Store`: 20 dims; small
categoricals: 2-6 dims) concatenated with batch-normalized continuous features, feeding
a 3-layer MLP (256/128/64 units, `Linear -> BatchNorm1d -> ReLU -> Dropout`) trained on
GPU with Adam and a plateau-based learning-rate schedule. Since the per-epoch validation
curve is noisier than Phase 3's tree capacity curves (see Pitfalls), the model trains
for a fixed 40-epoch budget and keeps the best-validation-RMSPE checkpoint rather than
the final epoch's weights.

| model | RMSPE |
|---|---|
| Phase 2: additive + Store x Promo | 0.199 |
| Phase 2: + StoreType x DayOfWeek | 0.185 |
| Ridge + lag features | 0.176 |
| Random Forest | 0.146 |
| Phase 1: store_x_dow_x_promo_mean | 0.1468 |
| Phase 1: + shrinkage refinement | 0.1448 |
| HistGradientBoostingRegressor | 0.1412 |
| **Entity-embedding MLP** | **0.1412** |

**Headline result: an exact tie, not an improvement.** The entity-embedding MLP matches
HGB's RMSPE to four decimal places on the real held-out `val_fold`. Given identical
features and split, a fundamentally different modeling paradigm converges to the same
performance ceiling rather than breaking past it — a legitimate, informative result in
its own right, not a failure to report around.

The per-day breakdown is more nuanced than the tied headline number suggests: the MLP's
Sunday RMSPE (0.174) is worse than HGB's (0.159), and its actual worst day is Friday
(0.187) — a different failure pattern than every tree/linear model so far, even though
the two models land on the same overall RMSPE.

A direct check of whether the network rediscovered `StoreType` structure on its own —
a silhouette score on the full 20-dimensional `Store` embedding, grouped by `StoreType`
— came back slightly negative (-0.04), meaning no meaningful clustering by store type.
A plausible explanation is that the rich lag/rolling sales history already carries most
of each store's identity signal, leaving the embedding to encode residual variation
rather than store type itself, but this is an open observation, not a settled one.

Given entity embeddings + MLP match but don't beat gradient boosting here, and Phase 3's
trees already closed most of the gap over Phase 1's baseline, the practical case for
preferring this architecture over HGB — given its added engineering and training-time
cost — is weak for this dataset and feature set.

Results and row-level predictions saved (`phase4_nn_results.csv`,
`phase4_val_predictions.csv`).

**Bar to beat in Phase 5: RMSPE 0.1412** (unchanged from Phase 3 — HGB remains the
project's best model).

## Phase 4.5 results (calibration + ensembling)

`rossmann_calibration.ipynb`. Two things the project already had evidence for but never
applied: RMSPE-oriented output shrinkage (Phase 1 found it, Phases 2-4 dropped it) and
ensembling (Phase 3's HGB and Phase 4's MLP score identically but fail on different
days). Both constants are fitted on the **inner** split — all four models retrained on
`inner_train`, fitted against `inner_val` — then frozen and applied unchanged to the
saved Phase 3/4 validation predictions. Nothing is tuned against `val_fold`.

| model | val RMSPE raw | + calibration | factor |
|---|---|---|---|
| Ridge + lags | 0.1761 | 0.1668 | 0.964 |
| Random Forest | 0.1460 | 0.1340 | 0.954 |
| HistGradientBoosting | 0.1412 | 0.1301 | 0.951 |
| Entity-embedding MLP | 0.1412 | 0.1328 | 0.961 |
| **Weighted ensemble + calibration** | — | **0.1257** | 0.961 |

**Calibration alone is worth more than Phases 2-4 combined.** Every model wants a shrink
factor of 0.95-0.96, and applying it moves HGB from 0.1412 to 0.1301 — an 8% relative
gain from one scalar. The consistency of that factor across four structurally unrelated
model families is the strongest evidence that it reflects the metric's geometry rather
than any one model's quirk.

**Ensembling adds a further 3.4%**, landing at 0.1257: a **13.2% relative improvement
over the Phase 1 baseline**, versus the 2.5% that Phases 2-4 had managed. Fitted weights
are MLP 0.50, HGB 0.30, RF 0.15, Ridge 0.05 — notably, Ridge earns a non-zero weight
despite being by far the weakest model (0.176), because its errors are decorrelated
enough from the others to still contribute.

**The honest-vs-cheating gap is 0.0003 RMSPE.** Fitting the constants on the inner split
rather than on `val_fold` cost essentially nothing: inner-fitted factors land within
0.004-0.022 of what `val_fold` would have chosen, and all of them err on the conservative
side (less shrinkage), exactly as predicted from the mild circularity of reusing Phase 3's
inner-selected hyperparameters. This is the check that makes the result trustworthy rather
than just good-looking.

One caveat worth carrying into Phase 5: the ensemble is *worse* than raw HGB on Friday
(0.149 vs 0.146) and Sunday (0.164 vs 0.159), because the MLP holds the largest weight and
those are precisely its weak days. A day-of-week-conditional weighting is the obvious next
idea, though with real overfitting risk given how few Sundays the data contains.

Results, predictions, and the frozen constants are saved (`phase45_calibration_results.csv`,
`phase45_val_predictions.csv`, `phase45_calibration_params.json`) so Phase 5 applies the
identical transformation to `test.csv` rather than re-deriving it.

**Bar to beat: RMSPE 0.1257.**

## Phase 4.6 results (feature rebuild)

`rossmann_features_v2.py` + `rossmann_features_v2.ipynb`. v2 is a separate module rather
than an edit to `rossmann_features.py`, so Phases 0-4.5 stay exactly reproducible. Run as
a one-change-at-a-time ablation with HGB as a fixed probe (Phase 3's hyperparameters, ~15s
per fit), so each fix is attributed rather than bundled. All numbers are **uncalibrated
single-model** — Phase 4.5's calibration and ensembling stack on top and are not included.

| feature set | val RMSPE | gain vs. control |
|---|---|---|
| A: v1 features (control) | 0.1412 | — |
| B: + train-like-you-serve horizons | 0.1356 | 4.0% |
| C: + `SalesYearAgo` fallback flag | 0.1358 | 3.8% |
| D: + known-schedule features | 0.1293 | 8.5% |
| **E: − `StoreMeanLogSales` (best)** | **0.1281** | **9.3%** |

The control reproduces Phase 3's 0.1412 exactly, so the ablation is measuring feature
changes and nothing else.

**The horizon fix works, but not by the mechanism intended.** Training on stale features
is worth 4% — yet `HorizonDay` itself still measures no useful importance even after it
starts varying. The gain comes from the training distribution matching the serving
distribution, not from the model reading a staleness signal. The clearest evidence is the
spread of RMSPE across horizon weeks, which falls monotonically at every step: 0.0566
(control) → 0.0418 → 0.0327 → 0.0317. The v1 control was the *best* model in forecast week
1 (0.108) and fell apart afterwards; v2 gives up a little in week 1 and is far steadier
across the full six weeks, which is what a 6-week-ahead forecast actually needs.

**Known-schedule features were the single biggest win** (D, +4.7% on top of B). Three of
the top five features by importance are new: `DayOfMonth` (payday effects),
`WasClosedYesterday` (the pre/post-closure sales spike), and `DaysSincePromo`. The
reopening detector fires on **180 stores immediately after the 181-store blackout** the
EDA identified back in Phase 0 — a finding that sat unused for four phases.

**Two features turned out to be substitutes.** Dropping `StoreMeanLogSales` *improves*
the model (0.1293 → 0.1281): the Phase 3 target encoding was only ever a workaround for
HGB's categorical-cardinality cap, and it is now redundant given the rolling, trend, and
schedule features around it. But dropping both it and `SalesYearAgo` is far worse (0.1436)
— the model needs one store-level summary, not two.

Two of the ideas that motivated this phase did **not** survive contact with the data: the
`SalesYearAgo` fallback flag was neutral (0.1358 vs 0.1356), and dropping `SalesYearAgo`
on the strength of its negative permutation importance made things worse. Both are written
up in Pitfalls above, since the reasoning errors are more reusable than the fixes.

Saved: `phase46_ablation_results.csv`, `phase46_val_predictions.csv`,
`phase46_feature_set.json` (a record of the winning column list; Phase 4.7 rebuilds the
same list in code rather than reading this file).

**Bar to beat in Phase 4.7: RMSPE 0.1281 uncalibrated / 0.1257 calibrated+ensembled.**

## Phase 4.7 results (LightGBM/CatBoost, tuned)

`rossmann_gbm_tuned.ipynb`. Same v2 features and same split. Every choice — loss
weighting, hyperparameters, `Store` representation, ensemble weights, shrink factor — is
made on the **inner** split; `val_fold` is scored once at the end.

| model | val RMSPE |
|---|---|
| Phase 1: baseline + shrinkage | 0.1448 |
| Phase 3: HGB (v1 features) | 0.1412 |
| Phase 4.5: calibrated ensemble (v1) | 0.1257 |
| Phase 4.6: HGB (v2 features) | 0.1281 |
| Phase 4.7: LightGBM tuned | 0.1111 |
| **Phase 4.7: calibrated ensemble (deliverable)** | **0.1107** |

**0.1107 is a 23.5% relative improvement on the Phase 1 baseline**, and 11.9% on Phase
4.5. Three separate contributions, each measured on the inner split:

- ***`Store` as a native categorical: the biggest single win.*** 0.1125 target-encoded →
  0.1038 native (see Pitfalls). Adding the target encoding back on top changes nothing
  (0.1035), and dropping store identity entirely costs 9% (0.1134).
- **Hyperparameter tuning: ~4%.** The winning config is `learning_rate=0.02`,
  `num_leaves=255`, `min_child_samples=50`, 0.8 feature/bagging fractions, `lambda_l2=10`,
  1,855 iterations — versus the `learning_rate=0.1` / 63-leaf default every prior phase
  used without ever questioning it.
- **Loss weighting: ~1.5%.** Sweeping `w = 1/log1p(y)^p` picked `p=1` (0.1094 vs 0.1111
  unweighted). Applying RMSPE's literal `w=1/y²` to the raw target scores 0.1579 — far
  worse — confirming on the inner split what the earlier diagnostic showed: matching the
  loss to the metric does not rescue a bad target parameterization.

**Ensembling has stopped paying.** Fitted weights are LightGBM 0.75 / CatBoost 0.25 /
HGB 0.00 — HGB is dropped entirely. The ensemble wins on the inner split (0.1034 vs
LightGBM's 0.1043) but on `val_fold` it lands at 0.1107 against 0.1104 for calibrated
LightGBM alone. That difference is well inside noise, and the ensemble stays the
deliverable because it is what the inner split selected; switching to the single model
now, *because* it happened to win on the evaluation fold, is exactly the discipline
failure this project has avoided throughout. Ensembling was worth 3.4% in Phase 4.5 and
~0% here — as one model pulls clear, its rivals stop adding independent information.

CatBoost is worth keeping only as a diversity member: 0.1194 standalone against
LightGBM's 0.1111, at roughly 2x the fit time.

**`SalesYearAgo` is now the top feature by a wide margin** (0.51 of total gain, ahead of
`SalesRollMean90` at 0.19 and `Store` at 0.05) — a final rebuttal to the Phase 4.6 review
that proposed deleting it on the strength of a negative permutation importance.

Sunday remains the hardest day (0.147) but is far better than Phase 2's 0.586, and the
horizon spread stays flat at 0.030, holding Phase 4.6's robustness gain.

Saved: `phase47_results.csv`, `phase47_val_predictions.csv`, `phase47_lgb_tuning.csv`,
`phase47_params.json`.

**Bar to beat in Phase 5: RMSPE 0.1107.**

## Phase 5 results (final scoring + error analysis)

`rossmann_final.ipynb`. The only phase that touches `test.csv`. The pipeline is frozen —
hyperparameters, ensemble weights, and shrink factor all come from `phase47_params.json`,
fitted on the inner split in Phase 4.7. Two things legitimately changed once selection was
over: the final model trains on **all** of `train.csv` (the holdout existed to make choices
honest, and those choices were made), and training origins are re-tiled at **48 days** to
match `test.csv`'s actual horizon rather than the 42 used during development.

### Final comparison on the shared holdout

| model | RMSPE |
|---|---|
| Phase 1 baseline (store × DoW × promo lookup) | 0.1468 |
| Phase 3 HGB | 0.1412 |
| Phase 4 entity-embedding MLP | 0.1412 |
| Phase 4.5 calibrated ensemble | 0.1257 |
| **Phase 4.7 calibrated ensemble (shipped)** | **0.1107** |

**23.5% better than the baseline.** The honest summary is that most of the gain came from
things that were not "better models": matching the training feature distribution to the
serving one, letting `Store` be a real categorical, tuning the booster, and calibrating for
the metric. The two most sophisticated things attempted — entity embeddings and a neural
network — tied the gradient booster exactly and were dropped from the final ensemble by a
weight search that gave them 0.00.

### Where the error actually lives

**The worst 1% of rows carry 25% of total squared percentage error; the worst 10% carry
59%.** Median per-store RMSPE is 0.094 against a worst store of 1.196 — the aggregate badly
misrepresents typical performance. The tail is not a size effect (hardest 50 stores average
6,662 mean sales vs 6,934 overall); it is dominated by **regime changes**. The worst store
(292) had 9 open days in July 2015 against ~25 in prior months — a store winding down, which
no forecast built on its own sales history can anticipate.

Remaining systematic weaknesses: Sunday 0.147 (only 33 stores ever open, so little to learn
from), Saturday 0.134, and forecast week 3 at 0.129 versus 0.101 in week 1. Promo days are
*easier* than non-promo days (0.105 vs 0.115) — a promo is a known, scheduled, high-signal
event.

### Submission

`submission.csv`, 41,088 rows. All checks pass: Ids match `sample_submission.csv` exactly,
no NaN or negative values, exactly zero on all closed days and non-zero on all open ones.
Predicted distribution tracks history closely (mean 6,908 vs 6,956; median 6,369 vs 6,369)
with slightly compressed tails (std 2,784 vs 3,104), as expected from a model trained on
squared error.

The main defect carried into that submission is the right-censored `DaysUntil*` features
described in Pitfalls — found by a train-vs-test distribution check during final scoring,
and deliberately **not** fixed here, since re-tuning against the test set at Phase 5 would
have discarded the discipline the whole project rests on. It is the first thing to fix next.

Saved: `submission.csv`, `phase5_error_analysis.csv`, `phase5_per_store_rmspe.csv`.

## Data and running

**Data.** The data comes from the Kaggle competition
[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) and is not
redistributed here. Accept the competition rules on Kaggle, then put the four CSV files in
this directory, either by downloading them from the competition's Data page in a browser or
with the Kaggle CLI:

```bash
kaggle competitions download -c rossmann-store-sales && unzip rossmann-store-sales.zip
```

The Kaggle CLI is not in `requirements.txt`: it is installed separately (`pip install kaggle`)
and needs a configured Kaggle API token.

**Environment.** Python 3.12 with the pinned top-level dependencies in `requirements.txt`
(`pip install -r requirements.txt`). It is not a complete lockfile, so transitive versions may
differ from the environment that produced the results.

**GPU.** The complete run requires a CUDA-capable GPU: `rossmann_gbm_tuned` and
`rossmann_final` fit CatBoost with `task_type="GPU"`. On a CPU-only machine, change those
calls to `task_type="CPU"`, and expect different timing and possibly different results. The
PyTorch model selects the GPU when one is available and otherwise runs on the CPU.

**Order.** Later notebooks read artifacts saved by earlier phases (`phase*_*.csv` / `.json`,
written to this directory and not tracked), so the safest complete run order is:
`rossmann_eda`, `rossmann_baseline`, `rossmann_linear`, `rossmann_trees`,
`rossmann_nn_sota`, `rossmann_calibration`, `rossmann_features_v2`, `rossmann_gbm_tuned`,
`rossmann_final`. Only `rossmann_final` reads `test.csv`.

**Repository notes.**

- The charts in the notebooks were interactive Plotly figures. In this repository each one is
  stored as a static PNG render of the same figure, so that it displays on GitHub; the code,
  the text and every other output are unchanged. Re-running a notebook restores the
  interactive version.
- `Initial_Plan.md` is kept as the original planning record, so its link to the assignment
  brief points to a file that is not included.
