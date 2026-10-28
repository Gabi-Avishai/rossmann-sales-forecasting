"""Shared, leakage-safe feature engineering for the Rossmann project.

Used by rossmann_linear.ipynb (Phase 2) and intended for reuse by the Phase 3 tree
notebooks and the final Phase 5 test.csv scoring, so the exact same logic is applied
everywhere rather than copy-pasted and risking drift.

Two-stage design, matching sklearn's fit/transform discipline:
  1. `prepare_store_table(store)` -- pure, stateless derivation from store.csv (no
     fitting involved: medians/dates are properties of the raw table, not learned).
  2. `add_features(df, store_prepared)` -- merges those onto a transactions frame
     (train_fold or val_fold or test.csv) and adds date-derived columns.
Encoding (OneHotEncoder/StandardScaler) is deliberately left to the caller's sklearn
Pipeline, which must be fit on train_fold only.
"""

import numpy as np
import pandas as pd

MONTH_ABBR_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
# store.csv fixed reference point for a continuous time trend feature -- must be a
# fixed constant (not derived from any fold), so it's identical across every notebook.
TREND_EPOCH = pd.Timestamp("2013-01-01")

CATEGORICAL_COLS = ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"]
NUMERIC_COLS = ["Promo", "SchoolHoliday", "CompetitionOpen", "IsPromo2Active", "Trend", "LogCompetitionDistance"]


def prepare_store_table(store):
    """Derive store-level fields once. Pure function of store.csv -- no fitting."""
    store = store.copy()

    has_comp_date = store["CompetitionOpenSinceYear"].notna() & store["CompetitionOpenSinceMonth"].notna()
    store["CompetitionOpenDate"] = pd.NaT
    store.loc[has_comp_date, "CompetitionOpenDate"] = pd.to_datetime(
        store.loc[has_comp_date, "CompetitionOpenSinceYear"].astype(int).astype(str) + "-" +
        store.loc[has_comp_date, "CompetitionOpenSinceMonth"].astype(int).astype(str) + "-01")
    # Missing CompetitionOpenSince* -> treated as "competitor already open before the
    # data window" (the common reading of this gap), not "no competitor" -- that's
    # what CompetitionDistance itself (only 3 missing) indicates instead.

    # The 3 stores with NaN CompetitionDistance genuinely have no nearby competitor at
    # all -- distinct from "distance unknown". Track that explicitly (`HasCompetitor`)
    # rather than letting median-imputed distance + the has_comp_date fallback above
    # combine into "a competitor at a typical distance, already open", which would be
    # wrong for these 3 stores.
    store["HasCompetitor"] = store["CompetitionDistance"].notna()
    dist_sentinel = store["CompetitionDistance"].max() * 2  # "no competitor" ~ unboundedly far
    store["LogCompetitionDistance"] = np.log1p(store["CompetitionDistance"].fillna(dist_sentinel))

    has_promo2_date = store["Promo2SinceYear"].notna() & store["Promo2SinceWeek"].notna()
    store["Promo2SinceDate"] = pd.NaT
    store.loc[has_promo2_date, "Promo2SinceDate"] = pd.to_datetime(
        store.loc[has_promo2_date, "Promo2SinceYear"].astype(int).astype(str) + "-" +
        store.loc[has_promo2_date, "Promo2SinceWeek"].astype(int).astype(str) + "-1",
        format="%Y-%W-%w")

    # PromoInterval has a real typo in the source data: "Mar,Jun,Sept,Dec" (106 rows)
    # uses "Sept" where every other row uses 3-letter abbreviations ("Sep"). The three
    # non-null patterns are exactly the three residue classes mod 3 (Jan/Apr/Jul/Oct
    # = 1 mod 3, Feb/May/Aug/Nov = 2 mod 3, Mar/Jun/Sep/Dec = 0 mod 3), so instead of
    # parsing and matching a list of month names per row, each store just gets a
    # single "PromoMonthMod" and month-matching becomes a vectorized `Month % 3`
    # comparison -- correct by construction and doesn't care about the "Sept" typo.
    first_month = store["PromoInterval"].str.replace("Sept", "Sep").str.split(",").str[0]
    store["PromoMonthMod"] = first_month.map(MONTH_ABBR_TO_NUM) % 3

    return store


def add_features(df, store_prepared):
    """Merge store-level fields onto a transactions frame and add date-derived columns.
    `df` must have Store, Date, DayOfWeek, Promo, SchoolHoliday, StateHoliday, Open."""
    df = df.merge(
        store_prepared[["Store", "StoreType", "Assortment", "CompetitionOpenDate", "HasCompetitor",
                         "LogCompetitionDistance", "Promo2", "Promo2SinceDate", "PromoMonthMod"]],
        on="Store", how="left")

    # test.csv has a handful of NaN Open values; default to "assume open" (the far more
    # common state) so the zero-prediction-on-closed-days step downstream doesn't
    # silently skip these rows for lack of an exact `== 0` match against NaN.
    df["Open"] = df["Open"].fillna(1).astype(int)

    df["Month"] = df["Date"].dt.month
    df["Trend"] = (df["Date"] - TREND_EPOCH).dt.days

    df["CompetitionOpen"] = (
        df["HasCompetitor"] & (df["CompetitionOpenDate"].isna() | (df["Date"] >= df["CompetitionOpenDate"]))
    ).astype(int)

    is_promo_month = df["Month"] % 3 == df["PromoMonthMod"]
    df["IsPromo2Active"] = (
        (df["Promo2"] == 1) & (df["Date"] >= df["Promo2SinceDate"]) & is_promo_month
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Phase 3: lag / rolling / target-encoded features.
#
# Two different modes for the rolling means, because they answer different
# questions:
#   - Training rows are fit one at a time against their OWN history, so each row
#     uses a per-row "as of the day before" rolling window (`add_rolling_features_training`).
#   - A validation/test BATCH is a single 6-week-ahead forecast made from one data
#     cutoff, not a fresh cutoff per day -- so every row in that batch must reuse the
#     SAME frozen rolling window computed as of `train_fold`'s last date
#     (`add_rolling_features_frozen`). Computing it per-row instead would leak data a
#     real 6-week-ahead forecast would never have.
# This leaves a real train/inference mismatch: training rows always see "yesterday's"
# rolling mean, while validation/test rows see one up to 42 days stale.
#
# KNOWN DEFECT: `HorizonDay` was intended to signal that staleness, but it cannot as
# written -- the training variant hard-codes it to 0, so it is constant across every
# training row and the model can learn nothing from it, while at inference it takes
# values 1..42 that were never seen during fitting. Measured importance is exactly 0.0
# for both tree models. The fix is to build training rows from simulated forecast
# origins so HorizonDay genuinely varies; tracked as Phase 4.6.
ROLLING_WINDOWS = (7, 30, 90)


def add_rolling_features_training(df, windows=ROLLING_WINDOWS):
    """Per-row trailing rolling means, strictly excluding the row's own day.
    `closed="left"` makes a date-offset window include [date-w, date) directly --
    cleaner and less error-prone than shifting first and rolling second, which
    doesn't actually exclude the current row the way it looks like it should.
    `.transform()` (not `.apply()`) guarantees the result aligns back to df's
    original row order regardless of pandas version quirks with grouped rolling."""
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["HorizonDay"] = 0  # by definition -- these rows are "day 0" of their own history
    dated = df.set_index("Date")
    for col, out_prefix in [("Sales", "SalesRollMean"), ("Customers", "CustomersRollMean")]:
        for w in windows:
            df[f"{out_prefix}{w}"] = dated.groupby("Store")[col].transform(
                lambda s, w=w: s.rolling(f"{w}D", closed="left").mean()
            ).to_numpy()
    return _fill_rolling_fallback(df, windows)


def add_rolling_features_frozen(df, history_open, cutoff_date, windows=ROLLING_WINDOWS):
    """Rolling means frozen at a single `cutoff_date` for every row -- for a
    validation/test batch that's all forecast from one data cutoff."""
    df = df.copy()
    df["HorizonDay"] = (df["Date"] - cutoff_date).dt.days
    for w in windows:
        window_start = cutoff_date - pd.Timedelta(days=w - 1)
        hist_w = history_open[(history_open["Date"] >= window_start) & (history_open["Date"] <= cutoff_date)]
        agg = hist_w.groupby("Store").agg(**{
            f"SalesRollMean{w}": ("Sales", "mean"),
            f"CustomersRollMean{w}": ("Customers", "mean"),
        })
        df = df.merge(agg, on="Store", how="left")
    return _fill_rolling_fallback(df, windows, fallback_source=history_open)


def _fill_rolling_fallback(df, windows, fallback_source=None):
    """Fill any window with no observations (store closed the whole window, or a
    training row too early to have history yet) with that store's overall mean."""
    source = fallback_source if fallback_source is not None else df
    sales_mean = source.groupby("Store")["Sales"].mean()
    cust_mean = source.groupby("Store")["Customers"].mean()
    for w in windows:
        df[f"SalesRollMean{w}"] = df[f"SalesRollMean{w}"].fillna(df["Store"].map(sales_mean))
        df[f"CustomersRollMean{w}"] = df[f"CustomersRollMean{w}"].fillna(df["Store"].map(cust_mean))
    return df


def add_yearago_lag(df, history_open, fallback):
    """Sales exactly 364 days before each row's own date -- always genuinely in the
    past regardless of forecast horizon, so no frozen-cutoff issue here (unlike the
    short rolling windows above). `fallback` (a plain array in df's row order) covers
    rows with no exact match: before the data starts, or inside the 181-store panel
    gap. Uses np.where on plain arrays rather than Series.fillna, since `.merge()`
    resets to a fresh RangeIndex -- combining it with a Series built on df's original
    (pre-merge) index would silently misalign via pandas' index-based alignment."""
    lookup = history_open[["Store", "Date", "Sales"]].rename(columns={"Sales": "SalesYearAgo"})
    shifted = df[["Store", "Date"]].copy()
    shifted["Date"] = shifted["Date"] - pd.Timedelta(days=364)
    merged = shifted.merge(lookup, on=["Store", "Date"], how="left")
    df = df.copy()
    df["SalesYearAgo"] = np.where(merged["SalesYearAgo"].notna(), merged["SalesYearAgo"], np.asarray(fallback))
    return df


def fit_store_dow_promo_mean(history_open):
    """The Phase 1 baseline lookup table, reused here as a fallback predictor for
    `add_yearago_lag` (and as a fallback in general -- it's a much better-conditioned
    guess than a global mean)."""
    return history_open.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean()


def predict_store_dow_promo_mean(df, table, fallback):
    key = pd.MultiIndex.from_frame(df[["Store", "DayOfWeek", "Promo"]])
    return table.reindex(key).fillna(fallback).to_numpy()


def fit_store_target_encoding(history_open):
    """Per-store mean of log1p(Sales) -- lets tree models use store identity as a
    single well-conditioned continuous feature instead of a 1,115-way categorical
    (HistGradientBoostingRegressor caps native categorical cardinality at 255, and
    leaving Store as a raw ordinal integer forces meaningless <= splits on arbitrary
    adjacent store IDs for every tree model, not just HGB)."""
    return np.log1p(history_open["Sales"]).groupby(history_open["Store"]).mean()


def add_store_target_encoding(df, encoding, fallback):
    df = df.copy()
    df["StoreMeanLogSales"] = df["Store"].map(encoding).fillna(fallback)
    return df
