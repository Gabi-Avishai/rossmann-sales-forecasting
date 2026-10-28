"""Phase 4.6 feature engineering — v2.

Kept separate from `rossmann_features.py` rather than editing it in place, so Phases 0-4.5
stay exactly reproducible. v1 is still imported for the parts that were already correct
(store table prep, base date/promo/competition features, target encoding).

Three changes over v1, in order of expected impact:

1. TRAIN LIKE YOU SERVE. v1 gave training rows a per-row "as of yesterday" rolling window
   while validation/test rows got one frozen at a single cutoff and up to 42 days stale.
   The model therefore learned to trust a feature far fresher than the one it is served,
   and `HorizonDay` -- meant to signal that staleness -- was hard-coded to 0 for every
   training row, making it constant and unlearnable (measured importance exactly 0.0).
   v2 computes rolling stats "as of an origin date" for BOTH cases: training rows are
   tiled into synthetic 42-day forecast blocks, each with its own origin, so HorizonDay
   genuinely varies over 1..42 exactly as it does at inference.

2. SPLIT `SalesYearAgo`. v1 packed two different quantities into one column: a real
   364-day-ago observation (87% of rows) and, where that was missing, the Phase 1 baseline
   lookup prediction. Nothing told the model which it was looking at. Phase 3 measured the
   consequence: RF spent 78% of its importance on this column while HGB's permutation
   importance for it was -0.185, the most negative of any feature. v2 adds an explicit
   fallback flag. (Smoothing the lag over neighbouring weeks was tried and rejected -- see
   `add_yearago_features` for why the exact 364-day lag is the right one.)

3. ADD KNOWN-SCHEDULE FEATURES. Promo/holiday proximity, closure-adjacent days, and
   reopening-after-a-long-gap. All are derived only from columns present in `test.csv`
   (`Promo`, `StateHoliday`, `SchoolHoliday`, `Open`, `Date`), never from `Sales`, so they
   are legitimately known at forecast time.
"""

import numpy as np
import pandas as pd

from rossmann_features import ROLLING_WINDOWS

HORIZON_DAYS = 42
NO_EVENT = 999  # sentinel for "no such event anywhere in the known calendar"


# ---------------------------------------------------------------------------
# 1. Rolling features, computed as of an explicit origin date.
# ---------------------------------------------------------------------------

def build_daily_rolling_table(history_open, windows=ROLLING_WINDOWS):
    """For every (Store, calendar date), the mean of open-day Sales/Customers over the
    trailing `w`-day window ENDING ON that date, inclusive.

    Defined on a complete daily calendar (not just open days) so it can be looked up at
    any origin, including one falling on a day the store was shut. Pivoting to a regular
    (date x store) matrix lets a fixed-size rolling window vectorize across all stores at
    once, which is far faster than a grouped date-offset rolling."""
    h = history_open[["Store", "Date", "Sales", "Customers"]]
    calendar = pd.date_range(h["Date"].min(), h["Date"].max(), freq="D")

    frames = []
    for col, prefix in [("Sales", "SalesRollMean"), ("Customers", "CustomersRollMean")]:
        wide = h.pivot_table(index="Date", columns="Store", values=col, aggfunc="mean")
        wide = wide.reindex(calendar)  # closed/missing days become NaN and are skipped
        for w in windows:
            rolled = wide.rolling(window=w, min_periods=1).mean()
            long = rolled.stack(future_stack=True).rename(f"{prefix}{w}").reset_index()
            long.columns = ["Date", "Store", f"{prefix}{w}"]
            frames.append(long.set_index(["Store", "Date"]))

    table = pd.concat(frames, axis=1).reset_index()
    return table


def assign_training_origins(df, train_end, horizon=HORIZON_DAYS):
    """Tile training rows into consecutive `horizon`-day forecast blocks counting back
    from `train_end`, each block forecast from its own origin.

    This mirrors inference exactly -- a batch of rows all predicted from one data cutoff --
    rather than giving every training row a fresh next-day cutoff. Deterministic (no
    sampling), and every row lands at HorizonDay in 1..horizon."""
    df = df.copy()
    days_back = (train_end - df["Date"]).dt.days
    block = days_back // horizon
    df["Origin"] = train_end - pd.to_timedelta((block + 1) * horizon, unit="D")
    df["HorizonDay"] = (df["Date"] - df["Origin"]).dt.days
    return df


def assign_fixed_origin(df, origin):
    """Inference-side counterpart: every row forecast from the same single origin."""
    df = df.copy()
    df["Origin"] = origin
    df["HorizonDay"] = (df["Date"] - origin).dt.days
    return df


def attach_rolling(df, rolling_table, windows=ROLLING_WINDOWS, fallback_source=None):
    """Look up each row's rolling features as of its own `Origin`. Requires `Origin` to
    already be set (by assign_training_origins / assign_fixed_origin)."""
    roll_cols = [f"SalesRollMean{w}" for w in windows] + [f"CustomersRollMean{w}" for w in windows]
    lookup = rolling_table.rename(columns={"Date": "Origin"})

    merged = df.merge(lookup, on=["Store", "Origin"], how="left")
    assert len(merged) == len(df), "rolling lookup changed row count"

    # Origins before the history starts (earliest training blocks) have no window to
    # average, so fall back to the store's overall mean, as v1 did.
    source = fallback_source if fallback_source is not None else df
    sales_mean = source.groupby("Store")["Sales"].mean()
    cust_mean = source.groupby("Store")["Customers"].mean()
    store = merged["Store"]
    for w in windows:
        s_col, c_col = f"SalesRollMean{w}", f"CustomersRollMean{w}"
        merged[s_col] = np.where(merged[s_col].notna(), merged[s_col], store.map(sales_mean))
        merged[c_col] = np.where(merged[c_col].notna(), merged[c_col], store.map(cust_mean))
    return merged


# ---------------------------------------------------------------------------
# 2. Year-ago lag, split into its two distinct quantities.
# ---------------------------------------------------------------------------

def add_yearago_features(df, history_open, fallback):
    """`SalesYearAgo` (the 364-day-ago point) plus `SalesYearAgoIsFallback`, a flag marking
    the rows where no real observation existed and the Phase 1 baseline lookup was
    substituted instead.

    The flag is the point of this function. v1 packed both quantities into one column, so
    the model could not tell a noisy single observation (87% of rows) from a smooth
    baseline prediction (13%) -- and the two behave very differently. Measured on
    validation, the substituted values correlate 0.933 with actual sales while the real
    lags correlate 0.927, so the blended column looked deceptively strong while giving the
    model nothing to condition on.

    Deliberately NOT smoothed. Averaging in neighbouring weeks (51/53 weeks back) was
    tried and made things clearly worse: the Rossmann promo calendar turns out to repeat
    on an exact 364-day cycle -- P(promo 364 days ago == promo today) = 1.000 -- so the
    364-day lag always matches the current row's promo state, while its week-adjacent
    neighbours carry the opposite state. Averaging them in dropped correlation with actual
    sales from 0.927 to 0.786. The single point is the right feature precisely because 364
    is promo-aligned, which is also why 364 rather than 365 was the correct lag all along.

    Sits ~a year in the past regardless of forecast horizon, so unlike the short rolling
    windows it needs no origin/staleness handling."""
    df = df.copy()
    fallback = np.asarray(fallback)
    lookup = history_open[["Store", "Date", "Sales"]]

    shifted = df[["Store", "Date"]].copy()
    shifted["Date"] = shifted["Date"] - pd.Timedelta(days=364)
    # .to_numpy() onto plain arrays, never Series.fillna: .merge() resets to a fresh
    # RangeIndex and pandas would silently align on the wrong index.
    point = shifted.merge(lookup, on=["Store", "Date"], how="left")["Sales"].to_numpy()

    has_real = ~np.isnan(point)
    df["SalesYearAgo"] = np.where(has_real, point, fallback)
    df["SalesYearAgoIsFallback"] = (~has_real).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Known-schedule features (no Sales involved -> safe at forecast time).
# ---------------------------------------------------------------------------

def add_schedule_features(panel):
    """Proximity-to-event and closure-adjacency features.

    `panel` must contain every row whose schedule is known -- train and validation (and,
    in Phase 5, test) concatenated -- so distances are correct across the fold boundary
    instead of being truncated at it. Uses only Store/Date/Promo/StateHoliday/
    SchoolHoliday/Open, all present in `test.csv`."""
    panel = panel.sort_values(["Store", "Date"]).reset_index(drop=True)
    is_state_holiday = panel["StateHoliday"].astype(str).isin(["a", "b", "c"])
    panel["_promo"] = panel["Promo"] == 1
    panel["_state"] = is_state_holiday
    panel["_school"] = panel["SchoolHoliday"] == 1

    g = panel.groupby("Store", sort=False)
    store = panel["Store"]
    # Vectorized rather than groupby.apply: mask the date column to the event days, then
    # forward/back-fill within store. The masked series keeps panel's own index, so this
    # cannot misalign the way a groupby.apply result reindexed by hand can.
    for name, flagcol in [("Promo", "_promo"), ("StateHoliday", "_state"), ("SchoolHoliday", "_school")]:
        marked = panel["Date"].where(panel[flagcol])
        panel[f"DaysSince{name}"] = (panel["Date"] - marked.groupby(store, sort=False).ffill()).dt.days
        panel[f"DaysUntil{name}"] = (marked.groupby(store, sort=False).bfill() - panel["Date"]).dt.days

    # Closure adjacency. A neighbouring row only counts as "yesterday"/"tomorrow" if it
    # really is one calendar day away -- during the 181-store reporting gap the rows are
    # absent entirely, and treating the row across a 6-month hole as "yesterday" would be
    # nonsense. Missing neighbours are treated as closed.
    prev_date = g["Date"].shift(1)
    next_date = g["Date"].shift(-1)
    prev_open = g["Open"].shift(1)
    next_open = g["Open"].shift(-1)
    panel["WasClosedYesterday"] = np.where(
        (panel["Date"] - prev_date).dt.days.eq(1), (prev_open == 0).astype(int), 1)
    panel["WillBeClosedTomorrow"] = np.where(
        (next_date - panel["Date"]).dt.days.eq(1), (next_open == 0).astype(int), 1)

    # Reopening after a long shutdown (refurbishment-like). Detected from the gap between
    # consecutive OPEN days, so it catches both Open==0 runs and absent rows.
    open_date = panel["Date"].where(panel["Open"] == 1)
    prev_open_date = open_date.groupby(panel["Store"], sort=False).ffill().groupby(panel["Store"], sort=False).shift(1)
    gap = (panel["Date"] - prev_open_date).dt.days
    is_reopen = (panel["Open"] == 1) & (gap >= 5)
    reopen_date = panel["Date"].where(is_reopen).groupby(panel["Store"], sort=False).ffill()
    panel["DaysSinceReopening"] = (panel["Date"] - reopen_date).dt.days

    prox = [c for c in panel.columns if c.startswith(("DaysSince", "DaysUntil"))]
    panel[prox] = panel[prox].fillna(NO_EVENT).clip(upper=NO_EVENT)

    panel["DayOfMonth"] = panel["Date"].dt.day
    panel["WeekOfYear"] = panel["Date"].dt.isocalendar().week.astype(int)
    return panel.drop(columns=["_promo", "_state", "_school"])


def fit_store_trend(history_open):
    """Per-store OLS slope of log1p(Sales) against time. Fitted on the fold's training
    history only, exactly like the target encoding it sits alongside."""
    h = history_open[["Store", "Date"]].copy()
    t = (history_open["Date"] - history_open["Date"].min()).dt.days.astype(float)
    y = np.log1p(history_open["Sales"].to_numpy())
    frame = pd.DataFrame({"Store": history_open["Store"].to_numpy(), "t": t.to_numpy(), "y": y})
    g = frame.groupby("Store")
    t_mean, y_mean = g["t"].transform("mean"), g["y"].transform("mean")
    frame["cov"] = (frame["t"] - t_mean) * (frame["y"] - y_mean)
    frame["var"] = (frame["t"] - t_mean) ** 2
    agg = frame.groupby("Store")[["cov", "var"]].sum()
    return (agg["cov"] / agg["var"].replace(0, np.nan)).fillna(0.0)


def add_store_trend(df, trends):
    df = df.copy()
    df["StoreTrend"] = df["Store"].map(trends).fillna(0.0)
    return df


def add_promo2_duration(df):
    """Weeks a store has been in its Promo2 programme (0 when not enrolled/not started) --
    strictly more informative than v1's binary active flag."""
    df = df.copy()
    weeks = (df["Date"] - df["Promo2SinceDate"]).dt.days / 7.0
    df["Promo2WeeksActive"] = weeks.where((df["Promo2"] == 1) & (weeks > 0), 0.0).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Column groups for the v2 feature set.
# ---------------------------------------------------------------------------

LAG_COLS_V2 = (
    [f"SalesRollMean{w}" for w in ROLLING_WINDOWS]
    + [f"CustomersRollMean{w}" for w in ROLLING_WINDOWS]
    + ["SalesYearAgo", "SalesYearAgoIsFallback", "HorizonDay"]
)

SCHEDULE_COLS = [
    "DaysSincePromo", "DaysUntilPromo",
    "DaysSinceStateHoliday", "DaysUntilStateHoliday",
    "DaysSinceSchoolHoliday", "DaysUntilSchoolHoliday",
    "WasClosedYesterday", "WillBeClosedTomorrow", "DaysSinceReopening",
    "DayOfMonth", "WeekOfYear",
]

EXTRA_NUMERIC_COLS = ["StoreTrend", "Promo2WeeksActive"]
