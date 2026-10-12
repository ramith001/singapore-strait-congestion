"""
Shared rolling out-of-sample baseline, anomaly clustering and event matching.

Used by both analysis.py (the Malacca pipeline) and model_selection.py (the
three-way target comparison), so the two can never drift apart.
"""
import numpy as np
import pandas as pd

MIN_TRAIN_DAYS = 365
WINDOW_DAYS = 730
REFIT_FREQ_DAYS = 90
THRESH = 2.5
OPEN_EVENT_WINDOW_DAYS = 365  # bound for events with no recorded end date


def build_design(dates, trend_origin):
    n = len(dates)
    trend = (dates - trend_origin).dt.days.values.astype(float)
    dow_cat = pd.Categorical(dates.dt.dayofweek, categories=list(range(7)))
    month_cat = pd.Categorical(dates.dt.month, categories=list(range(1, 13)))
    dow = pd.get_dummies(dow_cat, prefix="dow", drop_first=True).astype(float)
    month = pd.get_dummies(month_cat, prefix="mo", drop_first=True).astype(float)
    return np.column_stack([np.ones(n), trend, dow.values, month.values])


def rolling_oos_fit(df, colname):
    """Refit log(colname) ~ trend + day-of-week + month every REFIT_FREQ_DAYS on
    the trailing WINDOW_DAYS strictly before the refit date, then score the days
    until the next refit.

    Returns (fitted, z, mean_train_r2, oos_r2):
      mean_train_r2 -- mean in-sample R^2 of the quarterly training windows
                       (the figure analysis.py has always printed);
      oos_r2        -- R^2 of the scored (never-trained-on) days against their
                       own forecast, on the log scale, 1 - SSE/SST around the
                       scored days' mean.
    """
    t0 = df["date"].min()
    y_full = np.log(df[colname].values)  # callers must pass strictly positive data
    dates = df["date"]
    fitted = np.full(len(df), np.nan)
    z = np.full(len(df), np.nan)
    r2s = []
    first_refit_date = t0 + pd.Timedelta(days=MIN_TRAIN_DAYS)
    refit_dates = pd.date_range(first_refit_date, dates.max(), freq=f"{REFIT_FREQ_DAYS}D")
    for i, refit_date in enumerate(refit_dates):
        train_start = max(t0, refit_date - pd.Timedelta(days=WINDOW_DAYS))
        train_idx = (dates >= train_start) & (dates < refit_date)
        score_end = refit_dates[i + 1] if i + 1 < len(refit_dates) else dates.max() + pd.Timedelta(days=1)
        score_idx = (dates >= refit_date) & (dates < score_end)
        if train_idx.sum() < 60 or score_idx.sum() == 0:
            continue
        X_train = build_design(dates[train_idx], train_start)
        X_score = build_design(dates[score_idx], train_start)
        beta, *_ = np.linalg.lstsq(X_train, y_full[train_idx.values], rcond=None)
        train_fit = X_train @ beta
        resid_std = (y_full[train_idx.values] - train_fit).std(ddof=1)
        r2s.append(1 - np.var(y_full[train_idx.values] - train_fit) / np.var(y_full[train_idx.values]))
        score_fit = X_score @ beta
        fitted[score_idx.values] = score_fit
        z[score_idx.values] = (y_full[score_idx.values] - score_fit) / resid_std
    scored = ~np.isnan(fitted)
    ys = y_full[scored]
    oos_r2 = 1 - np.sum((ys - fitted[scored]) ** 2) / np.sum((ys - ys.mean()) ** 2)
    return fitted, z, np.mean(r2s), oos_r2


def cluster_anomalies(oos, zcol):
    """Flag |z| > THRESH on `zcol` and group flagged days (gap <= 3 days) into
    clusters. `oos` must be date-sorted with a default index. Adds `flag` and
    `cluster` columns and returns (oos, clusters table)."""
    oos = oos.copy()
    oos["flag"] = oos[zcol].abs() > THRESH
    cluster_ids, cid, prev_flag, prev_date = [], 0, False, None
    for _, row in oos.iterrows():
        if row["flag"]:
            if not prev_flag or (prev_date is not None and (row["date"] - prev_date).days > 3):
                cid += 1
            cluster_ids.append(cid)
            prev_date = row["date"]
        else:
            cluster_ids.append(np.nan)
        prev_flag = row["flag"]
    oos["cluster"] = cluster_ids
    clusters = (
        oos.dropna(subset=["cluster"])
        .groupby("cluster")
        .agg(start=("date", "min"), end=("date", "max"), n_days=("date", "count"),
             max_abs_z=(zcol, lambda s: s.abs().max()))
        .sort_values("start")
        .reset_index(drop=True)
    )
    return oos, clusters


def matched_event(events, c_start, c_end, pad_days=14):
    for _, ev in events.iterrows():
        ev_start = ev["fromdate"] - pd.Timedelta(days=pad_days)
        if pd.notna(ev["todate"]):
            ev_end = ev["todate"] + pd.Timedelta(days=pad_days)
        else:
            ev_end = ev["fromdate"] + pd.Timedelta(days=OPEN_EVENT_WINDOW_DAYS)
        if c_start <= ev_end and c_end >= ev_start:
            return ev["eventname"]
    return None
