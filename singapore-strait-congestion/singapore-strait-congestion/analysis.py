"""
Singapore Strait Congestion Anomaly Detection
-----------------------------------------------
Data: IMF PortWatch, 2019-01-01 to 2026-09.
  - Daily_Chokepoints_Data for the Malacca Strait (chokepoint5): daily vessel
    transit counts (n_total) -- this is the primary target, since it has by
    far the cleanest weekly/monthly seasonal structure of the candidates
    tried (see model-selection note below).
  - Daily_Ports_Data for the Port of Singapore (port1201): daily port calls
    and estimated import/export trade -- used as a companion series.

Model-selection note: three candidate targets were tried for the baseline
(Singapore daily trade value, Singapore daily port calls, Malacca daily
vessel count). Trade VALUE barely has any seasonal structure (in-sample
R^2 = 0.03 on a trend + day-of-week + month baseline) because it is a
capacity-weighted dollar estimate that swings a lot with vessel size mix,
so treating its residuals as "anomalies" would mostly be picking up noise,
not disruption. Vessel COUNTS are much better behaved (Malacca R^2 = 0.45,
Singapore port calls R^2 = 0.26), so counts are used as the anomaly-detection
target and trade value is kept only as a descriptive companion series.

Approach:
1. Fit trend + day-of-week + month-of-year seasonality on data strictly
   BEFORE the Red Sea Tensions disruption (cutoff 2023-12-16), the same
   out-of-sample discipline as the volatility-forecasting project: fit on
   one period, only ever call something "anomalous" in a period the model
   never saw.
2. Score every day after the cutoff against that baseline; flag |z| > 2.5
   as anomalous, group consecutive flagged days into clusters.
3. Check whether flagged clusters line up with real, documented disruption
   events -- same "is this a genuine explainable outlier" check used on the
   2020 crash window in the portfolio-optimization project. Events with no
   recorded end date (Red Sea Tensions, Hormuz-26 are still open per
   PortWatch) are only matched within a 365-day window of their start, not
   indefinitely -- matching an "ongoing" event forever would make every
   later anomaly trivially "explained" and defeat the point of the check.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------------------
sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])

sg["total_trade"] = sg["import"] + sg["export"]

df = mal[["date", "n_total", "capacity"]].merge(
    sg[["date", "portcalls", "total_trade"]], on="date", how="inner"
).sort_values("date").reset_index(drop=True)
print("Merged shape:", df.shape, "range:", df.date.min().date(), "to", df.date.max().date())

CUTOFF = pd.Timestamp("2023-12-16")  # Red Sea Tensions start
OPEN_EVENT_WINDOW_DAYS = 365          # bound for events with no recorded end date

# ---------------------------------------------------------------------------
# 2. Baseline: trend + day-of-week + month, numpy OLS, re-fit on a ROLLING
#    trailing window rather than one static fit extrapolated for years.
#
#    First attempt (kept as a documented dead end, not silently dropped):
#    a single trend fit on pre-2023-12-16 data and extrapolated forward
#    through Sept 2026 gave an in-sample R^2 of 0.45, but flagged EVERY
#    out-of-sample anomaly in 2026 and none in 2024 -- a sign the linear
#    trend was drifting from reality over a ~2.5-3 year extrapolation, not
#    that 2026 was truly more disruptive than 2024. That's the same kind of
#    "unconstrained optimization failure" that showed up in the GARCH vs ML
#    project (there from data leakage, here from over-extrapolating a static
#    trend) and it would have meant reporting a training artifact as a real
#    finding, so it's rejected rather than presented as a result.
#
#    Fix: refit trend + day-of-week + month every quarter, using only the
#    trailing 2 years of history strictly BEFORE that quarter (min. 1 year
#    of history before the first refit). Every fitted value is still purely
#    out-of-sample relative to its own training window, but the baseline
#    can no longer drift for years on one fixed trend line -- the same
#    rolling-window discipline used for the portfolio project's 9-window
#    out-of-sample validation.
# ---------------------------------------------------------------------------
t0 = df["date"].min()

def build_design(dates, trend_origin):
    n = len(dates)
    trend = (dates - trend_origin).dt.days.values.astype(float)
    dow_cat = pd.Categorical(dates.dt.dayofweek, categories=list(range(7)))
    month_cat = pd.Categorical(dates.dt.month, categories=list(range(1, 13)))
    dow = pd.get_dummies(dow_cat, prefix="dow", drop_first=True).astype(float)
    month = pd.get_dummies(month_cat, prefix="mo", drop_first=True).astype(float)
    X = np.column_stack([np.ones(n), trend, dow.values, month.values])
    return X

MIN_TRAIN_DAYS = 365
WINDOW_DAYS = 730
REFIT_FREQ_DAYS = 90

def rolling_oos_fit(colname):
    y_full = np.log(df[colname].values)
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
    return fitted, z, np.mean(r2s)

df["fitted_ntotal"], df["z_ntotal"], r2_ntotal = rolling_oos_fit("n_total")
df["fitted_portcalls"], df["z_portcalls"], r2_portcalls = rolling_oos_fit("portcalls")
print(f"Rolling-window baseline mean training R^2 -- Malacca vessel count: {r2_ntotal:.3f} "
      f"| Singapore port calls: {r2_portcalls:.3f}")

# ---------------------------------------------------------------------------
# 3. Flag anomalies (all scoring above is already out-of-sample by construction:
#    every day is scored against a model trained only on data before it)
# ---------------------------------------------------------------------------
THRESH = 2.5
oos = df[df["date"] >= (t0 + pd.Timedelta(days=MIN_TRAIN_DAYS))].copy().sort_values("date").reset_index(drop=True)
oos = oos.dropna(subset=["z_ntotal"])
oos["flag"] = oos["z_ntotal"].abs() > THRESH

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
         max_abs_z=("z_ntotal", lambda s: s.abs().max()))
    .sort_values("start")
    .reset_index(drop=True)
)
print(f"\n{len(clusters)} out-of-sample anomaly clusters flagged (|z| > {THRESH}) on Malacca vessel counts.")

# ---------------------------------------------------------------------------
# 4. Cross-reference against documented events (bounded window for open ones)
# ---------------------------------------------------------------------------
def matched_event(c_start, c_end, pad_days=14):
    for _, ev in events.iterrows():
        ev_start = ev["fromdate"] - pd.Timedelta(days=pad_days)
        if pd.notna(ev["todate"]):
            ev_end = ev["todate"] + pd.Timedelta(days=pad_days)
        else:
            ev_end = ev["fromdate"] + pd.Timedelta(days=OPEN_EVENT_WINDOW_DAYS)
        if c_start <= ev_end and c_end >= ev_start:
            return ev["eventname"]
    return None

clusters["matched_event"] = [matched_event(r.start, r.end) for r in clusters.itertuples()]
n_matched = clusters["matched_event"].notna().sum()
print(f"{n_matched} of {len(clusters)} clusters fall within +/-14 days of a documented disruption "
      f"event (open-ended events bounded to their first {OPEN_EVENT_WINDOW_DAYS} days).")
print(clusters.sort_values("max_abs_z", ascending=False).head(12).to_string(index=False))

clusters.to_csv("data/anomaly_clusters.csv", index=False)
df.to_csv("data/full_analysis_dataset.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Rolling volatility of daily Malacca vessel counts (GARCH-style lens)
# ---------------------------------------------------------------------------
df["pct_change_ntotal"] = df["n_total"].pct_change()
df["roll_vol_30d"] = df["pct_change_ntotal"].rolling(30).std()

# ---------------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------------
def shade_events(ax, ymin, ymax):
    for _, ev in events.iterrows():
        end = ev["todate"] if pd.notna(ev["todate"]) else ev["fromdate"] + pd.Timedelta(days=OPEN_EVENT_WINDOW_DAYS)
        ax.axvspan(ev["fromdate"], end, color="tab:red", alpha=0.08)

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

ax = axes[0]
ax.plot(df["date"], df["n_total"], color="#1f6f8b", lw=0.8, label="Malacca Strait daily vessel count")
ax.axvline(t0 + pd.Timedelta(days=MIN_TRAIN_DAYS), color="black", ls="--", lw=1,
           label="first out-of-sample day (after initial 1yr training window)")
shade_events(ax, df["n_total"].min(), df["n_total"].max())
ax.set_ylabel("Vessels/day")
ax.set_title("Malacca Strait daily vessel transits, with documented disruption windows shaded")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(oos["date"], oos["z_ntotal"], color="#1f6f8b", lw=0.9)
ax.axhline(THRESH, color="tab:red", ls=":", lw=1)
ax.axhline(-THRESH, color="tab:red", ls=":", lw=1)
flagged = oos[oos["flag"]]
ax.scatter(flagged["date"], flagged["z_ntotal"], color="tab:red", s=14, zorder=5, label="flagged anomaly")
shade_events(ax, -5, 5)
ax.set_ylabel("Residual z-score")
ax.set_title(f"Out-of-sample residual z-score vs. pre-cutoff baseline (threshold |z|>{THRESH})")
ax.legend(loc="upper left", fontsize=8)

ax = axes[2]
ax.plot(df["date"], df["roll_vol_30d"], color="#1f6f8b", lw=0.9)
shade_events(ax, 0, df["roll_vol_30d"].max())
ax.set_ylabel("30-day rolling std of % change")
ax.set_title("Rolling volatility of Malacca Strait daily vessel counts")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
fig.savefig("data/chart_overview.png", dpi=150)
print("\nSaved chart_overview.png")
