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

Model-selection note: three candidate targets (Malacca daily vessel count,
Singapore daily port calls, Singapore daily trade value) are compared under the
identical rolling out-of-sample method by model_selection.py, whose output is
printed at the top of this script and summarised in the README. The comparison
does NOT show a clean R^2 win for Malacca (in-sample window R^2 ranks port calls
first); Malacca is kept as the target because it has the best genuinely
out-of-sample fit and the most granular signal, and its anomalies are checked
against documented disruptions. Trade value is kept only as a descriptive
companion series.

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
import runpy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from baseline import (MIN_TRAIN_DAYS, THRESH, OPEN_EVENT_WINDOW_DAYS,
                      rolling_oos_fit, cluster_anomalies, matched_event)

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------------------
print("=== Model selection (model_selection.py) ===")
runpy.run_path("model_selection.py")
print("\n=== Malacca pipeline ===")

sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])

sg["total_trade"] = sg["import"] + sg["export"]

df = mal[["date", "n_total", "capacity"]].merge(
    sg[["date", "portcalls", "total_trade"]], on="date", how="inner"
).sort_values("date").reset_index(drop=True)
print("Merged shape:", df.shape, "range:", df.date.min().date(), "to", df.date.max().date())

CUTOFF = pd.Timestamp("2023-12-16")  # Red Sea Tensions start

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

df["fitted_ntotal"], df["z_ntotal"], r2_ntotal, _ = rolling_oos_fit(df, "n_total")
df["fitted_portcalls"], df["z_portcalls"], r2_portcalls, _ = rolling_oos_fit(df, "portcalls")
print(f"Rolling-window baseline mean training R^2 -- Malacca vessel count: {r2_ntotal:.3f} "
      f"| Singapore port calls: {r2_portcalls:.3f}")

# ---------------------------------------------------------------------------
# 3. Flag anomalies (all scoring above is already out-of-sample by construction:
#    every day is scored against a model trained only on data before it)
# ---------------------------------------------------------------------------
oos = df[df["date"] >= (t0 + pd.Timedelta(days=MIN_TRAIN_DAYS))].copy().sort_values("date").reset_index(drop=True)
oos = oos.dropna(subset=["z_ntotal"]).reset_index(drop=True)
oos, clusters = cluster_anomalies(oos, "z_ntotal")
print(f"\n{len(clusters)} out-of-sample anomaly clusters flagged (|z| > {THRESH}) on Malacca vessel counts.")

# ---------------------------------------------------------------------------
# 4. Cross-reference against documented events (bounded window for open ones)
# ---------------------------------------------------------------------------
clusters["matched_event"] = [matched_event(events, r.start, r.end) for r in clusters.itertuples()]
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
