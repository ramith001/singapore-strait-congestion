"""
Model selection: Malacca vessel count vs Singapore port calls vs Singapore trade value.

    python model_selection.py

All three candidates go through the identical pipeline (baseline.py): log-transform,
rolling trend + day-of-week + month re-fit every quarter on the trailing 2 years,
score the next quarter, flag |z| > 2.5, cluster, match to documented events within
+/-14 days. Nothing here is hand-entered; the README's model-selection numbers come
from this script's output.
"""
import numpy as np
import pandas as pd

from baseline import THRESH, MIN_TRAIN_DAYS, rolling_oos_fit, cluster_anomalies, matched_event

sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])
sg["total_trade"] = sg["import"] + sg["export"]

df = mal[["date", "n_total"]].merge(
    sg[["date", "portcalls", "total_trade"]], on="date", how="inner"
).sort_values("date").reset_index(drop=True)

CANDIDATES = [
    ("Malacca vessel count", "n_total"),
    ("Singapore port calls", "portcalls"),
    ("Singapore trade value", "total_trade"),
]

# The fit is on log(y), so every candidate must be strictly positive and finite.
for label, col in CANDIDATES:
    n_bad = int((~np.isfinite(df[col]) | (df[col] <= 0)).sum())
    assert n_bad == 0, f"{label}: {n_bad} non-positive/missing values, cannot log-fit"
print(f"Merged {len(df)} days, {df.date.min().date()} to {df.date.max().date()}; "
      f"all candidates strictly positive (min trade value {df['total_trade'].min():,.0f}).\n")

first_oos = df["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
last_day = df["date"].max()
span_days = (last_day - first_oos).days
rng = np.random.default_rng(0)


def chance_match_rate(clusters, n_draws=200):
    """Match rate if each cluster kept its length but sat at a random place in the
    out-of-sample span. The calendar covers a large share of 2020-2026, so a high
    raw match rate is partly just coverage; this is the number to beat."""
    hits = total = 0
    for _ in range(n_draws):
        for r in clusters.itertuples():
            length = (r.end - r.start).days
            start = first_oos + pd.Timedelta(days=int(rng.integers(0, max(span_days - length, 1))))
            hits += matched_event(events, start, start + pd.Timedelta(days=length)) is not None
            total += 1
    return hits / total if total else float("nan")


rows = []
for label, col in CANDIDATES:
    fitted, z, train_r2, oos_r2 = rolling_oos_fit(df, col)
    tmp = df[["date"]].copy()
    tmp["z"] = z
    tmp = tmp[tmp["date"] >= first_oos].dropna(subset=["z"]).reset_index(drop=True)
    _, clusters = cluster_anomalies(tmp, "z")
    matched = [matched_event(events, r.start, r.end) is not None for r in clusters.itertuples()]
    n_c, n_m = len(clusters), int(sum(matched))
    rows.append({
        "target": label,
        "rolling train R2": round(train_r2, 3),
        "rolling OOS R2": round(oos_r2, 3),
        "clusters": n_c,
        "matched (+/-14d)": n_m,
        "match rate": f"{n_m / n_c:.0%}" if n_c else "n/a",
        "chance rate": f"{chance_match_rate(clusters):.0%}",
        "flagged days": int((tmp['z'].abs() > THRESH).sum()),
    })

print(f"Identical rolling out-of-sample method for each target (|z| > {THRESH}):\n")
print(pd.DataFrame(rows).to_string(index=False))
print("\nchance rate = match rate if the same clusters were placed at random dates in the out-of-sample span.")
print("train R2 = mean in-sample R2 of the quarterly training windows (what analysis.py prints);")
print("OOS R2   = R2 of the scored, never-trained-on days against their own forecast (log scale).")
