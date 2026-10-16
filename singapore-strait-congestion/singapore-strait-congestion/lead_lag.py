"""
Part 3: lead-lag between Malacca and Singapore traffic anomalies. Does one side's disruption show up first?

WHAT "SINGAPORE" IS HERE. PortWatch's chokepoint layer has 28 chokepoints (Malacca is chokepoint5) and no Singapore
Strait series. The Singapore data this repo holds, and the only Singapore series PortWatch publishes at daily
frequency, is the Port of Singapore (port1201): port calls and trade by vessel type. So the comparison is
transits through the Strait of Malacca against arrivals at the Port of Singapore, which is also the natural
physical pair (a ship that transits Malacca and then calls at Singapore does so about a day later). Both
files carry per-vessel-type columns, so the vessel-type breakdown needs no new data: Malacca n_total / n_container /
n_tanker against Singapore portcalls / portcalls_container / portcalls_tanker. (Dry bulk is left out: Singapore logs
about 2 dry-bulk calls a day against Malacca's 51 transits, so the pair is not comparable. General cargo and RoRo are
small in both.)

ANOMALY SERIES. The z-score from baseline.rolling_oos_fit, unchanged, applied to each series: log(count) against a
trend + day-of-week + month baseline refit quarterly on the trailing 2 years and scored out of sample. Two cautions
carried from earlier parts:
  - Part 2's baseline edge: the first fitting window (2020-01-01 to 2020-03-30, scored by a model fitted on one year of
    data) is artifactual for Malacca. This script finds it is artifactual for Singapore too, with the OPPOSITE sign
    (Malacca z averages -3.9 over that window, Singapore +4.5), which on its own flips the sign of the whole-period
    correlation. Every result is therefore reported three ways, as Part 2 did: all out-of-sample days; excluding the
    first 90-day window; and from 2022-01-01, which also drops the COVID-era traffic collapse.
  - The Singapore baseline is poor out of sample (R^2 -0.28 for port calls, model_selection.py), so its z-score is
    partly baseline drift. As a check that does not depend on extrapolating a trend, the total-traffic pair is
    repeated on a causal high-pass series: log count, minus its day-of-week mean, minus its own trailing 28-day mean.

STATISTICS. Everything is a Pearson correlation of daily z-scores, with k > 0 meaning Malacca leads Singapore by k days.
  - Cross-correlation r(k) for k = -14..14, with a null band from 2,000 circular shifts of the Singapore series (shifts
    of at least 30 days, which keeps the autocorrelation and the fat tails and destroys any alignment).
  - Lead asymmetry A = mean r(k) over k = 1..7 minus mean r(k) over k = -7..-1. A > 0: Malacca leads. Permutation
    p-value from the same circular shifts.
  - Granger tests (OLS F-test, 7 lags): does Malacca's past help predict Singapore beyond Singapore's own past, and the reverse?
    Predictive, not causal, and with a shared common shock it can go either way.
  - Cluster timing: flagged clusters (|z| > 2.5, consecutive flagged days, as baseline.cluster_anomalies builds them) in each series; for each Malacca cluster
    the nearest Singapore cluster start within 14 days, and the signed gap. Compared against the same procedure with the
    Singapore series circularly shifted, since flagged days are dense enough that some pairing happens by chance.

    python lead_lag.py
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from baseline import THRESH, cluster_anomalies, rolling_oos_fit

PAIRS = {
    "total": ("n_total", "portcalls"),
    "container": ("n_container", "portcalls_container"),
    "tanker": ("n_tanker", "portcalls_tanker"),
}
VARIANTS = {
    "all out-of-sample": "2020-01-01",
    "excluding first 90 days": "2020-03-31",
    "from 2022 (also excl. COVID)": "2022-01-01",
}
MAX_LAG = 14
BAND = 7              # lead/lag asymmetry averaged over 1..7 days
GRANGER_LAGS = 7
MATCH_DAYS = 14
N_PERM = 2000
N_PERM_CLUSTER = 500
SEED = 0

# dataviz skill palette (references/palette.md), as in event_study.py: blue / orange / aqua categorical slots in order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load():
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"])
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"])
    cols_m = ["date"] + [m for m, _ in PAIRS.values()]
    cols_s = ["date"] + [s for _, s in PAIRS.values()]
    return mal[cols_m].merge(sg[cols_s], on="date").sort_values("date").reset_index(drop=True)


def z_series(df, col):
    _, z, _, oos_r2 = rolling_oos_fit(df, col)
    return pd.Series(z, index=df["date"]), oos_r2


def highpass(df, col, trail=28):
    """Causal anomaly series that needs no trend extrapolation: log count minus its day-of-week mean, minus its own
    trailing `trail`-day mean (standardized by the overall SD)."""
    y = pd.Series(np.log(df[col].values), index=df["date"])
    y = y - y.groupby(y.index.dayofweek).transform("mean")
    hp = y - y.rolling(trail).mean().shift(1)
    return hp / hp.std()


def xcorr(a, b, k):
    """Pearson correlation of a[t] with b[t+k] over the overlap (k > 0: a leads b)."""
    n = len(a)
    if k >= 0:
        x, y = a[:n - k], b[k:]
    else:
        x, y = a[-k:], b[:n + k]
    return float(np.corrcoef(x, y)[0, 1])


def xcorr_all(a, b):
    return np.array([xcorr(a, b, k) for k in range(-MAX_LAG, MAX_LAG + 1)])


def asym(r):
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    return r[(lags >= 1) & (lags <= BAND)].mean() - r[(lags >= -BAND) & (lags <= -1)].mean()


def circular_null(a, b, rng, n_perm=N_PERM):
    n = len(a)
    shifts = rng.integers(30, n - 30, size=n_perm)
    return np.array([xcorr_all(a, np.roll(b, s)) for s in shifts])


def granger_p(y, x, lags=GRANGER_LAGS):
    """F-test that x's past adds to y's own past in predicting y. Returns the p-value."""
    n = len(y)
    Y = y[lags:]
    own = np.column_stack([y[lags - j:n - j] for j in range(1, lags + 1)])
    oth = np.column_stack([x[lags - j:n - j] for j in range(1, lags + 1)])
    ones = np.ones((len(Y), 1))
    Xr, Xu = np.hstack([ones, own]), np.hstack([ones, own, oth])
    rss_r = np.sum((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2)
    rss_u = np.sum((Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]) ** 2)
    f = ((rss_r - rss_u) / lags) / (rss_u / (len(Y) - Xu.shape[1]))
    return float(stats.f.sf(f, lags, len(Y) - Xu.shape[1]))


def fast_clusters(z, dates):
    """Cluster starts, vectorized so it can run inside a permutation loop. This reproduces what
    baseline.cluster_anomalies actually does, which is slightly different from its docstring: a flagged day starts a
    new cluster whenever the PREVIOUS DAY was not flagged, so a cluster is a run of consecutive flagged days and the
    "gap <= 3 days" clause never merges anything. (baseline.py is left unchanged; check_cluster_function asserts the
    two agree on the real series.)"""
    idx = np.flatnonzero(np.abs(z) > THRESH)
    if len(idx) == 0:
        return np.array([], dtype="datetime64[ns]")
    d = dates[idx]
    new = np.r_[True, (d[1:] - d[:-1]).astype("timedelta64[D]").astype(int) > 1]
    return d[new]


def match_gaps(starts_a, starts_b, window=MATCH_DAYS):
    """For each cluster start in A, the signed gap (days) to the nearest B cluster start within +/-window
    (B minus A: positive means B comes later, i.e. A is first). NaN where none is within the window."""
    if len(starts_a) == 0 or len(starts_b) == 0:
        return np.full(len(starts_a), np.nan)
    gap = (starts_b[None, :] - starts_a[:, None]).astype("timedelta64[D]").astype(float)
    nearest = gap[np.arange(len(starts_a)), np.argmin(np.abs(gap), axis=1)]
    return np.where(np.abs(nearest) <= window, nearest, np.nan)


def check_cluster_function(z, dates):
    """The fast clustering must agree with baseline.cluster_anomalies on the real series."""
    frame = pd.DataFrame({"date": dates, "z": z})
    _, clusters = cluster_anomalies(frame, "z")
    fast = fast_clusters(z, dates.values)
    assert len(fast) == len(clusters) and (pd.DatetimeIndex(fast) == pd.DatetimeIndex(clusters["start"])).all(), \
        "fast_clusters disagrees with baseline.cluster_anomalies"


def analyse_pair(name, za, zb, rng, rows_x, rows_s, rows_c, variants=None):
    for variant, start in (variants or VARIANTS).items():
        a, b = za[start:].dropna(), zb[start:].dropna()
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        av, bv = a.values, b.values
        r = xcorr_all(av, bv)
        null = circular_null(av, bv, rng)
        a_stat, a_null = asym(r), np.array([asym(x) for x in null])
        lags = np.arange(-MAX_LAG, MAX_LAG + 1)
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        for k, rk, l_, h_ in zip(lags, r, lo, hi):
            rows_x.append({"pair": name, "variant": variant, "lag_days": int(k), "r": rk, "null_lo": l_, "null_hi": h_})
        in_band = (lags >= -BAND) & (lags <= BAND)
        k_peak = int(lags[in_band][np.argmax(np.abs(r[in_band]))])
        rows_s.append({
            "pair": name, "variant": variant, "n_days": len(a), "r0": r[MAX_LAG],
            "r0_null_lo": lo[MAX_LAG], "r0_null_hi": hi[MAX_LAG],
            "r_malacca_leads_1to7": r[(lags >= 1) & (lags <= BAND)].mean(),
            "r_singapore_leads_1to7": r[(lags >= -BAND) & (lags <= -1)].mean(),
            "asymmetry_A": a_stat, "p_A": float(np.mean(np.abs(a_null) >= abs(a_stat))), "peak_lag_within_7": k_peak,
            "granger_p_malacca_to_singapore": granger_p(bv, av), "granger_p_singapore_to_malacca": granger_p(av, bv)})

        # cluster timing
        dates = a.index.values
        sm, ss = fast_clusters(av, dates), fast_clusters(bv, dates)
        g_ms = match_gaps(sm, ss)       # Singapore start minus Malacca start, for each Malacca cluster
        matched = ~np.isnan(g_ms)
        shifts = rng.integers(30, len(bv) - 30, size=N_PERM_CLUSTER)
        null_frac = np.array([np.mean(~np.isnan(match_gaps(sm, fast_clusters(np.roll(bv, s), dates)))) for s in shifts]) \
            if len(sm) and len(ss) else np.array([np.nan])
        g = g_ms[matched]
        rows_c.append({
            "pair": name, "variant": variant, "malacca_clusters": len(sm), "singapore_clusters": len(ss),
            "malacca_clusters_with_singapore_within_14d": int(matched.sum()),
            "matched_share": matched.mean() if len(sm) else np.nan,
            "matched_share_random_placement_mean": float(np.nanmean(null_frac)),
            "matched_share_random_placement_95pct": float(np.nanpercentile(null_frac, 95)),
            "median_gap_days_singapore_minus_malacca": float(np.median(g)) if len(g) else np.nan,
            "share_malacca_first": float(np.mean(g > 0)) if len(g) else np.nan,
            "share_same_day_or_within_1d": float(np.mean(np.abs(g) <= 1)) if len(g) else np.nan,
            "share_singapore_first": float(np.mean(g < 0)) if len(g) else np.nan})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    df = load()
    print(f"Merged Malacca / Singapore frame: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    print("Vessel-type detail present in both files (no new data needed): "
          + ", ".join(f"{m} vs {s}" for m, s in PAIRS.values()))
    print("PortWatch has no Singapore Strait chokepoint series; 'Singapore' here is the Port of Singapore (port1201).")

    z = {}
    print("\nOut-of-sample R^2 of each baseline (1 = perfect, 0 = flat average):")
    for name, (m, s) in PAIRS.items():
        zm, r2m = z_series(df, m)
        zs, r2s = z_series(df, s)
        z[name] = (zm, zs)
        print(f"  {name:<10} Malacca {r2m:+.3f}   Singapore {r2s:+.3f}")
    zm, zs = z["total"]
    check_cluster_function(zm.dropna().values, zm.dropna().index)
    check_cluster_function(zs.dropna().values, zs.dropna().index)
    print("Fast cluster routine agrees with baseline.cluster_anomalies on both real series.")

    edge = slice("2020-01-01", "2020-03-30")
    print("\nBaseline edge window (2020-01-01 to 2020-03-30), mean z:")
    for name in PAIRS:
        print(f"  {name:<10} Malacca {z[name][0][edge].mean():+.2f}   Singapore {z[name][1][edge].mean():+.2f}")

    rows_x, rows_s, rows_c = [], [], []
    for name in PAIRS:
        analyse_pair(name, z[name][0], z[name][1], rng, rows_x, rows_s, rows_c)
    # high-pass robustness, total-traffic pair; the edge artifact does not exist in this series, so 2019 onward is used too
    hm, hs = highpass(df, "n_total"), highpass(df, "portcalls")
    hp_variants = {"from 2019-02 (all days)": "2019-02-01", "from 2020-03-31": "2020-03-31",
                   "from 2022 (also excl. COVID)": "2022-01-01"}
    hp_rows_x, hp_rows_s, hp_rows_c = [], [], []
    analyse_pair("total (high-pass)", hm, hs, rng, hp_rows_x, hp_rows_s, hp_rows_c, variants=hp_variants)

    xdf, sdf, cdf = pd.DataFrame(rows_x), pd.DataFrame(rows_s), pd.DataFrame(rows_c)
    hx, hsdf, hcdf = pd.DataFrame(hp_rows_x), pd.DataFrame(hp_rows_s), pd.DataFrame(hp_rows_c)
    pd.concat([xdf, hx]).to_csv("data/lead_lag_xcorr.csv", index=False)
    pd.concat([sdf, hsdf]).to_csv("data/lead_lag_summary.csv", index=False)
    pd.concat([cdf, hcdf]).to_csv("data/lead_lag_clusters.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    show = ["pair", "variant", "n_days", "r0", "r0_null_lo", "r0_null_hi", "r_malacca_leads_1to7", "r_singapore_leads_1to7",
            "asymmetry_A", "p_A", "peak_lag_within_7", "granger_p_malacca_to_singapore", "granger_p_singapore_to_malacca"]
    print("\nCross-correlation, lead asymmetry and Granger tests (k > 0: Malacca leads):")
    print(pd.concat([sdf, hsdf])[show].to_string(index=False))
    print("\nFlagged-cluster timing (each Malacca cluster to the nearest Singapore cluster within 14 days):")
    print(pd.concat([cdf, hcdf]).to_string(index=False))

    # ----- chart -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    ax = axes[0]
    colors = {"all out-of-sample": ORANGE, "excluding first 90 days": BLUE, "from 2022 (also excl. COVID)": AQUA}
    base = xdf[(xdf.pair == "total") & (xdf.variant == "excluding first 90 days")]
    ax.fill_between(lags, base["null_lo"], base["null_hi"], color=BASELINE, alpha=0.5, label="95% null band (circular shifts)")
    for variant in VARIANTS:
        s = xdf[(xdf.pair == "total") & (xdf.variant == variant)]
        ax.plot(lags, s["r"], color=colors[variant], lw=2, marker="o", ms=3.5, label=variant)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.axvline(0, color=BASELINE, lw=1, ls="--")
    ax.set_title("Total traffic: Malacca z vs. Singapore port-call z", loc="left", color=INK, fontsize=11.5)
    ax.set_xlabel("Lag in days (positive = Malacca leads Singapore)", color=SECONDARY_INK)
    ax.set_ylabel("Correlation", color=SECONDARY_INK)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax = axes[1]
    pcolors = {"total": BLUE, "container": ORANGE, "tanker": AQUA}
    for name in PAIRS:
        s = xdf[(xdf.pair == name) & (xdf.variant == "excluding first 90 days")]
        ax.plot(lags, s["r"], color=pcolors[name], lw=2, marker="o", ms=3.5, label=name)
    ax.fill_between(lags, base["null_lo"], base["null_hi"], color=BASELINE, alpha=0.5)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.axvline(0, color=BASELINE, lw=1, ls="--")
    ax.set_title("By vessel type, excluding the first 90 days", loc="left", color=INK, fontsize=11.5)
    ax.set_xlabel("Lag in days (positive = Malacca leads Singapore)", color=SECONDARY_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for a_ in axes:
        a_.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.spines["left"].set_color(BASELINE)
        a_.spines["bottom"].set_color(BASELINE)
        a_.tick_params(colors=MUTED, labelsize=9)
        a_.grid(color=GRIDLINE, lw=0.8)
        a_.set_axisbelow(True)
    fig.text(0.01, -0.03, "Grey band = 95% of cross-correlations when the Singapore series is circularly shifted (alignment destroyed, autocorrelation kept). "
             "Singapore = Port of Singapore port calls; PortWatch has no Singapore Strait series.", fontsize=8.5, color=MUTED, ha="left")
    plt.tight_layout()
    fig.savefig("data/chart_lead_lag.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_lead_lag.png")


if __name__ == "__main__":
    main()
