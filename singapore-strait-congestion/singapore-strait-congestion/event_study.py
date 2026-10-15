"""
Event study around the six documented disruptions: what does the Malacca anomaly (the out-of-sample z-score from
baseline.py) actually do in the days before and after each event starts, and is the average path any different
from what the same procedure gives for randomly placed dates?

The headline result in the README (64% of flagged clusters within 14 days of a documented event, vs 56% by chance)
asks "are the anomalies near events?". This asks the other direction: "are the events accompanied by anomalies,
and what does the path look like around them?" For each event, the z-score is lined up on the event's start date
(day 0) from 45 days before to 45 days after, then averaged across events.

Two measures are averaged, because they answer different questions and signed z can cancel across events:
  - signed z: is traffic systematically below (negative) or above (positive) its forecast around events?
  - |z|: is traffic unusually far from forecast in either direction, i.e. is there more anomaly, whatever its sign?

Two kinds of uncertainty band are drawn, and they mean different things:
  - a 95% confidence interval across the events (t-distribution, so with 6 events it is wide);
  - a placebo envelope: the 2.5th-97.5th percentile of the same cross-event average when the event dates are
    drawn at random from the out-of-sample period instead. This is the band that says what "nothing special
    happens around events" looks like for this z-score, which is autocorrelated and fat-tailed, so it is the
    fairer yardstick than a t-interval. Any event whose window is truncated by the data (COVID starts only 22 days
    after the out-of-sample period does) is truncated identically on its placebo counterpart.

A caveat found while building this, and why every result is shown twice (all six events, and without COVID): the
COVID window sits in the baseline's first fitting window. The very first model is fitted on one year of data
(2019) with about 20 parameters, and it over-forecasts January 2020 by roughly 30% even though actual traffic
was about 4% ABOVE January 2019. So the COVID event's z-score is already around -3 before the event begins,
which an event cannot cause. That is an edge artifact of the baseline, not a traffic drop, and it matters beyond
this script (see baseline_edge_check below). The five events from November 2023 on all have full +/-45-day
windows and a baseline fitted on two years of data.

Reuses baseline.rolling_oos_fit unchanged (the z-scores are the same ones analysis.py, model_selection.py and
sensitivity_analysis.py use) and the same Malacca/Singapore date alignment as sensitivity_analysis.py.

    python event_study.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from baseline import MIN_TRAIN_DAYS, REFIT_FREQ_DAYS, THRESH, rolling_oos_fit, cluster_anomalies, matched_event

HALF = 45                  # days either side of the event start
PRE = (-30, -1)            # summary windows, in days relative to the event start
POST = (0, 30)
FLAG_WINDOW = 14           # "flagged near the event" = any |z| > THRESH within +/- this many days of the start
N_PLACEBO = 2000
SEED = 0

# dataviz skill palette (references/palette.md), as in sensitivity_analysis.py: blue/orange categorical slots,
# diverging blue<->red with a neutral midpoint for the signed heatmap.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SHORT = {
    "COVID-19 GLOBAL SHIPPING DISRUPTION": "COVID-19 (Jan 2020)",
    "RESTRICTIONS PANAMA CANAL": "Panama Canal restrictions (Nov 2023)",
    "RED SEA TENSIONS": "Red Sea tensions (Dec 2023)",
    "COLLAPSE FRANCIS SCOTT KEY BRIDGE": "Key Bridge collapse (Mar 2024)",
    "US PORTS STRIKE": "US ports strike (Oct 2024)",
    "HORMUZ-26": "Hormuz (Mar 2026)",
}


def load_data():
    """The merged Malacca/Singapore frame, the out-of-sample z-score series (indexed by date, with missing days as
    NaN so day offsets never silently shift), and the forecast, as in sensitivity_analysis.py."""
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    df = mal[["date", "n_total"]].merge(sg[["date"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    fitted, z, _, _ = rolling_oos_fit(df, "n_total")
    df["z"] = z
    df["forecast"] = np.exp(fitted)
    s = pd.Series(z, index=df["date"]).dropna()
    return df, s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


def event_paths(z, starts, half=HALF):
    """(n_events, 2*half+1) array of z lined up on each start date; NaN where the z-score does not exist."""
    offsets = np.arange(-half, half + 1)
    return np.array([z.reindex(pd.DatetimeIndex(s + pd.to_timedelta(offsets, unit="D"))).values for s in starts])


def window_mean(paths, window):
    lo, hi = window
    return np.nanmean(paths[:, lo + HALF:hi + HALF + 1], axis=1)


def cross_event_summary(paths):
    """Per-day mean across events and its 95% t-interval (NaN-aware: events with no data that day are skipped)."""
    n = np.sum(~np.isnan(paths), axis=0)
    mean = np.nanmean(paths, axis=0)
    sd = np.nanstd(paths, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n)
    half[n < 2] = np.nan
    return mean, mean - half, mean + half, n


def analyse(paths, eligible, z, rng, label):
    """Cross-event mean paths with t-intervals and placebo envelopes, plus the post-minus-pre summary and its
    placebo p-values, for one set of events. The placebo events inherit each real event's missing days."""
    offsets = np.arange(-HALF, HALF + 1)
    masks = np.isnan(paths)
    abs_paths = np.abs(paths)
    out = {"label": label, "n": len(paths)}
    out["mean_s"], out["lo_s"], out["hi_s"], out["n_per_day"] = cross_event_summary(paths)
    out["mean_a"], out["lo_a"], out["hi_a"], _ = cross_event_summary(abs_paths)
    pl_s = np.empty((N_PLACEBO, len(offsets)))
    pl_a = np.empty_like(pl_s)
    for i in range(N_PLACEBO):
        draw = rng.choice(len(eligible), size=len(paths), replace=True)
        p = event_paths(z, list(eligible[draw]))
        p[masks] = np.nan
        pl_s[i], pl_a[i] = np.nanmean(p, axis=0), np.nanmean(np.abs(p), axis=0)
    out["env_s"] = np.nanpercentile(pl_s, [2.5, 97.5], axis=0)
    out["env_a"] = np.nanpercentile(pl_a, [2.5, 97.5], axis=0)

    pre_s, post_s = window_mean(paths, PRE), window_mean(paths, POST)
    pre_a, post_a = window_mean(abs_paths, PRE), window_mean(abs_paths, POST)
    d_s, d_a = np.mean(post_s - pre_s), np.mean(post_a - pre_a)

    def pdiff(pl):
        return (np.nanmean(pl[:, POST[0] + HALF:POST[1] + HALF + 1], axis=1)
                - np.nanmean(pl[:, PRE[0] + HALF:PRE[1] + HALF + 1], axis=1))
    pd_s, pd_a = pdiff(pl_s), pdiff(pl_a)
    out.update({
        "d_s": d_s, "d_a": d_a,
        "p_s": float(np.mean(np.abs(pd_s) >= abs(d_s))), "p_a": float(np.mean(pd_a >= d_a)),
        "ci_s": stats.t.interval(0.95, len(paths) - 1, loc=d_s, scale=stats.sem(post_s - pre_s)),
        "ci_a": stats.t.interval(0.95, len(paths) - 1, loc=d_a, scale=stats.sem(post_a - pre_a)),
        "pl_s_range": np.percentile(pd_s, [2.5, 97.5]), "pl_a_range": np.percentile(pd_a, [2.5, 97.5]),
        "pre_s": pre_s, "post_s": post_s, "pre_a": pre_a, "post_a": post_a})
    print(f"  {label}: post (0 to +{POST[1]} d) minus pre ({PRE[0]} to {PRE[1]} d), n = {len(paths)} events")
    print(f"    signed z: {d_s:+.2f}  (95% CI across events {out['ci_s'][0]:+.2f} to {out['ci_s'][1]:+.2f}; placebo 95% range "
          f"{out['pl_s_range'][0]:+.2f} to {out['pl_s_range'][1]:+.2f}; two-sided placebo p = {out['p_s']:.3f})")
    print(f"    |z|     : {d_a:+.2f}  (95% CI across events {out['ci_a'][0]:+.2f} to {out['ci_a'][1]:+.2f}; placebo 95% range "
          f"{out['pl_a_range'][0]:+.2f} to {out['pl_a_range'][1]:+.2f}; one-sided placebo p = {out['p_a']:.3f})")
    return out


def baseline_edge_check(df, events):
    """How much of the headline match rate sits in the baseline's first fitting window (the first REFIT_FREQ_DAYS
    days of the out-of-sample period, scored by a model fitted on a single year)?"""
    t0 = df["date"].min()
    first_oos = t0 + pd.Timedelta(days=MIN_TRAIN_DAYS)
    oos = df[df["date"] >= first_oos].dropna(subset=["z"]).sort_values("date").reset_index(drop=True)
    oos, clusters = cluster_anomalies(oos, "z")
    clusters["matched"] = [matched_event(events, r.start, r.end) is not None for r in clusters.itertuples()]
    edge_end = first_oos + pd.Timedelta(days=REFIT_FREQ_DAYS)
    in_edge = clusters["start"] < edge_end
    print(f"  Clusters overall: {len(clusters)}, matched within 14 days of an event: {int(clusters['matched'].sum())} "
          f"({clusters['matched'].mean()*100:.0f}%)")
    print(f"  Clusters that start in the first {REFIT_FREQ_DAYS} out-of-sample days ({first_oos.date()} to "
          f"{(edge_end - pd.Timedelta(days=1)).date()}): {int(in_edge.sum())}, of which matched: {int(clusters.loc[in_edge, 'matched'].sum())}")
    rest = clusters[~in_edge]
    print(f"  Excluding that first window: {len(rest)} clusters, {int(rest['matched'].sum())} matched ({rest['matched'].mean()*100:.0f}%)")
    j = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2020-01-22")]
    j19 = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2019-01-22")]
    print(f"  January 1-22 2020: actual mean {j['n_total'].mean():.0f} vessels/day vs {j19['n_total'].mean():.0f} in January 2019 "
          f"({(j['n_total'].mean()/j19['n_total'].mean()-1)*100:+.1f}%), but the baseline forecast {j['forecast'].mean():.0f} "
          f"({(j['forecast'].mean()/j19['n_total'].mean()-1)*100:+.0f}% vs. 2019)")
    return {"n_clusters": len(clusters), "n_matched": int(clusters["matched"].sum()),
            "n_edge": int(in_edge.sum()), "n_edge_matched": int(clusters.loc[in_edge, "matched"].sum()),
            "n_rest": len(rest), "n_rest_matched": int(rest["matched"].sum())}


def main():
    df, z = load_data()
    events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"]).sort_values("fromdate")
    starts = list(events["fromdate"])
    names = [SHORT.get(n, n) for n in events["eventname"]]
    offsets = np.arange(-HALF, HALF + 1)
    first_oos, last_oos = z.index.min(), z.index.max()
    print(f"Out-of-sample z-scores: {z.notna().sum()} days, {first_oos.date()} to {last_oos.date()}; "
          f"{len(starts)} documented events, windows of +/-{HALF} days around each start date.")

    paths = event_paths(z, starts)
    avail_pre = [int(np.sum(~np.isnan(p[:HALF]))) for p in paths]
    avail_post = [int(np.sum(~np.isnan(p[HALF + 1:]))) for p in paths]
    for name, s, a, b in zip(names, starts, avail_pre, avail_post):
        note = "" if a == HALF and b == HALF else "   <-- window truncated by the data"
        print(f"  {name:<40} start {s.date()}  days available: {a}/{HALF} before, {b}/{HALF} after{note}")

    print("\nBaseline edge check (why COVID is shown separately):")
    edge = baseline_edge_check(df, events)

    rng = np.random.default_rng(SEED)
    eligible = z.index[(z.index >= first_oos + pd.Timedelta(days=HALF)) & (z.index <= last_oos - pd.Timedelta(days=HALF))]
    eligible = eligible[z.reindex(eligible).notna()]

    print("\nCross-event averages:")
    all6 = analyse(paths, eligible, z, rng, "All six events")
    ex = analyse(paths[1:], eligible, z, rng, "Without COVID (the five events from Nov 2023)")

    # ---------------------------------------------------------------------------- per-event table
    abs_paths = np.abs(paths)
    pre_s, post_s = window_mean(paths, PRE), window_mean(paths, POST)
    pre_a, post_a = window_mean(abs_paths, PRE), window_mean(abs_paths, POST)
    near = (offsets >= -FLAG_WINDOW) & (offsets <= FLAG_WINDOW)
    rows = []
    for i, (name, s) in enumerate(zip(names, starts)):
        win = np.where(near, abs_paths[i], np.nan)
        rows.append({
            "event": name, "start": s.date(), "days_before": avail_pre[i], "days_after": avail_post[i],
            "mean_z_pre": pre_s[i], "mean_z_post": post_s[i], "change_z": post_s[i] - pre_s[i],
            "mean_abs_z_pre": pre_a[i], "mean_abs_z_post": post_a[i],
            "peak_abs_z_within_14d": np.nanmax(win), "peak_day_offset": int(offsets[np.nanargmax(win)]),
            "flagged_within_14d": bool(np.nanmax(win) > THRESH), "days_flagged_within_14d": int(np.nansum(win > THRESH))})
    table = pd.DataFrame(rows)
    table.to_csv("data/event_study_per_event.csv", index=False)
    print("\nPer event:")
    print(table.round(2).to_string(index=False))
    rand_flag = [np.nanmax(np.abs(event_paths(z, [eligible[rng.integers(len(eligible))]])[0])[near]) > THRESH
                 for _ in range(N_PLACEBO)]
    n_flag = int(table["flagged_within_14d"].sum())
    print(f"\n{n_flag} of {len(table)} events have at least one |z| > {THRESH} within +/-{FLAG_WINDOW} days of their start; "
          f"for a randomly placed date the chance of that is {np.mean(rand_flag)*100:.0f}%. Excluding COVID (whose window is "
          f"contaminated by the baseline edge): {n_flag - int(table.loc[0, 'flagged_within_14d'])} of {len(table) - 1}.")

    paths_out = pd.DataFrame({"day_offset": offsets, "n_events": all6["n_per_day"],
                              "mean_z": all6["mean_s"], "ci_lo_z": all6["lo_s"], "ci_hi_z": all6["hi_s"],
                              "placebo_lo_z": all6["env_s"][0], "placebo_hi_z": all6["env_s"][1],
                              "mean_abs_z": all6["mean_a"], "ci_lo_abs_z": all6["lo_a"], "ci_hi_abs_z": all6["hi_a"],
                              "placebo_lo_abs_z": all6["env_a"][0], "placebo_hi_abs_z": all6["env_a"][1],
                              "exCOVID_mean_z": ex["mean_s"], "exCOVID_ci_lo_z": ex["lo_s"], "exCOVID_ci_hi_z": ex["hi_s"],
                              "exCOVID_placebo_lo_z": ex["env_s"][0], "exCOVID_placebo_hi_z": ex["env_s"][1],
                              "exCOVID_mean_abs_z": ex["mean_a"], "exCOVID_ci_lo_abs_z": ex["lo_a"], "exCOVID_ci_hi_abs_z": ex["hi_a"],
                              "exCOVID_placebo_lo_abs_z": ex["env_a"][0], "exCOVID_placebo_hi_abs_z": ex["env_a"][1]})
    for name, p in zip(names, paths):
        paths_out[f"z: {name}"] = p
    paths_out.to_csv("data/event_study_paths.csv", index=False)
    pd.DataFrame([{"set": r["label"], "n_events": r["n"], "post_minus_pre_signed_z": r["d_s"],
                   "signed_ci_lo": r["ci_s"][0], "signed_ci_hi": r["ci_s"][1], "signed_placebo_lo": r["pl_s_range"][0],
                   "signed_placebo_hi": r["pl_s_range"][1], "signed_placebo_p": r["p_s"],
                   "post_minus_pre_abs_z": r["d_a"], "abs_ci_lo": r["ci_a"][0], "abs_ci_hi": r["ci_a"][1],
                   "abs_placebo_lo": r["pl_a_range"][0], "abs_placebo_hi": r["pl_a_range"][1], "abs_placebo_p": r["p_a"]}
                  for r in (all6, ex)]).to_csv("data/event_study_summary.csv", index=False)
    pd.DataFrame([edge]).to_csv("data/event_study_baseline_edge.csv", index=False)

    # ------------------------------------------------------------------------------------- chart
    fig = plt.figure(figsize=(14, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.38, wspace=0.18)

    def band_panel(ax, key, ylabel, title, color, signed):
        s = "s" if signed else "a"
        ax.fill_between(offsets, all6[f"env_{s}"][0], all6[f"env_{s}"][1], color=NEUTRAL, zorder=1,
                        label="placebo: 95% range for 6 random dates")
        ax.fill_between(offsets, all6[f"lo_{s}"], all6[f"hi_{s}"], color=color, alpha=0.25, zorder=2, lw=0,
                        label="95% CI across the 6 events")
        ax.plot(offsets, all6[f"mean_{s}"], color=color, lw=2, zorder=3, label="mean, all six events")
        ax.plot(offsets, ex[f"mean_{s}"], color=INK, lw=1.6, ls="--", zorder=3, label="mean, without COVID")
        ax.axvline(0, color=INK, lw=0.8, ls=":", zorder=4)
        if signed:
            ax.axhline(0, color=BASELINE, lw=1, zorder=0)
        ax.set_xlabel("Days from event start", fontsize=10, color=SECONDARY_INK)
        ax.set_ylabel(ylabel, fontsize=10, color=SECONDARY_INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(-HALF, HALF)

    ax1 = fig.add_subplot(gs[0, 0])
    band_panel(ax1, "s", "Mean z-score (signed)", "Signed anomaly: below (-) or above (+) forecast", BLUE, True)
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax2 = fig.add_subplot(gs[0, 1])
    band_panel(ax2, "a", "Mean |z|", "Anomaly size, either direction", ORANGE, False)

    ax3 = fig.add_subplot(gs[1, :])
    cmap = LinearSegmentedColormap.from_list("diverging", [RED, NEUTRAL, BLUE])
    vmax = np.nanmax(np.abs(paths))
    im = ax3.imshow(np.ma.masked_invalid(paths), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                    extent=[-HALF - 0.5, HALF + 0.5, len(starts) - 0.5, -0.5])
    ax3.axvline(0, color=INK, lw=0.8, ls=":")
    ax3.set_yticks(range(len(starts)))
    ax3.set_yticklabels(names, fontsize=9, color=INK)
    ax3.set_xlabel("Days from event start (grey = no out-of-sample z-score that day)", fontsize=10, color=SECONDARY_INK)
    ax3.set_title("Each event on its own: daily z-score (COVID is negative before day 0: a baseline edge artifact, see README)",
                  fontsize=11, color=INK, loc="left", fontweight="bold")
    ax3.set_facecolor("#d9d8d2")
    cbar = fig.colorbar(im, ax=ax3, shrink=0.9, pad=0.015)
    cbar.set_label("z-score (blue above forecast, red below)", fontsize=9, color=SECONDARY_INK)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRIDLINE, lw=0.8)
        ax.set_facecolor(SURFACE)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(colors=MUTED, labelsize=9, length=0)

    fig.suptitle("Malacca anomaly around the six documented disruptions", fontsize=13, color=INK, x=0.01, ha="left", y=0.97)
    fig.text(0.01, 0.015,
              "z-scores from the rolling out-of-sample baseline (baseline.py), lined up on each event's start date. Windows overlap for "
              "events less than 90 days apart (Panama / Red Sea / Key Bridge), and open-ended events (Red Sea, Hormuz) have no defined end.",
              fontsize=8.5, color=MUTED, ha="left")
    fig.savefig("data/chart_event_study.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_event_study.png and the data/event_study_*.csv files")


if __name__ == "__main__":
    main()
