"""
Part 5: are the flagged clusters specific to the one baseline?

Everything so far uses one detector (baseline.py): log count regressed on a linear trend, day of week and month, refit
quarterly on the trailing two years, z = forecast error / training-residual sd, flag |z| > 2.5. This script runs two
structurally different detectors on the same series and asks how much of the flagged set survives.

  B1. Weekly median. Forecast for day t = median of the log count on the same weekday over the previous 8 weeks.
      It has no trend, month or refit schedule and adapts within weeks. z = error / (1.4826 x median absolute
      deviation of the previous 56 days' one-step errors).
  B2. Seasonal decomposition. Weekly seasonal indices from the previous 52 weeks (log count minus its centred 7-day
      mean, averaged by weekday), a level from the mean of the last 28 de-seasonalised days, forecast = level +
      index for the weekday. z = error / (1.4826 x MAD of the previous 90 days' one-step errors).

Both are causal (every forecast uses only earlier days), cover the same out-of-sample days as the regression baseline
(from day 365), and use the same flagging rule, clustering and event matching, so the comparison is like for like. The
Malacca total is the main series; the Singapore port-call series gets the same treatment.

Comparisons: forecast accuracy (R^2 and mean absolute error on the log scale), z-score correlation, flagged days and
daily-flag agreement (Cohen's kappa), cluster-level overlap in both directions, event matching against the random
placement rate, an equal-budget comparison (each detector's top 184 days), the baseline edge window, and the Part 2
event study repeated with each detector's z.

    python second_detector.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import MIN_TRAIN_DAYS, THRESH, rolling_oos_fit, cluster_anomalies, matched_event
from event_study import event_paths, analyse, HALF, SHORT

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

N_CHANCE = 100
SEED = 0
MATCH_PAD = 3          # a cluster of one detector "has a counterpart" if the other has one overlapping +/- this many days


def weekly_median_z(y, dates):
    """B1. y is log counts on consecutive daily dates."""
    n = len(y)
    f = np.full(n, np.nan)
    for t in range(56, n):
        f[t] = np.median(y[t - 7 * np.arange(1, 9)])
    e = y - f
    z = np.full(n, np.nan)
    for t in range(56 + 56, n):
        past = e[t - 56:t]
        mad = 1.4826 * np.median(np.abs(past - np.median(past)))
        z[t] = e[t] / mad
    return f, z


def decomposition_z(y, dates):
    """B2. Weekly seasonal indices (previous 52 weeks) plus a 28-day de-seasonalised level."""
    n = len(y)
    dow = np.asarray(dates.dt.dayofweek)
    ma7 = pd.Series(y).rolling(7, center=True).mean().values       # centred 7-day mean (uses +/-3 days; only for indices of past days)
    f = np.full(n, np.nan)
    for t in range(364 + 3, n):
        lo = t - 364
        hi = t - 3                                                # keep the centred mean entirely in the past
        dev = y[lo:hi] - ma7[lo:hi]
        idx = np.array([np.nanmean(dev[dow[lo:hi] == d]) for d in range(7)])
        idx -= idx.mean()
        lvl = np.mean(y[t - 28:t] - idx[dow[t - 28:t]])
        f[t] = lvl + idx[dow[t]]
    e = y - f
    z = np.full(n, np.nan)
    for t in range(364 + 3 + 90, n):
        past = e[t - 90:t]
        mad = 1.4826 * np.median(np.abs(past - np.median(past)))
        z[t] = e[t] / mad
    return f, z


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def overlap_share(cl_a, cl_b, pad=MATCH_PAD):
    """Share of A's clusters that have a B cluster overlapping within +/- pad days."""
    if len(cl_a) == 0:
        return np.nan
    hit = 0
    for r in cl_a.itertuples():
        hit += bool(((cl_b["start"] - pd.Timedelta(days=pad) <= r.end) & (cl_b["end"] + pd.Timedelta(days=pad) >= r.start)).any())
    return hit / len(cl_a)


def main():
    rng = np.random.default_rng(SEED)
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"]).sort_values("fromdate")
    starts = list(events["fromdate"])
    names = [SHORT.get(n, n) for n in events["eventname"]]
    df = mal[["date", "n_total"]].merge(sg[["date", "portcalls"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    first_oos = df["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    span_days = (df["date"].max() - first_oos).days
    edge_end = first_oos + pd.Timedelta(days=90)
    oos_mask = (df["date"] >= first_oos).values

    def chance(clusters):
        hits = total = 0
        for _ in range(N_CHANCE):
            for r in clusters.itertuples():
                length = (r.end - r.start).days
                st = first_oos + pd.Timedelta(days=int(rng.integers(0, max(span_days - length, 1))))
                hits += matched_event(events, st, st + pd.Timedelta(days=length)) is not None
                total += 1
        return hits / total if total else np.nan

    summary_rows, clusters_out, zs_out = [], [], {}
    for series, col in (("Malacca total", "n_total"), ("Singapore port calls", "portcalls")):
        y = np.log(df[col].values.astype(float))
        fa, za, _, oos_a = rolling_oos_fit(df, col)
        fb1, zb1 = weekly_median_z(y, df["date"])
        fb2, zb2 = decomposition_z(y, df["date"])
        dets = {"A: trend + weekday + month, refit quarterly": (fa, za), "B1: weekly median": (fb1, zb1), "B2: seasonal decomposition": (fb2, zb2)}
        zs_out[series] = {k: pd.Series(v[1], index=df["date"]) for k, v in dets.items()}
        # common scored days: every detector defined
        common = oos_mask & np.all([~np.isnan(v[1]) for v in dets.values()], axis=0)
        print(f"\n=== {series}: {int(common.sum())} common out-of-sample days, {df['date'][common].min().date()} to {df['date'][common].max().date()} ===")
        flags, cls = {}, {}
        for name, (f, z) in dets.items():
            ys, fs = y[common], f[common]
            r2 = 1 - np.sum((ys - fs) ** 2) / np.sum((ys - ys.mean()) ** 2)
            mae = np.mean(np.abs(ys - fs))
            tmp = pd.DataFrame({"date": df["date"][common].values, "z": z[common]})
            _, cl = cluster_anomalies(tmp, "z")
            flag = (tmp["z"].abs() > THRESH).values
            flags[name], cls[name] = flag, cl
            m = int(sum(matched_event(events, r.start, r.end) is not None for r in cl.itertuples()))
            # excluding the baseline edge window
            tmp2 = tmp[tmp["date"] >= edge_end].reset_index(drop=True)
            _, cl2 = cluster_anomalies(tmp2, "z")
            m2 = int(sum(matched_event(events, r.start, r.end) is not None for r in cl2.itertuples()))
            row = {"series": series, "detector": name, "oos_R2_log": r2, "MAE_log": mae, "flagged_days": int(flag.sum()),
                   "flagged_share_%": 100 * flag.mean(), "clusters": len(cl), "matched": m,
                   "match_rate": m / len(cl) if len(cl) else np.nan, "chance_rate": chance(cl) if len(cl) else np.nan,
                   "clusters_ex_edge": len(cl2), "matched_ex_edge": m2,
                   "median_cluster_days": float(cl["n_days"].median()) if len(cl) else np.nan,
                   "longest_cluster_days": int(cl["n_days"].max()) if len(cl) else 0}
            summary_rows.append(row)
            for r in cl.itertuples():
                clusters_out.append({"series": series, "detector": name, "start": r.start, "end": r.end, "n_days": r.n_days,
                                     "max_abs_z": r.max_abs_z, "matched_event": matched_event(events, r.start, r.end)})
        names_d = list(dets)
        A = names_d[0]
        for b in names_d[1:]:
            zc = np.corrcoef(dets[A][1][common], dets[b][1][common])[0, 1]
            k = kappa(flags[A], flags[b])
            both = int((flags[A] & flags[b]).sum())
            jac = both / max(int((flags[A] | flags[b]).sum()), 1)
            ov_ab = overlap_share(cls[A], cls[b])
            ov_ba = overlap_share(cls[b], cls[A])
            # chance level for cluster overlap: shift B's flagged days circularly and recluster
            fb = flags[b]
            dates_c = df["date"][common].values
            sh_ab = []
            for _ in range(200):
                sh = np.roll(fb, int(rng.integers(30, len(fb) - 30)))
                tmp = pd.DataFrame({"date": dates_c, "z": np.where(sh, 3.0, 0.0)})
                _, clb = cluster_anomalies(tmp, "z")
                sh_ab.append(overlap_share(cls[A], clb))
            summary_rows[len(summary_rows) - 3 + names_d.index(b)].update({"z_corr_with_A": zc, "kappa_daily_flags_vs_A": k, "jaccard_flag_days_vs_A": jac,
                                     "A_clusters_with_counterpart": ov_ab, "counterpart_shifted_null_median": float(np.median(sh_ab)),
                                     "counterpart_shifted_null_95%": float(np.percentile(sh_ab, 95)),
                                     "this_clusters_with_A_counterpart": ov_ba})
            print(f"  {b} vs A: z correlation {zc:.2f}; daily flags kappa {k:.2f}, Jaccard {jac:.2f} ({both} days in both of "
                  f"{int(flags[A].sum())} A and {int(flags[b].sum())} B); {ov_ab*100:.0f}% of A's clusters have a B cluster within +/-{MATCH_PAD} days "
                  f"(shifted-flags null median {np.median(sh_ab)*100:.0f}%, 95th pct {np.percentile(sh_ab, 95)*100:.0f}%); "
                  f"{ov_ba*100:.0f}% of B's clusters have an A counterpart")
        # equal budget: top-K days by |z| in each detector
        K = int(flags[A].sum())
        tops = {}
        for name, (f, z) in dets.items():
            absz = np.abs(np.where(common, z, np.nan))
            order = np.argsort(-np.nan_to_num(absz, nan=-1))[:K]
            tops[name] = set(order)
        for b in names_d[1:]:
            inter = len(tops[A] & tops[b])
            print(f"  equal budget, top {K} days by |z|: {A.split(':')[0]} and {b.split(':')[0]} share {inter} ({inter / K * 100:.0f}%)")
            summary_rows[len(summary_rows) - 3 + names_d.index(b)]["equal_budget_top_days_shared_with_A"] = inter / K
        # edge window and the largest anomalies
        for name, (f, z) in dets.items():
            zz = pd.Series(z, index=df["date"])
            win = zz[(zz.index >= first_oos) & (zz.index < edge_end)].dropna()
            print(f"  {name}: mean z in the first 90 days {win.mean():+.2f} (n={len(win)}), flagged days there {int((win.abs() > THRESH).sum())}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv("data/second_detector_summary.csv", index=False)
    pd.DataFrame(clusters_out).to_csv("data/second_detector_clusters.csv", index=False)
    print("\n=== Summary ===")
    print(summ[["series", "detector", "oos_R2_log", "MAE_log", "flagged_days", "clusters", "matched", "match_rate", "chance_rate",
                "clusters_ex_edge", "matched_ex_edge", "median_cluster_days", "longest_cluster_days"]].round(3).to_string(index=False))

    # ------------------------------------------------------------- event study with each detector's z (Malacca total)
    print("\n=== Event study (Malacca total) with each detector ===")
    ev_rows = []
    for name, z in zs_out["Malacca total"].items():
        zz = z[z.index >= first_oos].dropna()
        full = zz.reindex(pd.date_range(zz.index.min(), zz.index.max(), freq="D"))
        paths = event_paths(full, starts)
        eligible = full.index[(full.index >= full.index.min() + pd.Timedelta(days=HALF)) & (full.index <= full.index.max() - pd.Timedelta(days=HALF))]
        eligible = eligible[full.reindex(eligible).notna()]
        r6 = analyse(paths, eligible, full, np.random.default_rng(SEED), "All six events")
        r5 = analyse(paths[1:], eligible, full, np.random.default_rng(SEED), "Without COVID")
        near = (np.arange(-HALF, HALF + 1) >= -14) & (np.arange(-HALF, HALF + 1) <= 14)
        flagged = [bool(np.nanmax(np.where(near, np.abs(p), np.nan)) > THRESH) for p in paths]
        for r, n_ev in ((r6, 6), (r5, 5)):
            ev_rows.append({"detector": name, "set": r["label"], "post_minus_pre_signed_z": r["d_s"], "signed_placebo_p": r["p_s"],
                            "post_minus_pre_abs_z": r["d_a"], "abs_placebo_p": r["p_a"],
                            "events_flagged_within_14d": int(sum(flagged)) if n_ev == 6 else int(sum(flagged[1:]))})
            print(f"  {name} ({r['label']}): signed z change {r['d_s']:+.2f} (placebo p {r['p_s']:.2f}); |z| change {r['d_a']:+.2f} (p {r['p_a']:.2f}); "
                  f"events with a flagged day within 14 days: {ev_rows[-1]['events_flagged_within_14d']} of {n_ev}")
    pd.DataFrame(ev_rows).to_csv("data/second_detector_event_study.csv", index=False)

    # ------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4), facecolor=SURFACE)
    zm = zs_out["Malacca total"]
    cols = {"A: trend + weekday + month, refit quarterly": BLUE, "B1: weekly median": ORANGE, "B2: seasonal decomposition": AQUA}
    ax = axes[0, 0]
    for name, col in cols.items():
        z = zm[name]
        z = z[z.index >= first_oos].dropna()
        ax.plot(z.index, z.values, color=col, lw=0.7, alpha=0.85, label=name)
    ax.axhline(THRESH, color=INK, lw=0.8, ls="--")
    ax.axhline(-THRESH, color=INK, lw=0.8, ls="--")
    for _, e in events.iterrows():
        ax.axvline(e["fromdate"], color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(-12, 9)
    ax.set_ylabel("z-score", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. Malacca total: z-score by detector (dotted: event starts)", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[0, 1]
    sm = summ[summ["series"] == "Malacca total"]
    x = np.arange(len(sm))
    ax.bar(x - 0.2, 100 * sm["match_rate"], 0.4, color=[cols[d] for d in sm["detector"]], label="matched to an event")
    ax.bar(x + 0.2, 100 * sm["chance_rate"], 0.4, color=MUTED, label="random placement")
    for xi, (m, c, n) in enumerate(zip(sm["matched"], sm["clusters"], sm["clusters"])):
        ax.text(xi - 0.2, 100 * sm["match_rate"].iloc[xi] + 1, f"{m}/{c}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B1\nweekly median", "B2\ndecomposition"])
    ax.set_ylabel("Clusters within 14 days of an event (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylim(0, 100)
    ax.set_title("B. Event matching, Malacca total", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1, 0]
    labs, vals, nulls = [], [], []
    for series in ("Malacca total", "Singapore port calls"):
        for b, short in (("B1: weekly median", "B1"), ("B2: seasonal decomposition", "B2")):
            r = summ[(summ["series"] == series) & (summ["detector"] == b)].iloc[0]
            labs.append(f"{series.split()[0]}\n{short}")
            vals.append(100 * r["A_clusters_with_counterpart"])
            nulls.append(100 * r["counterpart_shifted_null_95%"])
    x = np.arange(len(labs))
    ax.bar(x, vals, 0.5, color=BLUE)
    ax.scatter(x, nulls, marker="_", s=900, color=RED, zorder=3, label="95th percentile if B's flags were shifted at random")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("A clusters with a B cluster within 3 days (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylim(0, 100)
    ax.set_title("C. How much of A's flagged set the other detectors reproduce", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1, 1]
    ss = summ[summ["series"] == "Singapore port calls"]
    x = np.arange(3)
    ax.bar(x - 0.2, sm["flagged_days"], 0.4, color=BLUE, label="Malacca total")
    ax.bar(x + 0.2, ss["flagged_days"], 0.4, color=ORANGE, label="Singapore port calls")
    for xi, (a_, b_) in enumerate(zip(sm["flagged_days"], ss["flagged_days"])):
        ax.text(xi - 0.2, a_ + 2, str(a_), ha="center", fontsize=8.5, color=INK)
        ax.text(xi + 0.2, b_ + 2, str(b_), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B1\nweekly median", "B2\ndecomposition"])
    ax.set_ylabel("Days with |z| > 2.5 (same 2,347 days)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Flagged days by detector", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_second_detector.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_second_detector.png")


if __name__ == "__main__":
    main()
