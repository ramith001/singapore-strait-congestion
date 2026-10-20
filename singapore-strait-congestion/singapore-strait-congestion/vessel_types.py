"""
Part 4: does the anomaly story differ by vessel type?

The saved PortWatch files carry counts by vessel type (Malacca transits: container, dry bulk, general cargo, roro,
tanker; Singapore port calls: the same five), so no new data is needed. Part 3 used the total, container and tanker
columns only for the Malacca-Singapore lead-lag. Here every type goes through the identical pipeline as the total
(baseline.py, unchanged): log counts, trend + day-of-week + month re-fit quarterly on the trailing 2 years, |z| > 2.5,
clusters, matched to the documented events within 14 days.

  A. Inventory: mean daily count and share of the total per type and site.
  B. Baseline quality and flagged clusters per series, with the match rate against the random-placement chance
     rate, over all out-of-sample days and excluding the first 90 days (the baseline edge window of Part 2).
  C. Event by type: change in mean z from the 30 days before each event start to the 30 days from it, with a
     placebo p-value (the same change for 1,000 randomly placed dates in the same series).
  D. Which types move with the total: same-day z correlation with the total, and the share of the total's flagged
     days that are also flagged in each type.
  E. The same type at the two sites: same-day z correlation of Malacca and Singapore, by type (Part 3 did this for
     the total, container and tanker).

Series with small counts (Singapore dry bulk 2.3 calls a day with 341 zero days, roro 3.7 with 49, general cargo 5.7
with 11; Malacca roro 6.3 with 17) cannot be logged as they are, so every type series is fitted as log(count + 1);
the totals are fitted as in the rest of the repo. For such small counts the Gaussian z-score is a poor description
of the noise, which the results table shows (high flagged-day shares, low R^2).

    python vessel_types.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import MIN_TRAIN_DAYS, THRESH, rolling_oos_fit, cluster_anomalies, matched_event
from event_study import event_paths, window_mean, HALF, PRE, POST, SHORT

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

TYPES = ["container", "dry_bulk", "general_cargo", "roro", "tanker"]
TYPE_LABEL = {"container": "Container", "dry_bulk": "Dry bulk", "general_cargo": "Gen. cargo", "roro": "Roro",
              "tanker": "Tanker", "total": "Total"}
MAL = {t: f"n_{t}" for t in TYPES} | {"total": "n_total"}
SGP = {t: f"portcalls_{t}" for t in TYPES} | {"total": "portcalls"}
EDGE_DAYS = 90
N_PLACEBO = 1000
N_CHANCE = 100
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"]).sort_values("fromdate")
    starts = list(events["fromdate"])
    names = [SHORT.get(n, n) for n in events["eventname"]]
    cols_m = ["date"] + list(MAL.values())
    cols_s = ["date"] + list(SGP.values())
    df = mal[cols_m].merge(sg[cols_s], on="date", how="inner").sort_values("date").reset_index(drop=True)
    first_oos = df["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    last_day = df["date"].max()
    span_days = (last_day - first_oos).days
    edge_end = first_oos + pd.Timedelta(days=EDGE_DAYS)

    # ----------------------------------------------------------------------------------- A. inventory
    inv = []
    for site, cols in (("Malacca", MAL), ("Singapore", SGP)):
        tot = df[cols["total"]].mean()
        for t in TYPES + ["total"]:
            c = df[cols[t]]
            inv.append({"site": site, "type": t, "mean_per_day": c.mean(), "share_of_total_%": 100 * c.mean() / tot if t != "total" else 100.0,
                        "min": int(c.min()), "zero_days": int((c == 0).sum())})
    inv = pd.DataFrame(inv)
    inv.to_csv("data/vessel_type_inventory.csv", index=False)
    print("=== A. Mean daily counts by type ===")
    print(inv.round(1).to_string(index=False))

    # ----------------------------------------------------------------- B. baseline and clusters per series
    def chance(clusters):
        hits = total = 0
        for _ in range(N_CHANCE):
            for r in clusters.itertuples():
                length = (r.end - r.start).days
                st = first_oos + pd.Timedelta(days=int(rng.integers(0, max(span_days - length, 1))))
                hits += matched_event(events, st, st + pd.Timedelta(days=length)) is not None
                total += 1
        return hits / total if total else np.nan

    series = {}      # (site, type) -> z indexed by date
    rows = []
    for site, cols in (("Malacca", MAL), ("Singapore", SGP)):
        for t in TYPES + ["total"]:
            col = cols[t]
            work = df[["date"]].copy()
            work["y"] = df[col] + (1 if t != "total" else 0)
            fitted, z, train_r2, oos_r2 = rolling_oos_fit(work, "y")
            zs = pd.Series(z, index=df["date"])
            series[(site, t)] = zs
            row = {"site": site, "type": t, "train_R2": train_r2, "oos_R2": oos_r2}
            for tag, start in (("all", first_oos), ("ex_edge", edge_end)):
                tmp = pd.DataFrame({"date": df["date"], "z": z})
                tmp = tmp[tmp["date"] >= start].dropna(subset=["z"]).reset_index(drop=True)
                _, cl = cluster_anomalies(tmp, "z")
                m = int(sum(matched_event(events, r.start, r.end) is not None for r in cl.itertuples()))
                row.update({f"flagged_days_{tag}": int((tmp["z"].abs() > THRESH).sum()), f"scored_days_{tag}": len(tmp),
                            f"clusters_{tag}": len(cl), f"matched_{tag}": m,
                            f"match_rate_{tag}": m / len(cl) if len(cl) else np.nan,
                            f"chance_rate_{tag}": chance(cl) if len(cl) else np.nan})
            rows.append(row)
    base = pd.DataFrame(rows)
    base["flagged_share_ex_edge_%"] = 100 * base["flagged_days_ex_edge"] / base["scored_days_ex_edge"]
    base.to_csv("data/vessel_type_baseline.csv", index=False)
    print("\n=== B. Baseline and flagged clusters by series (|z| > %.1f) ===" % THRESH)
    show = base[["site", "type", "train_R2", "oos_R2", "flagged_share_ex_edge_%", "clusters_ex_edge", "matched_ex_edge",
                 "match_rate_ex_edge", "chance_rate_ex_edge"]]
    print(show.round(3).to_string(index=False))

    # ---------------------------------------------------------------------------- C. event by type
    ev_rows = []
    for (site, t), zs in series.items():
        z = zs[zs.index >= first_oos].dropna()
        full = z.reindex(pd.date_range(z.index.min(), z.index.max(), freq="D"))
        eligible = full.index[(full.index >= full.index.min() + pd.Timedelta(days=HALF)) &
                              (full.index <= full.index.max() - pd.Timedelta(days=HALF))]
        eligible = eligible[full.reindex(eligible).notna()]
        paths = event_paths(full, starts)
        d = window_mean(paths, POST) - window_mean(paths, PRE)
        # placebo changes for single random dates
        pp = event_paths(full, list(eligible[rng.integers(0, len(eligible), N_PLACEBO)]))
        pl = window_mean(pp, POST) - window_mean(pp, PRE)
        for name, di in zip(names, d):
            ev_rows.append({"site": site, "type": t, "event": name, "change_z": di,
                            "placebo_p": float((np.sum(np.abs(pl) >= abs(di)) + 1) / (len(pl) + 1)) if np.isfinite(di) else np.nan})
    ev = pd.DataFrame(ev_rows)
    ev.to_csv("data/vessel_type_event_changes.csv", index=False)
    typed = ev[ev["type"] != "total"].dropna(subset=["change_z"])
    ncells = len(typed)
    nsig = int((typed["placebo_p"] < 0.05).sum())
    print("\n=== C. Change in mean z, 30 days from the event start vs 30 days before ===")
    for site in ("Malacca", "Singapore"):
        pv = ev[ev["site"] == site].pivot(index="event", columns="type", values="change_z").reindex(names)[TYPES + ["total"]]
        print(f"\n{site}:")
        print(pv.round(2).to_string())
    print(f"\n{nsig} of {ncells} event-by-type cells have placebo p < 0.05 (about {0.05*ncells:.1f} expected by chance); ")
    print(typed[typed["placebo_p"] < 0.05].sort_values("placebo_p").round(3).to_string(index=False))

    # ------------------------------------------------------------------ D. who moves with the total
    d_rows = []
    for site in ("Malacca", "Singapore"):
        zt = series[(site, "total")]
        mask = (zt.index >= edge_end)
        for t in TYPES:
            zz = series[(site, t)]
            ok = mask & zt.notna().values & zz.notna().values
            both = ((zt[ok].abs() > THRESH) & (zz[ok].abs() > THRESH)).sum()
            n_tot = (zt[ok].abs() > THRESH).sum()
            n_typ = (zz[ok].abs() > THRESH).sum()
            same_sign = ((zt[ok] * zz[ok]) > 0)[(zt[ok].abs() > THRESH) & (zz[ok].abs() > THRESH)].mean() if both else np.nan
            d_rows.append({"site": site, "type": t, "corr_with_total_z": float(np.corrcoef(zt[ok], zz[ok])[0, 1]),
                           "total_flagged_days": int(n_tot), "also_flagged_in_type": int(both),
                           "share_of_total_flags_%": 100 * both / n_tot if n_tot else np.nan,
                           "type_flagged_days": int(n_typ), "share_of_type_flags_in_total_%": 100 * both / n_typ if n_typ else np.nan,
                           "same_sign_share": same_sign})
    dd = pd.DataFrame(d_rows)
    dd.to_csv("data/vessel_type_vs_total.csv", index=False)
    print("\n=== D. Each type against the total (excluding the first 90 days) ===")
    print(dd.round(2).to_string(index=False))

    # ------------------------------------------------------------------ E. the same type at both sites
    e_rows = []
    for t in TYPES + ["total"]:
        zm, zs_ = series[("Malacca", t)], series[("Singapore", t)]
        ok = (zm.index >= edge_end) & zm.notna().values & zs_.notna().values
        r0 = float(np.corrcoef(zm[ok], zs_[ok])[0, 1])
        # circular-shift null for the zero-lag correlation, as in Part 3
        a, b = zm[ok].values, zs_[ok].values
        null = np.array([np.corrcoef(a, np.roll(b, int(rng.integers(30, len(b) - 30))))[0, 1] for _ in range(1000)])
        e_rows.append({"type": t, "corr_malacca_singapore": r0, "null_2.5%": np.percentile(null, 2.5), "null_97.5%": np.percentile(null, 97.5),
                       "above_null": bool(r0 > np.percentile(null, 97.5))})
    ee = pd.DataFrame(e_rows)
    ee.to_csv("data/vessel_type_cross_site.csv", index=False)
    print("\n=== E. Same type, Malacca vs Singapore, zero-lag z correlation (excluding the first 90 days) ===")
    print(ee.round(3).to_string(index=False))

    # ----------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), facecolor=SURFACE)
    ax = axes[0, 0]
    x = np.arange(len(TYPES))
    for off, site, col in ((-0.2, "Malacca", BLUE), (0.2, "Singapore", ORANGE)):
        v = [inv[(inv["site"] == site) & (inv["type"] == t)]["mean_per_day"].iloc[0] for t in TYPES]
        ax.bar(x + off, v, 0.4, color=col, label=f"{site} ({'transits' if site == 'Malacca' else 'port calls'})")
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}" if vi >= 10 else f"{vi:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABEL[t] for t in TYPES], fontsize=9, rotation=20)
    ax.set_ylabel("Mean per day", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. Daily counts by vessel type", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    xt = np.arange(len(TYPES) + 1)
    for off, site, col in ((-0.2, "Malacca", BLUE), (0.2, "Singapore", ORANGE)):
        v = [base[(base["site"] == site) & (base["type"] == t)]["oos_R2"].iloc[0] for t in TYPES + ["total"]]
        ax.bar(xt + off, v, 0.4, color=col, label=site)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(xt)
    ax.set_xticklabels([TYPE_LABEL[t] for t in TYPES + ["total"]], fontsize=8.5, rotation=20)
    ax.set_ylabel("Out-of-sample R^2 (log scale)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. How well the baseline forecasts each type", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 2]
    for off, site, col in ((-0.27, "Malacca", BLUE), (0.0, "Singapore", ORANGE)):
        v = [100 * base[(base["site"] == site) & (base["type"] == t)]["match_rate_ex_edge"].iloc[0] for t in TYPES + ["total"]]
        ax.bar(xt + off, v, 0.27, color=col, label=f"{site}: matched")
    vc = [100 * base[(base["site"] == "Malacca") & (base["type"] == t)]["chance_rate_ex_edge"].iloc[0] for t in TYPES + ["total"]]
    vs = [100 * base[(base["site"] == "Singapore") & (base["type"] == t)]["chance_rate_ex_edge"].iloc[0] for t in TYPES + ["total"]]
    ax.bar(xt + 0.27, [(a + b) / 2 for a, b in zip(vc, vs)], 0.27, color=MUTED, label="random placement (mean of sites)")
    ax.set_xticks(xt)
    ax.set_xticklabels([TYPE_LABEL[t] for t in TYPES + ["total"]], fontsize=8.5, rotation=20)
    ax.set_ylabel("Clusters within 14 days of an event (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("C. Event matching by type (excl. first 90 days)", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=1)

    for k, site in enumerate(("Malacca", "Singapore")):
        ax = axes[1, k]
        pv = ev[ev["site"] == site].pivot(index="event", columns="type", values="change_z").reindex(names)[TYPES]
        pp_ = ev[ev["site"] == site].pivot(index="event", columns="type", values="placebo_p").reindex(names)[TYPES]
        lim = np.nanmax(np.abs(pv.values))
        im = ax.imshow(pv.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    star = "*" if pp_.values[i, j] < 0.05 else ""
                    ax.text(j, i, f"{v:+.1f}{star}", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > lim * 0.6 else INK)
        ax.set_xticks(range(len(TYPES)))
        ax.set_xticklabels([TYPE_LABEL[t] for t in TYPES], fontsize=8.5, rotation=20)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"{'D' if k == 0 else 'E'}. {site}: change in mean z (* placebo p < 0.05)", fontsize=10, color=INK, loc="left", fontweight="bold")
    ax = axes[1, 2]
    x = np.arange(len(ee))
    ax.bar(x, ee["corr_malacca_singapore"], 0.5, color=[AQUA if a else MUTED for a in ee["above_null"]])
    ax.errorbar(x, (ee["null_2.5%"] + ee["null_97.5%"]) / 2, yerr=(ee["null_97.5%"] - ee["null_2.5%"]) / 2, fmt="none", ecolor=INK, lw=1, capsize=3, label="95% null (circular shifts)")
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABEL[t] for t in ee["type"]], fontsize=8.5, rotation=20)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_ylabel("Same-day z correlation, Malacca vs Singapore", fontsize=9.5, color=SECONDARY_INK)
    ax.set_title("F. Same type at the two sites", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    for a in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_vessel_types.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_vessel_types.png")


if __name__ == "__main__":
    main()
