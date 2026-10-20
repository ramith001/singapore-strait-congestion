"""
Part 7: a live PortWatch refresh, and what it changes.

The analysis in this repository was run on data fetched up to early September 2026 (Malacca to 2026-09-06, Singapore to
2026-09-04). This script pulls the current series and event calendar from the PortWatch API into data/refresh/ without
touching the committed files, so every number in the README stays tied to the data it was computed on, and then:

  1. Checks what the API now returns for the dates already held: new days, and whether any earlier value was revised.
  2. Re-runs the headline pipeline on both vintages (the committed data, as a check that it reproduces the README, and the
     refreshed data): flagged clusters and event match rate against random placement, out-of-sample R^2, the Malacca-Singapore
     zero-lag correlation and lead asymmetry, and the Part 2 event study.
  3. Compares the flagged days on the dates both vintages cover (revisions alone can change them), and lists the clusters
     flagged in the new days.

The refit schedule is anchored to the first date (every 90 days from day 365), so the same data give the same z-scores in
both vintages; differences come from revised values, new days, and the new refit if one falls in the added period.

    python refresh_check.py            # needs network access to the PortWatch ArcGIS service
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import MIN_TRAIN_DAYS, THRESH, rolling_oos_fit, cluster_anomalies, matched_event
from event_study import event_paths, analyse, HALF
from fetch_data import fetch_chokepoint, fetch_port, fetch_disruption_events

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

STAGE = "data/refresh"
SEED = 0
N_CHANCE = 100
VALUE_COLS_M = ["n_container", "n_dry_bulk", "n_general_cargo", "n_roro", "n_tanker", "n_cargo", "n_total"]
VALUE_COLS_S = ["portcalls_container", "portcalls_dry_bulk", "portcalls_general_cargo", "portcalls_roro", "portcalls_tanker",
                "portcalls_cargo", "portcalls", "import", "export"]


def headline(mal, sg, events, label):
    """sg is None when the Singapore series is not available (the Malacca-only pipeline)."""
    rng = np.random.default_rng(SEED)
    if sg is None:
        df = mal[["date", "n_total"]].sort_values("date").reset_index(drop=True)
    else:
        df = mal[["date", "n_total"]].merge(sg[["date", "portcalls"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    first_oos = df["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    last_day = df["date"].max()
    span_days = (last_day - first_oos).days
    fm, zm, _, oos_m = rolling_oos_fit(df, "n_total")
    out = {"vintage": label, "last_date": last_day.date(), "days": len(df), "oos_R2_malacca_total": oos_m}
    zsM = pd.Series(zm, index=df["date"])
    zsS = None
    series_list = [("malacca", zsM, "m")]
    if sg is not None:
        fs, zs, _, oos_s = rolling_oos_fit(df, "portcalls")
        out["oos_R2_singapore_calls"] = oos_s
        zsS = pd.Series(zs, index=df["date"])
        series_list.append(("singapore", zsS, "s"))
    for series, z, tag in series_list:
        tmp = pd.DataFrame({"date": z.index, "z": z.values})
        tmp = tmp[tmp["date"] >= first_oos].dropna(subset=["z"]).reset_index(drop=True)
        _, cl = cluster_anomalies(tmp, "z")
        n_m = int(sum(matched_event(events, r.start, r.end) is not None for r in cl.itertuples()))
        hits = total = 0
        for _ in range(N_CHANCE):
            for r in cl.itertuples():
                L = (r.end - r.start).days
                st = first_oos + pd.Timedelta(days=int(rng.integers(0, max(span_days - L, 1))))
                hits += matched_event(events, st, st + pd.Timedelta(days=L)) is not None
                total += 1
        out.update({f"{series}_flagged_days": int((tmp["z"].abs() > THRESH).sum()), f"{series}_clusters": len(cl),
                    f"{series}_matched": n_m, f"{series}_match_rate": n_m / len(cl), f"{series}_chance_rate": hits / total})
        if tag == "m":
            clusters_m = cl
    # cross-correlation, excluding the first 90 days as in Part 3 (only when both series exist)
    edge_end = first_oos + pd.Timedelta(days=90)
    if zsS is not None:
        both = pd.concat([zsM, zsS], axis=1, keys=["m", "s"]).dropna()
        both = both[both.index >= edge_end]
        out["zero_lag_corr_ex_edge"] = float(both["m"].corr(both["s"]))
        zS_full = zsS.reindex(zsM.index)

        def r_at(k):
            d = pd.concat([zsM, zS_full.shift(-k)], axis=1).dropna()
            d = d[d.index >= edge_end]
            return float(d.iloc[:, 0].corr(d.iloc[:, 1]))
        lead = np.mean([r_at(k) for k in range(1, 8)])
        lag = np.mean([r_at(k) for k in range(-7, 0)])
        out["lead_asymmetry_A"] = float(lead - lag)
    # event study (Malacca total z, all six and without COVID)
    zz = zsM[zsM.index >= first_oos].dropna()
    zfull = zz.reindex(pd.date_range(zz.index.min(), zz.index.max(), freq="D"))
    ev = events.sort_values("fromdate")
    starts = list(ev["fromdate"])
    paths = event_paths(zfull, starts)
    eligible = zfull.index[(zfull.index >= zfull.index.min() + pd.Timedelta(days=HALF)) & (zfull.index <= zfull.index.max() - pd.Timedelta(days=HALF))]
    eligible = eligible[zfull.reindex(eligible).notna()]
    r6 = analyse(paths, eligible, zfull, np.random.default_rng(SEED), "All six events")
    r5 = analyse(paths[1:], eligible, zfull, np.random.default_rng(SEED), "Without COVID")
    out.update({"event_study_signed_z_change": r6["d_s"], "event_study_signed_p": r6["p_s"],
                "event_study_exCOVID_signed_z_change": r5["d_s"], "event_study_exCOVID_signed_p": r5["p_s"],
                "event_study_abs_z_change": r6["d_a"], "event_study_abs_p": r6["p_a"]})
    return out, zsM, zsS, clusters_m, df


def main():
    os.makedirs(STAGE, exist_ok=True)
    print("Fetching the current series and event calendar from PortWatch...")
    mal_new = fetch_chokepoint("chokepoint5")
    try:
        sg_new = fetch_port("port1201")
    except KeyError:      # the API returns no rows at all for this port (see the README)
        sg_new = pd.DataFrame()
    ev_new = fetch_disruption_events()
    for name, d in (("malacca_strait_daily.csv", mal_new), ("singapore_port_daily.csv", sg_new), ("disruption_events.csv", ev_new)):
        if len(d):
            d.to_csv(f"{STAGE}/{name}", index=False)
    mal_new["date"] = pd.to_datetime(mal_new["date"])
    sg_available = len(sg_new) > 0
    if sg_available:
        sg_new["date"] = pd.to_datetime(sg_new["date"])
    mal_old = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    sg_old = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    ev_old = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])
    ev_new["fromdate"] = pd.to_datetime(ev_new["fromdate"])
    ev_new["todate"] = pd.to_datetime(ev_new["todate"])
    print(f"Malacca: committed {len(mal_old)} days to {mal_old['date'].max().date()}; now {len(mal_new)} days to {mal_new['date'].max().date()}")
    if sg_available:
        print(f"Singapore: committed {len(sg_old)} days to {sg_old['date'].max().date()}; now {len(sg_new)} days to {sg_new['date'].max().date()}")
    else:
        print(f"Singapore: committed {len(sg_old)} days to {sg_old['date'].max().date()}; the API now returns 0 rows for port1201")

    # ------------------------------------------------------------------------------------ 1. revisions
    rev_rows = []
    pairs = [("Malacca", mal_old, mal_new, VALUE_COLS_M)] + ([("Singapore", sg_old, sg_new, VALUE_COLS_S)] if sg_available else [])
    for site, old, new, cols in pairs:
        m = old.merge(new, on="date", suffixes=("_old", "_new"))
        for c in cols:
            a, b = m[f"{c}_old"].astype(float), m[f"{c}_new"].astype(float)
            diff = (b - a)
            nz = diff != 0
            rev_rows.append({"site": site, "column": c, "overlap_days": len(m), "revised_days": int(nz.sum()),
                             "revised_share_%": 100 * nz.mean(), "max_abs_change": float(diff.abs().max()),
                             "median_abs_change_when_revised": float(diff[nz].abs().median()) if nz.any() else 0.0,
                             "median_rel_change_%_when_revised": float((diff[nz].abs() / a[nz].abs().replace(0, np.nan)).median() * 100) if nz.any() else 0.0,
                             "first_revised_date": m.loc[nz, "date"].min().date() if nz.any() else None,
                             "last_revised_date": m.loc[nz, "date"].max().date() if nz.any() else None})
        missing = set(old["date"]) - set(new["date"])
        print(f"  {site}: {len(missing)} committed dates absent from the new pull; {len(set(new['date']) - set(old['date']))} new dates")
    rev = pd.DataFrame(rev_rows)
    rev.to_csv(f"{STAGE}/revisions.csv", index=False)
    print("\n=== 1. Revisions to dates already held ===")
    print(rev[["site", "column", "overlap_days", "revised_days", "revised_share_%", "max_abs_change", "median_abs_change_when_revised"]].round(2).to_string(index=False))

    ev_names_old = set(ev_old["eventname"])
    ev_names_new = set(ev_new["eventname"])
    print(f"\nEvent calendar: {len(ev_old)} events held, {len(ev_new)} now. New: {sorted(ev_names_new - ev_names_old)}; dropped: {sorted(ev_names_old - ev_names_new)}")
    chg = ev_old.merge(ev_new, on="eventname", suffixes=("_old", "_new"))
    for r in chg.itertuples():
        if (r.fromdate_old != r.fromdate_new) or (pd.notna(r.todate_old) != pd.notna(r.todate_new)) or (pd.notna(r.todate_old) and r.todate_old != r.todate_new):
            print(f"  changed: {r.eventname}: {r.fromdate_old.date()} to {r.todate_old} -> {r.fromdate_new.date()} to {r.todate_new}")

    # -------------------------------------------------------------------------------------- 2. headline
    print("\n=== 2. Headline numbers on both vintages ===")
    res_old, zM_old, zS_old, cl_old, df_old = headline(mal_old, sg_old, ev_old, "committed (Malacca and Singapore dates, as in the README)")
    res_mid, zM_mid, _, cl_mid, _ = headline(mal_old, None, ev_old, "committed, Malacca dates only")
    res_new, zM_new, _, cl_new, df_new = headline(mal_new, sg_new if sg_available else None, ev_new, "refreshed" + ("" if sg_available else ", Malacca only"))
    summ = pd.DataFrame([res_old, res_mid, res_new])
    summ.to_csv(f"{STAGE}/headline_comparison.csv", index=False)
    print(summ.T.to_string(header=False))

    # --------------------------------------------------------------- 3. flag stability and the new days
    z_both = pd.concat([zM_mid, zM_new], axis=1, keys=["old", "new"]).dropna()
    f_old, f_new = z_both["old"].abs() > THRESH, z_both["new"].abs() > THRESH
    both = int((f_old & f_new).sum())
    print(f"\n=== 3. Flagged days on the {len(z_both)} dates both vintages score ===")
    print(f"  flagged in committed {int(f_old.sum())}, in refreshed {int(f_new.sum())}, in both {both}; "
          f"only committed {int((f_old & ~f_new).sum())}, only refreshed {int((~f_old & f_new).sum())}; "
          f"max |z difference| {float((z_both['new'] - z_both['old']).abs().max()):.3f}")
    old_last = mal_old["date"].max()
    new_days = zM_new[zM_new.index > old_last].dropna()
    print(f"\nNew days {old_last.date() + pd.Timedelta(days=1)} to {zM_new.index.max().date()}: {len(new_days)} scored; "
          f"{int((new_days.abs() > THRESH).sum())} flagged")
    events_new = ev_new
    tmp = pd.DataFrame({"date": zM_new.index, "z": zM_new.values}).dropna()
    _, cl_all = cluster_anomalies(tmp.reset_index(drop=True), "z")
    cl_recent = cl_all[cl_all["end"] > old_last - pd.Timedelta(days=10)].copy()
    cl_recent["matched_event"] = [matched_event(events_new, r.start, r.end) for r in cl_recent.itertuples()]
    print(cl_recent.to_string(index=False))
    cl_recent.to_csv(f"{STAGE}/recent_clusters.csv", index=False)
    pd.DataFrame({"date": z_both.index, "z_committed": z_both["old"], "z_refreshed": z_both["new"]}).to_csv(f"{STAGE}/z_overlap.csv", index=False)
    new_days.rename("z").to_frame().to_csv(f"{STAGE}/z_new_days.csv")

    # ------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=SURFACE)
    ax = axes[0]
    tail = mal_new[mal_new["date"] >= mal_new["date"].max() - pd.Timedelta(days=150)]
    ax.plot(tail["date"], tail["n_total"], color=BLUE, lw=1.3)
    ax.axvspan(old_last, tail["date"].max(), color=NEUTRAL, zorder=0, label="added since the committed pull")
    ax.set_ylabel("Malacca vessel transits per day", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. The last 150 days", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[1]
    zr = zM_new[zM_new.index >= zM_new.index.max() - pd.Timedelta(days=150)]
    ax.plot(zr.index, zr.values, color=ORANGE, lw=1.2)
    ax.axhline(THRESH, color=INK, lw=0.8, ls="--")
    ax.axhline(-THRESH, color=INK, lw=0.8, ls="--")
    ax.axvspan(old_last, zr.index.max(), color=NEUTRAL, zorder=0)
    ax.set_ylabel("Out-of-sample z-score (refreshed data)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. z-score, last 150 days", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[2]
    labels = ["Malacca\nmatch rate", "Malacca\nchance rate", "Singapore\nmatch rate", "Singapore\nchance rate"]
    keys = ["malacca_match_rate", "malacca_chance_rate", "singapore_match_rate", "singapore_chance_rate"]
    x = np.arange(4)
    ax.bar(x - 0.2, [100 * res_old[k] for k in keys], 0.4, color=BLUE, label="committed data")
    ax.bar(x + 0.2, [100 * res_new.get(k, np.nan) for k in keys], 0.4, color=ORANGE, label="refreshed data")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Clusters within 14 days of an event (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylim(0, 90)
    ax.set_title("C. Match rate (no Singapore refresh available)" if not sg_available else "C. Match rate, both vintages", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_refresh.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_refresh.png")


if __name__ == "__main__":
    main()
