"""
Part 6: a false-positive audit. Of the 129 flagged Malacca clusters, 83 sit within 14 days of a documented event and 46
do not. What do the 46 line up with?

Each unmatched cluster (the table in data/anomaly_clusters.csv, as written by analysis.py) is checked against:
  1. The baseline edge window: the first 90 out-of-sample days, where Part 2 showed the regression baseline is an artifact.
  2. Public holidays and festivals in Singapore, Malaysia and Indonesia (the `holidays` package): any holiday from a day
     before the cluster to a day after it. Holidays are frequent, so the question is whether unmatched clusters fall
     near them more often than randomly placed clusters of the same lengths do.
  3. The rest of PortWatch's own disruption layer, which fetch_data.py filters out: tropical cyclones, floods,
     earthquakes, volcanoes, droughts and wildfires, kept if the epicentre is within 2,500 km of Singapore and the event
     (plus 7 days either side) overlaps the cluster. Again compared with random placement.
  4. Data and measurement checks: missing days, zero or repeated values (none exist in this series), whether Singapore's
     port calls are also flagged within 3 days (Malacca-only anomalies are more likely to be counting artifacts), whether
     the anomaly is broad across vessel types at the peak day (3 or more of 5 types beyond |z| 2 with the same sign) or
     confined to one, and whether it is a single day.
The same checks are run on the 83 matched clusters, on a "typical" random placement, and an unexplained residual is
reported: unmatched clusters that no check accounts for.

    python false_positive_audit.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays as hol

from baseline import MIN_TRAIN_DAYS, THRESH, rolling_oos_fit, cluster_anomalies, matched_event
from fetch_data import BASE, paginate

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

SINGAPORE = (1.29, 103.85)
RADIUS_KM = 2500
EVENT_PAD = 7
EDGE_DAYS = 90
TYPES = ["container", "dry_bulk", "general_cargo", "roro", "tanker"]
N_RANDOM = 400
SEED = 0

FESTIVALS = [
    ("Lunar New Year", ("chinese new year", "lunar new year", "imlek")),
    ("Hari Raya Aidilfitri", ("hari raya puasa", "aidilfitri", "eid al-fitr", "idul fitri", "hari raya")),
    ("Hari Raya Haji", ("hari raya haji", "aidiladha", "eid al-adha", "idul adha")),
    ("Deepavali", ("deepavali", "diwali")),
    ("Vesak", ("vesak", "wesak", "waisak")),
    ("Christmas", ("christmas",)),
    ("New Year's Day", ("new year's day", "new year")),
]


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def festival_of(name):
    n = name.lower()
    for label, keys in FESTIVALS:
        if any(k in n for k in keys):
            return label
    return "other public holiday"


def main():
    rng = np.random.default_rng(SEED)
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])
    clusters = pd.read_csv("data/anomaly_clusters.csv", parse_dates=["start", "end"])
    df = mal.merge(sg[["date"] + [c for c in sg.columns if c.startswith("portcalls")]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    first_oos = df["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    last_day = df["date"].max()
    edge_end = first_oos + pd.Timedelta(days=EDGE_DAYS)
    span_days = (last_day - first_oos).days

    # --------------------------------------------------------------------------- data checks (no gaps, zeros)
    days = df["date"]
    full = pd.date_range(days.min(), days.max(), freq="D")
    n_missing = len(full) - len(days)
    repeats = int((df["n_total"].diff() == 0).sum())
    print(f"Data checks: {len(days)} days, {n_missing} missing dates, {int((df['n_total'] <= 0).sum())} zero-or-negative totals, "
          f"{repeats} days equal to the previous day's total, last day {days.max().date()}.")
    # does the saved table match the pipeline?
    _, z_m, _, _ = rolling_oos_fit(df, "n_total")
    tmp = pd.DataFrame({"date": df["date"], "z": z_m})
    tmp = tmp[tmp["date"] >= first_oos].dropna(subset=["z"]).reset_index(drop=True)
    _, cl_check = cluster_anomalies(tmp, "z")
    assert len(cl_check) == len(clusters) and (cl_check["start"].values == clusters["start"].values).all(), \
        "data/anomaly_clusters.csv no longer matches the pipeline; run analysis.py first"
    zs = pd.Series(z_m, index=df["date"])

    clusters["matched"] = clusters["matched_event"].notna()
    clusters["edge"] = clusters["start"] < edge_end
    clusters["sign"] = [np.sign(zs[(zs.index >= r.start) & (zs.index <= r.end)].iloc[np.argmax(np.abs(zs[(zs.index >= r.start) & (zs.index <= r.end)].values))]) for r in clusters.itertuples()]
    print(f"{len(clusters)} clusters: {int(clusters['matched'].sum())} matched, {int((~clusters['matched']).sum())} unmatched; "
          f"{int(clusters['edge'].sum())} start inside the first {EDGE_DAYS} days ({int((clusters['edge'] & ~clusters['matched']).sum())} of them unmatched).")

    # ------------------------------------------------------------------------------------------- holidays
    years = range(2019, 2027)
    hol_days = {}
    for ctry in ("SG", "MY", "ID"):
        for d, name in hol.country_holidays(ctry, years=years).items():
            hol_days.setdefault(pd.Timestamp(d), set()).add((ctry, festival_of(name)))
    hol_index = pd.DatetimeIndex(sorted(hol_days))
    print(f"Holidays 2019-2026 in SG, MY and ID: {len(hol_index)} distinct dates ({len(hol_index) / (8 * 365) * 100:.0f}% of days).")

    def holiday_hit(start, end):
        lo, hi = start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)
        hits = hol_index[(hol_index >= lo) & (hol_index <= hi)]
        return sorted({f for d in hits for (_, f) in hol_days[d]})

    # -------------------------------------------------------------------- other PortWatch disruption events
    url = f"{BASE}/portwatch_disruptions_database/FeatureServer/0/query"
    ev_all = pd.DataFrame(paginate(url, where="1=1", extra_params={"outFields": "eventid,eventtype,eventname,alertlevel,country,fromdate,todate,lat,long", "returnGeometry": "false"}))
    ev_all["fromdate"] = pd.to_datetime(ev_all["fromdate"], unit="ms")
    ev_all["todate"] = pd.to_datetime(ev_all["todate"], unit="ms")
    other = ev_all[ev_all["eventtype"] != "OT"].copy()
    other["km"] = haversine(SINGAPORE[0], SINGAPORE[1], other["lat"].astype(float), other["long"].astype(float))
    near = other[other["km"] <= RADIUS_KM].copy()
    near["end"] = near["todate"].fillna(near["fromdate"] + pd.Timedelta(days=3))
    print(f"PortWatch events other than the 6 used so far: {len(other)}; {len(near)} within {RADIUS_KM} km of Singapore "
          f"({near['eventtype'].value_counts().to_dict()}).")
    near.to_csv("data/audit_nearby_events.csv", index=False)

    def event_hit(start, end):
        m = near[(near["fromdate"] - pd.Timedelta(days=EVENT_PAD) <= end) & (near["end"] + pd.Timedelta(days=EVENT_PAD) >= start)]
        return [f"{r.eventname} ({r.eventtype})" for r in m.itertuples()]

    # -------------------------------------------------------------- Singapore corroboration and type breadth
    _, z_s, _, _ = rolling_oos_fit(df, "portcalls")
    zsg = pd.Series(z_s, index=df["date"])
    sg_flag = (zsg.abs() > THRESH)

    def sg_hit(start, end, pad=3):
        w = sg_flag[(sg_flag.index >= start - pd.Timedelta(days=pad)) & (sg_flag.index <= end + pd.Timedelta(days=pad))]
        return bool(w.any())

    ztype = {}
    for t in TYPES:
        work = df[["date"]].copy()
        work["y"] = df[f"n_{t}"] + 1
        ztype[t] = pd.Series(rolling_oos_fit(work, "y")[1], index=df["date"])

    def breadth(start, end, sign):
        seg = zs[(zs.index >= start) & (zs.index <= end)]
        day = seg.index[int(np.argmax(np.abs(seg.values)))]
        return int(sum((np.sign(ztype[t].get(day, np.nan)) == sign) and (abs(ztype[t].get(day, np.nan)) > 2) for t in TYPES))

    rows = []
    for r in clusters.itertuples():
        hh = holiday_hit(r.start, r.end)
        ee = event_hit(r.start, r.end)
        rows.append({"start": r.start.date(), "end": r.end.date(), "n_days": r.n_days, "max_abs_z": r.max_abs_z, "sign": int(r.sign),
                     "matched": r.matched, "matched_event": r.matched_event, "edge_window": r.edge, "single_day": r.n_days == 1,
                     "holiday": ";".join(hh), "other_event": ";".join(ee), "singapore_also_flagged": sg_hit(r.start, r.end),
                     "types_beyond_2": breadth(r.start, r.end, r.sign)})
    audit = pd.DataFrame(rows)
    audit["holiday_hit"] = audit["holiday"] != ""
    audit["other_event_hit"] = audit["other_event"] != ""
    audit["broad_across_types"] = audit["types_beyond_2"] >= 3
    explained = audit["edge_window"] | audit["holiday_hit"] | audit["other_event_hit"]
    audit["explained_by_a_check"] = explained
    audit.to_csv("data/audit_clusters.csv", index=False)

    # ------------------------------------------------------------------ random placement null (same lengths)
    def random_rates(lengths, exclude_edge=True):
        lo = (edge_end if exclude_edge else first_oos)
        span = (last_day - lo).days
        n_h = n_e = n_sg = 0
        total = 0
        for _ in range(N_RANDOM):
            for L in lengths:
                st = lo + pd.Timedelta(days=int(rng.integers(0, max(span - (L - 1), 1))))
                en = st + pd.Timedelta(days=L - 1)
                n_h += bool(holiday_hit(st, en))
                n_e += bool(event_hit(st, en))
                n_sg += sg_hit(st, en)
                total += 1
        return n_h / total, n_e / total, n_sg / total

    unm = audit[~audit["matched"]]
    mat = audit[audit["matched"]]
    unm_core = unm[~unm["edge_window"]]
    rh, re_, rs = random_rates(list(unm_core["n_days"]))
    rows_s = []

    def summarize(label, g, base=None):
        n = len(g)
        row = {"group": label, "clusters": n, "single_day_%": 100 * g["single_day"].mean(), "median_max_abs_z": g["max_abs_z"].median(),
               "negative_z_%": 100 * (g["sign"] < 0).mean(), "edge_window_%": 100 * g["edge_window"].mean(),
               "holiday_adjacent_%": 100 * g["holiday_hit"].mean(), "other_PortWatch_event_%": 100 * g["other_event_hit"].mean(),
               "singapore_also_flagged_%": 100 * g["singapore_also_flagged"].mean(), "broad_across_types_%": 100 * g["broad_across_types"].mean()}
        if base is not None:
            row.update({"random_holiday_%": 100 * base[0], "random_other_event_%": 100 * base[1], "random_singapore_%": 100 * base[2]})
        return row

    rows_s.append(summarize("Unmatched, outside the edge window", unm_core, (rh, re_, rs)))
    rows_s.append(summarize("Unmatched, inside the edge window", unm[unm["edge_window"]]))
    mh, me, ms = random_rates(list(mat[~mat["edge_window"]]["n_days"]))
    rows_s.append(summarize("Matched, outside the edge window", mat[~mat["edge_window"]], (mh, me, ms)))
    summ = pd.DataFrame(rows_s)
    summ.to_csv("data/audit_summary.csv", index=False)
    print("\n=== Audit summary ===")
    print(summ.round(1).to_string(index=False))

    # festivals among unmatched, compared with random
    fest_rows = []
    fest_counts = {}
    for r in unm_core.itertuples():
        for f in (r.holiday.split(";") if r.holiday else []):
            fest_counts[f] = fest_counts.get(f, 0) + 1
    rand_fest = {}
    lengths = list(unm_core["n_days"])
    lo, span = edge_end, (last_day - edge_end).days
    for _ in range(N_RANDOM):
        for L in lengths:
            st = lo + pd.Timedelta(days=int(rng.integers(0, max(span - (L - 1), 1))))
            for f in holiday_hit(st, st + pd.Timedelta(days=L - 1)):
                rand_fest[f] = rand_fest.get(f, 0) + 1
    print("\nFestivals near the %d unmatched clusters (outside the edge window), against the expected count under random placement:" % len(unm_core))
    for f in sorted(set(fest_counts) | set(rand_fest), key=lambda k: -fest_counts.get(k, 0)):
        exp = rand_fest.get(f, 0) / N_RANDOM
        fest_rows.append({"festival": f, "unmatched_clusters_near": fest_counts.get(f, 0), "expected_under_random": exp})
        print(f"  {f:<24} {fest_counts.get(f, 0):2d} observed, {exp:4.1f} expected")
    pd.DataFrame(fest_rows).to_csv("data/audit_festivals.csv", index=False)

    # residual
    resid = unm_core[~(unm_core["holiday_hit"] | unm_core["other_event_hit"])]
    print(f"\nUnmatched clusters outside the edge window: {len(unm_core)}; holiday-adjacent {int(unm_core['holiday_hit'].sum())}, "
          f"near another PortWatch event {int(unm_core['other_event_hit'].sum())}, either {int((unm_core['holiday_hit'] | unm_core['other_event_hit']).sum())}; "
          f"neither {len(resid)} ({len(resid) / len(unm_core) * 100:.0f}%)")
    print(f"  of the {len(resid)} with neither: Singapore also flagged {int(resid['singapore_also_flagged'].sum())}, broad across types {int(resid['broad_across_types'].sum())}, "
          f"single-day {int(resid['single_day'].sum())}, negative direction {int((resid['sign'] < 0).sum())}")
    print("\nNearby non-OT PortWatch events that overlap an unmatched cluster:")
    ev_hits = unm_core[unm_core["other_event_hit"]][["start", "end", "n_days", "other_event"]]
    print(ev_hits.to_string(index=False))

    # ----------------------------------------------------- how much of "unmatched" is just a silent calendar?
    days_ex = pd.date_range(edge_end, last_day, freq="D")
    covered = pd.Series([matched_event(events, d, d) is not None for d in days_ex], index=days_ex)
    core = audit[~audit["edge_window"]].copy()
    core["start_ts"] = pd.to_datetime(core["start"])
    starts_cov = core["start_ts"].map(covered).astype(bool)
    n_cov, n_unc = int(covered.sum()), int((~covered).sum())
    rate_cov = int(starts_cov.sum()) / n_cov * 100
    rate_unc = int((~starts_cov).sum()) / n_unc * 100
    obs = int(starts_cov.sum())
    idx = np.array([days_ex.get_loc(t) for t in core["start_ts"]])
    cov_arr = covered.values
    null = []
    for _ in range(5000):
        sh = np.roll(cov_arr, int(rng.integers(30, len(cov_arr) - 30)))
        null.append(int(sh[idx].sum()))
    null = np.array(null)
    p_cov = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    print(f"\nCalendar coverage: {n_cov} of {len(days_ex)} out-of-sample days ({n_cov / len(days_ex) * 100:.0f}%) are within the 14-day matching "
          f"window of a documented event. Cluster starts: {obs} on covered days ({rate_cov:.1f} per 100 days), {len(core) - obs} on uncovered days "
          f"({rate_unc:.1f} per 100 days); with the covered-day mask shifted at random, {np.median(null):.0f} would be expected on covered days "
          f"(95th percentile {np.percentile(null, 95):.0f}); p = {p_cov:.3f}.")
    cov_year = covered.groupby(covered.index.year).mean()
    cl_year = core.groupby([core["start_ts"].dt.year, "matched"]).size().unstack(fill_value=0).reindex(cov_year.index).fillna(0)
    cov_tab = pd.DataFrame({"year": cov_year.index, "days_covered_%": 100 * cov_year.values,
                            "matched_clusters": cl_year.get(True, 0).values, "unmatched_clusters": cl_year.get(False, 0).values})
    cov_tab.to_csv("data/audit_calendar_coverage.csv", index=False)
    print(cov_tab.round(0).to_string(index=False))
    pd.DataFrame([{"covered_days": n_cov, "uncovered_days": n_unc, "clusters_on_covered_days": obs,
                   "clusters_on_uncovered_days": len(core) - obs, "rate_covered_per_100d": rate_cov, "rate_uncovered_per_100d": rate_unc,
                   "shifted_mask_median": float(np.median(null)), "shifted_mask_95pct": float(np.percentile(null, 95)), "p_value": p_cov}]
                 ).to_csv("data/audit_coverage_test.csv", index=False)

    # -------------------------------------------------------------------- when do unmatched clusters happen
    unm_core = unm_core.copy()
    unm_core["month"] = pd.to_datetime(unm_core["start"]).dt.month
    unm_core["year"] = pd.to_datetime(unm_core["start"]).dt.year
    print("\nUnmatched (outside the edge window) by year:", unm_core["year"].value_counts().sort_index().to_dict())
    print("Matched by year:", pd.to_datetime(mat[~mat["edge_window"]]["start"]).dt.year.value_counts().sort_index().to_dict())
    print("Unmatched by month:", unm_core["month"].value_counts().sort_index().to_dict())
    print("Unmatched by day of week of the peak:", pd.to_datetime(unm_core["start"]).dt.dayofweek.value_counts().sort_index().to_dict())

    # ----------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4), facecolor=SURFACE)
    ax = axes[0, 0]
    cats = [("edge window", int(unm["edge_window"].sum())),
            ("holiday-adjacent", int(unm_core["holiday_hit"].sum())),
            ("near other PortWatch event", int(unm_core["other_event_hit"].sum())),
            ("holiday or event", int((unm_core["holiday_hit"] | unm_core["other_event_hit"]).sum())),
            ("neither", len(resid))]
    ax.barh(range(len(cats)), [c[1] for c in cats], color=[MUTED, BLUE, ORANGE, AQUA, RED])
    for i, (k, v) in enumerate(cats):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c[0] for c in cats], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Unmatched clusters (%d outside the edge window, %d inside)" % (len(unm_core), int(unm["edge_window"].sum())), fontsize=9.5, color=SECONDARY_INK)
    ax.set_title("A. What the 46 unmatched clusters line up with", fontsize=11, color=INK, loc="left", fontweight="bold")

    ax = axes[0, 1]
    labs = ["holiday-adjacent", "near other event", "Singapore also flagged"]
    keys = ["holiday_adjacent_%", "other_PortWatch_event_%", "singapore_also_flagged_%"]
    rk = ["random_holiday_%", "random_other_event_%", "random_singapore_%"]
    x = np.arange(3)
    ax.bar(x - 0.27, [summ.iloc[0][k] for k in keys], 0.27, color=RED, label="unmatched")
    ax.bar(x, [summ.iloc[2][k] for k in keys], 0.27, color=BLUE, label="matched")
    ax.bar(x + 0.27, [summ.iloc[0][k] for k in rk], 0.27, color=MUTED, label="random placement")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Share of clusters (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. Alignment rates against random placement", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    fd = pd.DataFrame(fest_rows).sort_values("unmatched_clusters_near", ascending=False)
    y = np.arange(len(fd))
    ax.barh(y - 0.2, fd["unmatched_clusters_near"], 0.4, color=RED, label="unmatched clusters")
    ax.barh(y + 0.2, fd["expected_under_random"], 0.4, color=MUTED, label="expected under random placement")
    ax.set_yticks(y)
    ax.set_yticklabels(fd["festival"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Clusters adjacent to the festival", fontsize=9.5, color=SECONDARY_INK)
    ax.set_title("C. Festivals near unmatched clusters", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1, 1]
    yrs = range(2020, 2027)
    um = pd.to_datetime(unm_core["start"]).dt.year.value_counts().reindex(yrs).fillna(0)
    ma = pd.to_datetime(mat[~mat["edge_window"]]["start"]).dt.year.value_counts().reindex(yrs).fillna(0)
    x = np.arange(len(list(yrs)))
    ax.bar(x - 0.2, ma.values, 0.4, color=BLUE, label="matched")
    ax.bar(x + 0.2, um.values, 0.4, color=RED, label="unmatched")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y_}{chr(10)}{cov_year.get(y_, 0) * 100:.0f}%" for y_ in yrs], fontsize=8.5)
    ax.set_ylabel("Clusters (excluding the edge window)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Clusters by year (% of days inside an event window)", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_false_positive_audit.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_false_positive_audit.png")


if __name__ == "__main__":
    main()
