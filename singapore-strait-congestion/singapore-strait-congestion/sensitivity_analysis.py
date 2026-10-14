"""
Robustness check on the headline anomaly-detection result: is "64% of flagged
clusters match a documented disruption within 14 days, against 56% expected
by chance" specific to the exact threshold (|z| > 2.5) and matching window
(+/-14 days) chosen, or does the same real-vs-chance gap hold across a
neighbourhood of reasonable choices?

Reuses the z-scores already fit by baseline.py (fitting is threshold-independent,
so it's computed once); only reclusters and rematches for each (threshold,
window) pair in the grid. The chance-rate draws are vectorised with numpy
(events axis has only 6 rows, so this is a few broadcasted array comparisons
per grid cell instead of baseline.py/model_selection.py's per-draw Python
loop over matched_event, which is the right tool for one (threshold, window)
pair but too slow repeated across a 5x5 grid).

    python sensitivity_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import MIN_TRAIN_DAYS, OPEN_EVENT_WINDOW_DAYS, rolling_oos_fit

THRESH_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
PAD_GRID = [3, 7, 14, 21, 30]
N_DRAWS = 500
SEED = 0

# dataviz skill palette (references/palette.md): diverging blue<->red pair,
# neutral gray midpoint, for a signed quantity (match rate minus chance rate).
BLUE = "#2a78d6"
RED = "#e34948"
NEUTRAL = "#f0efec"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"


def cluster_days(z_series, dates, thresh):
    """Same gap<=3-day clustering as baseline.cluster_anomalies, inlined so the
    threshold can vary (baseline.cluster_anomalies always uses its own
    module-level THRESH=2.5). Returns arrays of cluster start/end as integer
    day-offsets from dates.min()."""
    flag = z_series.abs() > thresh
    day0 = dates.min()
    day_num = (dates - day0).dt.days.values
    starts, ends = [], []
    cur_start = None
    prev_day = None
    for f, d in zip(flag.values, day_num):
        if f:
            if cur_start is None or (prev_day is not None and d - prev_day > 3):
                if cur_start is not None:
                    starts.append(cur_start)
                    ends.append(prev_day)
                cur_start = d
            prev_day = d
        else:
            if cur_start is not None:
                starts.append(cur_start)
                ends.append(prev_day)
                cur_start = None
    if cur_start is not None:
        starts.append(cur_start)
        ends.append(prev_day)
    return np.array(starts), np.array(ends)


def main():
    sg = pd.read_csv("data/singapore_port_daily.csv", parse_dates=["date"]).sort_values("date")
    mal = pd.read_csv("data/malacca_strait_daily.csv", parse_dates=["date"]).sort_values("date")
    events = pd.read_csv("data/disruption_events.csv", parse_dates=["fromdate", "todate"])

    df = mal[["date", "n_total"]].merge(sg[["date"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    t0 = df["date"].min()
    first_oos = t0 + pd.Timedelta(days=MIN_TRAIN_DAYS)

    print("Fitting the rolling out-of-sample baseline once (threshold-independent)...")
    _, z, _, _ = rolling_oos_fit(df, "n_total")
    tmp = df[["date"]].copy()
    tmp["z"] = pd.Series(z)
    tmp = tmp[tmp["date"] >= first_oos].dropna(subset=["z"]).reset_index(drop=True)
    span_days = (tmp["date"].max() - first_oos).days
    print(f"Baseline fit done. {len(tmp)} out-of-sample days, span {span_days} days.")

    # events as day-offsets from first_oos (matches the day-offset frame the
    # OOS dates/clusters use below)
    ev_from = (events["fromdate"] - first_oos).dt.days.values.astype(float)
    has_end = events["todate"].notna().values
    ev_to = np.where(
        has_end,
        (events["todate"] - first_oos).dt.days.values.astype(float),
        ev_from + OPEN_EVENT_WINDOW_DAYS,
    )

    rng = np.random.default_rng(SEED)
    rows = []
    diff_grid = np.full((len(THRESH_GRID), len(PAD_GRID)), np.nan)
    n_clusters_grid = np.full((len(THRESH_GRID), len(PAD_GRID)), np.nan)

    for i, thresh in enumerate(THRESH_GRID):
        c_start, c_end = cluster_days(tmp["z"], tmp["date"], thresh)
        n_c = len(c_start)
        lengths = c_end - c_start
        for j, pad in enumerate(PAD_GRID):
            ev_lo, ev_hi = ev_from - pad, ev_to + pad
            # real match rate
            real_match = (
                (c_start[:, None] <= ev_hi[None, :]) & (c_end[:, None] >= ev_lo[None, :])
            ).any(axis=1)
            match_rate = real_match.mean() if n_c else float("nan")

            # vectorised chance rate: (n_clusters, n_draws) random start offsets
            if n_c:
                max_start = np.maximum(span_days - lengths, 1)
                draws = rng.integers(0, max_start[:, None], size=(n_c, N_DRAWS))
                draw_ends = draws + lengths[:, None]
                # (n_clusters, n_draws, n_events) -> any over events
                hit = (
                    (draws[:, :, None] <= ev_hi[None, None, :])
                    & (draw_ends[:, :, None] >= ev_lo[None, None, :])
                ).any(axis=2)
                chance = hit.mean()
            else:
                chance = float("nan")

            diff = (match_rate - chance) * 100
            diff_grid[i, j] = diff
            n_clusters_grid[i, j] = n_c
            rows.append({"thresh": thresh, "pad_days": pad, "n_clusters": int(n_c),
                         "match_rate": round(float(match_rate), 3), "chance_rate": round(float(chance), 3),
                         "diff_pp": round(float(diff), 1)})

    out = pd.DataFrame(rows)
    out.to_csv("data/sensitivity_grid.csv", index=False)
    print(f"\nGrid of {len(THRESH_GRID)}x{len(PAD_GRID)} = {len(out)} (threshold, matching-window) combinations:")
    print(out.to_string(index=False))

    baseline_row = out[(out["thresh"] == 2.5) & (out["pad_days"] == 14)]
    print(f"\nBaseline cell (thresh=2.5, pad=14), should read close to the README's 64% vs 56%:\n"
          f"{baseline_row.to_string(index=False)}")

    n_positive = (out["diff_pp"] > 0).sum()
    print(f"\n{n_positive} of {len(out)} grid cells show match rate beating chance rate "
          f"(positive diff). " +
          ("The real-vs-chance gap is a stable pattern across the grid, not an artifact of the exact "
           "(2.5, 14) choice." if n_positive == len(out)
           else "The gap is not universal across the grid -- see which cells flip."))

    # ---------------------------------------------------------------------
    # Heatmap: diverging (match rate - chance rate), neutral at 0
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    vmax = np.nanmax(np.abs(diff_grid))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("diverging", [RED, NEUTRAL, BLUE])
    im = ax.imshow(diff_grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(PAD_GRID)))
    ax.set_xticklabels([f"±{p}d" for p in PAD_GRID], fontsize=10, color=INK)
    ax.set_yticks(range(len(THRESH_GRID)))
    ax.set_yticklabels([f"|z|>{t}" for t in THRESH_GRID], fontsize=10, color=INK)
    ax.set_xlabel("Event-matching window", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Anomaly threshold", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Match rate minus chance rate (percentage points)\nacross threshold and matching-window choices",
                 fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")

    for i in range(len(THRESH_GRID)):
        for j in range(len(PAD_GRID)):
            val = diff_grid[i, j]
            n_c = int(n_clusters_grid[i, j])
            text_color = INK if abs(val) < vmax * 0.6 else SURFACE
            ax.text(j, i, f"{val:+.0f}pp\n(n={n_c})", ha="center", va="center",
                    fontsize=8.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Match rate − chance rate (pp)", fontsize=9, color=SECONDARY_INK)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.text(0.01, -0.02,
              "Each cell: percentage-point gap between the real match rate and the random-placement chance rate "
              "at that (threshold, window) combination. Positive (blue) = real pattern beats chance.",
              fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("data/chart_sensitivity_heatmap.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_sensitivity_heatmap.png")


if __name__ == "__main__":
    main()
