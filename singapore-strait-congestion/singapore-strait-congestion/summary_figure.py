"""
Part 7: one consolidated figure for the repo. Reads the CSVs the earlier parts saved and runs no analysis.

    python summary_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def main():
    grid = pd.read_csv("data/sensitivity_grid.csv")
    paths = pd.read_csv("data/event_study_paths.csv")
    xc = pd.read_csv("data/lead_lag_xcorr.csv")
    vt = pd.read_csv("data/vessel_type_baseline.csv")
    sd = pd.read_csv("data/second_detector_summary.csv")
    cov = pd.read_csv("data/audit_calendar_coverage.csv")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.8), facecolor=SURFACE)

    ax = axes[0, 0]
    piv = grid.pivot(index="thresh", columns="pad_days", values="diff_pp")
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"+{piv.values[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if piv.values[i, j] > piv.values.max() * 0.6 else INK)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c}d" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Matching window (+/- days)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("|z| threshold", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 1: match rate minus chance rate (points)", fontsize=11, color=INK, loc="left", fontweight="bold")

    ax = axes[0, 1]
    d = paths["day_offset"]
    ax.fill_between(d, paths["placebo_lo_z"], paths["placebo_hi_z"], color=NEUTRAL, label="placebo: 6 random dates")
    ax.fill_between(d, paths["ci_lo_z"], paths["ci_hi_z"], color=BLUE, alpha=0.25, lw=0, label="95% CI across the 6 events")
    ax.plot(d, paths["mean_z"], color=BLUE, lw=2, label="mean z, six events")
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("Days from event start", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Malacca z-score", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 2: event study, inside the placebo band", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[0, 2]
    s = xc[(xc["pair"] == "total") & (xc["variant"] == "excluding first 90 days")]
    ax.fill_between(s["lag_days"], s["null_lo"], s["null_hi"], color=NEUTRAL, label="95% null (circular shifts)")
    ax.plot(s["lag_days"], s["r"], color=ORANGE, lw=2, label="z-score correlation")
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("Lag (days; > 0: Malacca leads)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Correlation, Malacca vs Singapore", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 3: contemporaneous, no lead either way", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1, 0]
    types = ["container", "dry_bulk", "general_cargo", "roro", "tanker", "total"]
    lab = ["Container", "Dry bulk", "Gen. cargo", "Roro", "Tanker", "Total"]
    x = np.arange(len(types))
    for off, site, col in ((-0.2, "Malacca", BLUE), (0.2, "Singapore", ORANGE)):
        r = vt[vt["site"] == site].set_index("type").loc[types]
        ax.bar(x + off, 100 * (r["match_rate_ex_edge"] - r["chance_rate_ex_edge"]), 0.4, color=col, label=site)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=20, fontsize=8.5)
    ax.set_ylabel("Match rate minus chance rate (points)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 4: by vessel type", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    dets = ["A: trend + weekday + month, refit quarterly", "B1: weekly median", "B2: seasonal decomposition"]
    labs = ["A\nregression", "B1\nweekly median", "B2\ndecomposition"]
    x = np.arange(3)
    for off, series, col in ((-0.2, "Malacca total", BLUE), (0.2, "Singapore port calls", ORANGE)):
        r = sd[sd["series"] == series].set_index("detector").loc[dets]
        ax.bar(x + off, 100 * (r["match_rate"] - r["chance_rate"]), 0.4, color=col, label=series)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Match rate minus chance rate (points)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 5: the lift depends on the detector", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    x = np.arange(len(cov))
    ax.bar(x - 0.2, cov["matched_clusters"], 0.4, color=BLUE, label="matched")
    ax.bar(x + 0.2, cov["unmatched_clusters"], 0.4, color=RED, label="unmatched")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(y)}\n{c:.0f}%" for y, c in zip(cov["year"], cov["days_covered_%"])], fontsize=8.5)
    ax.set_ylabel("Clusters (excluding the edge window)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 6: unmatched clusters sit in uncovered years (% of days covered)", fontsize=10.5, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_summary.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("Saved data/chart_summary.png")


if __name__ == "__main__":
    main()
