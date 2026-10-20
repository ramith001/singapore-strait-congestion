# Case study: do shipping anomalies in the Malacca and Singapore straits track documented disruptions?

Ramith Vaid, NTU Singapore. Numbers below come from the README and the tables in `data/`.

## The question

Shipping through the Strait of Malacca and the Port of Singapore is a daily, public indicator of trade. IMF/Oxford PortWatch publishes vessel counts by type for both, and a calendar of documented disruptions (COVID-19, Panama Canal restrictions, Red Sea tensions, the Key Bridge collapse, the US ports strike, Hormuz 2026). The question was whether a simple detector that flags days far from an out-of-sample forecast finds those disruptions, and what else it finds.

## What was built

A baseline fitted only on the past: log of the daily count on a trend, day of week and month, refit every quarter on the trailing two years and scored on the following quarter, so every z-score is a forecast error on a day the model never saw. Days with |z| above 2.5 are grouped into clusters and compared with the event calendar (a cluster matches an event if it is within 14 days). The comparison that matters is against clusters placed at random dates in the same period, because the calendar covers more than half of the days.

## What the checks found

1. **A small lift over chance.** 83 of 129 clusters (64%) match an event, against about 56% for random placement, and the real rate is above chance in all 25 combinations of threshold and matching window tried (by 2 to 28 points).
2. **No distinct path around events.** Lining the z-score up on each of the six event starts and averaging, the path from 30 days before to 30 days after is not distinguishable from random dates (signed change -0.42, placebo p = 0.12). One apparent dip, COVID in early 2020, is an artifact: the first model is fitted on a single year and over-forecasts January 2020 by about 30%.
3. **Malacca and Singapore move together, with no lead.** The correlation of their anomalies is +0.263 on the same day (null band -0.19 to +0.20) and there is no asymmetry between leading and lagging days. That is what two-way traffic would produce.
4. **Tankers carry the flags.** Tanker anomalies correlate 0.79 (Malacca) and 0.83 (Singapore) with each site's total, but only container (+0.305) and roro (+0.269) anomalies correlate across the two sites.
5. **The lift depends on the detector.** A weekly-median detector and a seasonal-decomposition detector forecast better out of sample (R^2 0.73 and 0.78 against 0.48 at Malacca) and flag about a quarter to a third as many days. They reproduce 26% to 35% of the baseline's clusters against 9% to 14% by chance, but at Singapore their event-match rates (56% and 51%) sit at their chance rates (55% and 56%), where the baseline's is 72%.
6. **The unmatched clusters are mostly a calendar effect.** None of the 44 unmatched clusters outside the first fitting window is explained by holidays or other PortWatch events more often than random placement would give, and all 44 are in 2022, 2023 and 2025, years in which 4%, 20% and 0% of days are inside any event's matching window. With the coverage mask shifted at random, the excess of clusters on covered days (78 against 65 expected) has p = 0.21.

## A live data refresh

Pulling the current data on 2026-10-20 added 42 Malacca days, revised none of the 2,806 earlier days, flagged none of the new days, and left every headline number unchanged. The Port of Singapore could not be refreshed: the API now returns no rows for it, though it still does for Shanghai, Rotterdam and Ningbo. The analysis data are therefore kept as committed, the refresh is stored separately, and the fetch script stops instead of writing a Malacca series that no longer lines up with the Singapore one.

## What I would do next

Recover a Singapore series (or use MPA's vessel-movement data, which has direction), use a detector whose level adapts as the main one, add a moving-festival term, and extend the event calendar through 2022 and 2023 so that "unmatched" can be tested instead of assumed.

## What I take from it

The detector is a reasonable way to describe unusual days in a noisy count series, and it picks out a few large changes (COVID, Hormuz). Calling the flagged clusters evidence for documented disruptions needs more than a match rate: the result has to survive random placement, other detectors, an event study and an audit of what is left over. Here it survives the first, and weakens under each of the others.
