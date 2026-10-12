# Singapore Strait Congestion: Out-of-Sample Anomaly Detection

https://github.com/ramith001/singapore-strait-congestion (suggested repo name)

- Pulled 7.7 years of daily vessel-transit data (2019-01-01 to present) for the Malacca Strait and the Port of Singapore directly from the IMF/Oxford PortWatch public ArcGIS API, alongside PortWatch's own documented disruption-events calendar, using a `requests`-based pagination script rather than a static download.
- **Model selection** (`python model_selection.py`, also printed at the top of `analysis.py`): compared three candidate targets (Malacca daily vessel count, Singapore daily port calls, Singapore daily trade value) under the identical rolling out-of-sample method: log-transform, trend + day-of-week + month baseline refit quarterly on the trailing 2 years, scored only on the following quarter, |z| > 2.5, clustered, matched to the event calendar within 14 days. Results:

  | Target | Mean training-window R^2 | Out-of-sample R^2 | Clusters | Matched within 14 days | Match rate if clusters were placed at random |
  |---|---|---|---|---|---|
  | Malacca vessel count | 0.230 | 0.383 | 129 | 83 (64%) | 56% |
  | Singapore port calls | 0.347 | -0.283 | 83 | 60 (72%) | 56% |
  | Singapore trade value | 0.073 | -0.004 | 65 | 52 (80%) | 55% |

  The choice of Malacca does not rest on a clean R^2 win. In-sample, port calls fit the training windows better (0.347 vs 0.230). Out of sample, which is what an anomaly detector actually depends on, Malacca is the only candidate whose forecast beats a flat average (0.383); the port-calls baseline does worse than the mean on the days it scores (-0.283), and trade value is about zero. Malacca was kept because of that out-of-sample fit and because it is a direct count of vessels through one chokepoint, with the most flagged days and largest anomalies. Matching to documented disruptions does not separate the candidates either: all three beat the random-placement rate, and Malacca's lift (64% vs 56%) is the smallest, so disruption matching is supporting evidence, not a reason to prefer Malacca. An earlier version of this note cited R^2 = 0.45 for Malacca; that was the static trend model rejected below, and the old trade-value figure (0.03) was never produced by the committed code.
- Built the baseline with a rolling 2-year re-fit (numpy OLS, refit quarterly) instead of one static trend extrapolated forward for years; an initial static-trend version flagged anomalies almost exclusively in the final year of data and none in 2024, a sign of trend-extrapolation drift rather than genuine disruption, so it was rejected in favor of the rolling design before drawing any conclusions from it.
- Validated flagged anomalies against a ground-truth calendar of documented global shipping disruptions (COVID-19, Red Sea Tensions, Panama Canal drought restrictions, the Francis Scott Key Bridge collapse, the US ports strike, the 2026 Strait of Hormuz tensions): 83 of 129 flagged anomaly clusters fall within 14 days of a documented event (64%, against about 56% expected if the same clusters sat at random dates, since the calendar covers much of 2020-2026), and the four largest anomalies in the entire 2019-2026 series (z-scores up to 7.3) all land inside the COVID-19 window, correctly identifying it as the single largest genuine disruption rather than treating it as noise.
- Flagged, rather than hid, weaker matches: two 2026 clusters coincide with the Strait of Hormuz tensions despite Hormuz having no direct shipping-lane overlap with Malacca, so they're reported as a plausible but mechanistically unconfirmed link, not stated as a confirmed cause.
- Tech stack: Python, requests, Pandas, NumPy, matplotlib.

## Files
- `fetch_data.py` — pulls the raw daily data and disruption calendar from PortWatch's public API (no key required) into `data/`.
- `baseline.py` — the shared rolling out-of-sample fit, anomaly clustering and event matching, used by both scripts below.
- `model_selection.py` — the three-way target comparison above; regenerates every model-selection number in this README.
- `analysis.py` — builds the rolling out-of-sample baseline, flags anomalies, cross-references them against the disruption calendar, and produces `data/chart_overview.png`.
- `data/` — raw and processed CSVs, plus the anomaly cluster table.

## Possible extensions
- Bring in the Singapore Maritime Data Hub (MPA's own live vessel-movement API) to see whether Singapore-specific vessel positioning corroborates the port-level story at finer granularity.
- Extend the same rolling-baseline method to other Asia-Pacific chokepoints (Taiwan Strait, Korea Strait) PortWatch tracks, to see whether Malacca's anomaly timing is regional or Malacca-specific.
