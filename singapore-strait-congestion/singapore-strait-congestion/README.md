# Singapore Strait Congestion: Out-of-Sample Anomaly Detection

https://github.com/ramith001/singapore-strait-congestion (suggested repo name)

- Pulled 7.7 years of daily vessel-transit data (2019-01-01 to present) for the Malacca Strait and the Port of Singapore directly from the IMF/Oxford PortWatch public ArcGIS API, alongside PortWatch's own documented disruption-events calendar, using a `requests`-based pagination script rather than a static download.
- Compared three candidate anomaly-detection targets (Singapore daily trade value, Singapore daily port calls, Malacca daily vessel count) on a trend + day-of-week + month-of-year baseline; trade value had almost no explainable seasonal structure (R^2 = 0.03) and was dropped in favor of vessel counts, which fit far better (R^2 up to 0.45), rather than forcing a poorly-fit target into the anomaly detector.
- Built the baseline with a rolling 2-year re-fit (numpy OLS, refit quarterly) instead of one static trend extrapolated forward for years; an initial static-trend version flagged anomalies almost exclusively in the final year of data and none in 2024, a sign of trend-extrapolation drift rather than genuine disruption, so it was rejected in favor of the rolling design before drawing any conclusions from it.
- Validated flagged anomalies against a ground-truth calendar of documented global shipping disruptions (COVID-19, Red Sea Tensions, Panama Canal drought restrictions, the Francis Scott Key Bridge collapse, the US ports strike, the 2026 Strait of Hormuz tensions): 83 of 129 flagged anomaly clusters fall within 14 days of a documented event, and the four largest anomalies in the entire 2019-2026 series (z-scores up to 7.3) all land inside the COVID-19 window, correctly identifying it as the single largest genuine disruption rather than treating it as noise.
- Flagged, rather than hid, weaker matches: two 2026 clusters coincide with the Strait of Hormuz tensions despite Hormuz having no direct shipping-lane overlap with Malacca, so they're reported as a plausible but mechanistically unconfirmed link, not stated as a confirmed cause.
- Tech stack: Python, requests, Pandas, NumPy, matplotlib.

## Files
- `fetch_data.py` — pulls the raw daily data and disruption calendar from PortWatch's public API (no key required) into `data/`.
- `analysis.py` — builds the rolling out-of-sample baseline, flags anomalies, cross-references them against the disruption calendar, and produces `data/chart_overview.png`.
- `data/` — raw and processed CSVs, plus the anomaly cluster table.

## Possible extensions
- Bring in the Singapore Maritime Data Hub (MPA's own live vessel-movement API) to see whether Singapore-specific vessel positioning corroborates the port-level story at finer granularity.
- Extend the same rolling-baseline method to other Asia-Pacific chokepoints (Taiwan Strait, Korea Strait) PortWatch tracks, to see whether Malacca's anomaly timing is regional or Malacca-specific.
