# Forecast accuracy (causal, MAE in gCO2/kWh)

Forecast issued every 7 h across 2025 after a 7-day warm-up (so every hour of the day is covered); lower is better.


## DE  (mean intensity 340 gCO2/kWh)

| forecaster | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| persistence | 18.1 | 49.5 | 85.5 | 115.5 | 72.8 |
| seasonal_naive | 73.9 | 73.0 | 71.2 | 71.3 | 72.8 |
| profile_7d | 81.8 | 82.4 | 82.0 | 82.4 | 82.3 |
| profile_anomaly | 11.6 | 29.2 | 46.9 | 62.5 | 71.5 |

## IN  (mean intensity 597 gCO2/kWh)

| forecaster | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| persistence | 15.8 | 44.6 | 81.3 | 109.2 | 12.1 |
| seasonal_naive | 11.9 | 12.1 | 11.7 | 11.6 | 12.1 |
| profile_7d | 15.0 | 14.9 | 14.7 | 14.7 | 14.9 |
| profile_anomaly | 4.2 | 8.7 | 11.4 | 12.4 | 12.8 |

## US-CAL-CISO  (mean intensity 206 gCO2/kWh)

| forecaster | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| persistence | 14.9 | 43.1 | 82.7 | 121.7 | 25.8 |
| seasonal_naive | 25.5 | 25.7 | 25.0 | 25.2 | 25.8 |
| profile_7d | 28.2 | 28.0 | 28.3 | 28.3 | 28.2 |
| profile_anomaly | 6.2 | 12.7 | 19.1 | 24.3 | 24.8 |
