"""
Centralized configuration for Green-Pipe.

Every threshold, weight, path, and seed used anywhere in the scheduler,
orchestrator, forecaster, or experiment framework should be read from here.
Nothing below is a research contribution — these are engineering knobs.
"""

import os

# ── Carbon thresholds (gCO2/kWh) ──────────────────────────────────────
THRESHOLD_LOW = 200    # below this: clean, train at full batch
THRESHOLD_HIGH = 300   # above this: dirty, pause

# ── WAIT band ──────────────────────────────────────────────────────────
# A forecast-integrated decision can choose WAIT (defer, do not checkpoint
# a full pause) when current carbon is in the WAIT band AND the forecast
# shows a materially better window within WAIT_HORIZON_HOURS.
WAIT_BAND_LOW = THRESHOLD_LOW
WAIT_BAND_HIGH = THRESHOLD_HIGH
WAIT_HORIZON_HOURS = 6
WAIT_IMPROVEMENT_THRESHOLD = 0.15   # forecast window must be >=15% cleaner to justify waiting

# ── Multi-objective scoring weights ───────────────────────────────────
W_CARBON = 0.5
W_COST = 0.3
W_DELAY = 0.2

# ── Zone electricity cost ($/kWh, approximate) ────────────────────────
ZONE_COST = {
    "DE": 0.35,
    "IN": 0.08,
    "US-CAL-CISO": 0.22,
}

ZONES = list(ZONE_COST.keys())
DEFAULT_ZONE = "DE"

# ── Adaptive batch size bands ──────────────────────────────────────────
BATCH_SIZE_HIGH_CARBON = 16
BATCH_SIZE_MED_CARBON = 32
BATCH_SIZE_LOW_CARBON = 64

# ── Reproducibility ─────────────────────────────────────────────────────
SEEDS = [42, 123, 7]

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RAW_RUNS_DIR = os.path.join(RESULTS_DIR, "raw_runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
EMISSIONS_DIR = os.path.join(RESULTS_DIR, "emissions")
FORECASTING_DIR = os.path.join(ROOT_DIR, "forecasting")

# ── Forecast horizon used for scheduling decisions ────────────────────
FORECAST_HORIZON_HOURS = 24
FORECAST_LOOKBACK_HOURS = 720   # matches forecasting/*.py "last 30 days"

# ── Experiment strategies ──────────────────────────────────────────────
STRATEGIES = [
    "baseline",       # always TRAIN, no carbon awareness
    "reactive",       # current-carbon-only threshold (original decision_engine behavior)
    "predictive",     # forecast-integrated (ARIMA or LSTM) TRAIN/WAIT/PAUSE
    "multiregion",    # predictive + region-simulation layer
    "adaptive",       # reactive + adaptive batch size
    "green_pipe",     # final integrated: predictive + adaptive batch + multiregion
]


def resolve_zone_csv(zone: str) -> str:
    return os.path.join(DATA_PROCESSED_DIR, f"{zone}_historical.csv")