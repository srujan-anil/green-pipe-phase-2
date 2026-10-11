
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler import config

# ── Backward-compatible module-level aliases (existing callers/tests
#    import these names directly from decision_engine) ─────────────────
THRESHOLD_LOW = config.THRESHOLD_LOW
THRESHOLD_HIGH = config.THRESHOLD_HIGH
W_CARBON = config.W_CARBON
W_COST = config.W_COST
W_DELAY = config.W_DELAY
ZONE_COST = config.ZONE_COST


# ═══════════════════════════════════════════════════════════════════════
# Legacy / reactive layer (current-carbon-only)
# ═══════════════════════════════════════════════════════════════════════

def get_decision(carbon_intensity, zone="DE", delay_hours=0):
    """Current-carbon-only TRAIN/PAUSE. Unchanged behavior from the
    original repo — this is the 'reactive' strategy baseline."""
    if carbon_intensity is None:
        return "TRAIN", "No carbon data available — defaulting to train"

    if carbon_intensity <= config.THRESHOLD_LOW:
        return "TRAIN", f"Carbon {carbon_intensity} gCO2/kWh — below low threshold {config.THRESHOLD_LOW}"
    elif carbon_intensity >= config.THRESHOLD_HIGH:
        return "PAUSE", f"Carbon {carbon_intensity} gCO2/kWh — above high threshold {config.THRESHOLD_HIGH}"
    else:
        return "TRAIN", f"Carbon {carbon_intensity} gCO2/kWh — within acceptable range"


def score_window(carbon_intensity, zone="DE", delay_hours=0):
    """Multi-objective score for a training window. Lower = better."""
    cost = config.ZONE_COST.get(zone, 0.20)
    score = (config.W_CARBON * carbon_intensity) + \
            (config.W_COST * cost * 1000) + \
            (config.W_DELAY * delay_hours)
    return round(score, 4)


def get_best_window(forecast_values, zone="DE"):
    """Best hour index (lowest score) within a forecast array."""
    if not forecast_values:
        return 0, "No forecast available — train immediately"

    scores = [(i, c, score_window(c, zone, delay_hours=i))
              for i, c in enumerate(forecast_values)]
    hour, carbon, score = min(scores, key=lambda x: x[2])
    return hour, f"Best window at hour +{hour} — Carbon: {carbon:.1f} gCO2/kWh — Score: {score}"


def get_best_region(timestamp_hour, zones=None):
    """Compare zones at a fixed historical timestamp index. Reads
    data/processed/{zone}_historical.csv directly — used by the legacy
    __main__ demo and by region_simulator.py's underlying data access."""
    if zones is None:
        zones = config.ZONES

    zone_carbon = {}
    for zone in zones:
        path = config.resolve_zone_csv(zone)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df = df.tail(720).reset_index(drop=True)
            if timestamp_hour < len(df):
                zone_carbon[zone] = df["carbon_intensity"].iloc[timestamp_hour]
        except Exception as e:
            print(f"Could not read {zone}: {e}")

    if not zone_carbon:
        return config.DEFAULT_ZONE, "No zone data available — defaulting to " + config.DEFAULT_ZONE

    best_zone = min(zone_carbon, key=zone_carbon.get)
    result = " | ".join(f"{z}: {v:.1f}" for z, v in zone_carbon.items())
    return best_zone, f"Zone comparison — {result} -> Best: {best_zone}"


def get_adaptive_batch_size(carbon_intensity):
    """Batch size scales inversely with carbon intensity."""
    if carbon_intensity > config.THRESHOLD_HIGH:
        return config.BATCH_SIZE_HIGH_CARBON, "High carbon — reduced batch size"
    elif carbon_intensity > config.THRESHOLD_LOW:
        return config.BATCH_SIZE_MED_CARBON, "Medium carbon — standard batch size"
    else:
        return config.BATCH_SIZE_LOW_CARBON, "Low carbon — maximum batch size"


# ═══════════════════════════════════════════════════════════════════════
# Forecast-integrated layer — the piece Step 3 asked for
# ═══════════════════════════════════════════════════════════════════════

def decide(current_carbon, forecast=None, zone="DE"):
    
    batch_size, batch_reason = get_adaptive_batch_size(
        current_carbon if current_carbon is not None else config.THRESHOLD_LOW
    )

    if current_carbon is None:
        return {
            "decision": "TRAIN",
            "reason": "No carbon data available — defaulting to train (fail-open)",
            "batch_size": batch_size,
            "wait_hours": 0,
            "used_forecast": False,
        }

    if current_carbon <= config.THRESHOLD_LOW:
        return {
            "decision": "TRAIN",
            "reason": f"Current carbon {current_carbon:.1f} gCO2/kWh already below "
                       f"low threshold {config.THRESHOLD_LOW} — train now, no need to consult forecast",
            "batch_size": batch_size,
            "wait_hours": 0,
            "used_forecast": False,
        }

    if current_carbon >= config.THRESHOLD_HIGH:
        return {
            "decision": "PAUSE",
            "reason": f"Current carbon {current_carbon:.1f} gCO2/kWh above high threshold "
                       f"{config.THRESHOLD_HIGH} — pause regardless of forecast",
            "batch_size": batch_size,
            "wait_hours": 0,
            "used_forecast": False,
        }

    # WAIT band
    if not forecast:
        return {
            "decision": "TRAIN",
            "reason": f"Current carbon {current_carbon:.1f} gCO2/kWh in WAIT band "
                       f"({config.THRESHOLD_LOW}-{config.THRESHOLD_HIGH}) but no forecast "
                       f"available — falling back to reactive TRAIN (matches current-carbon-only baseline)",
            "batch_size": batch_size,
            "wait_hours": 0,
            "used_forecast": False,
        }

    horizon = forecast[:config.WAIT_HORIZON_HOURS]
    best_hour = int(np.argmin(horizon))
    best_value = float(horizon[best_hour])
    improvement = (current_carbon - best_value) / current_carbon if current_carbon > 0 else 0.0

    if best_hour > 0 and improvement >= config.WAIT_IMPROVEMENT_THRESHOLD:
        return {
            "decision": "WAIT",
            "reason": f"Current carbon {current_carbon:.1f} gCO2/kWh in WAIT band. "
                       f"Forecast shows {best_value:.1f} gCO2/kWh at hour +{best_hour} "
                       f"({improvement:.0%} cleaner, exceeds {config.WAIT_IMPROVEMENT_THRESHOLD:.0%} threshold) — waiting",
            "batch_size": batch_size,
            "wait_hours": best_hour,
            "used_forecast": True,
        }

    return {
        "decision": "TRAIN",
        "reason": f"Current carbon {current_carbon:.1f} gCO2/kWh in WAIT band. "
                   f"Forecast's best window (+{best_hour}h, {best_value:.1f} gCO2/kWh) does not "
                   f"improve enough ({improvement:.0%} < {config.WAIT_IMPROVEMENT_THRESHOLD:.0%}) — training now",
        "batch_size": batch_size,
        "wait_hours": 0,
        "used_forecast": True,
    }


# ── Self-test (uses synthetic values, not persisted anywhere as a
#    result — this is a code sanity check, not an experiment) ──────────
if __name__ == "__main__":
    print("=" * 60)
    print(" Decision Engine Self-Test [SYNTHETIC — DEV FIXTURE ONLY]")
    print("=" * 60)

    print("\nTest 1: Legacy reactive decision")
    for carbon in [150, 250, 350]:
        d, r = get_decision(carbon, zone="DE")
        print(f"  carbon={carbon} -> {d} | {r}")

    print("\nTest 2: Forecast-integrated decide() — WAIT band, forecast improves")
    result = decide(250, forecast=[240, 230, 180, 170, 190, 210], zone="DE")
    print(f"  {result}")
    assert result["decision"] == "WAIT", "Expected WAIT when forecast improves >=15%"

    print("\nTest 3: Forecast-integrated decide() — WAIT band, forecast doesn't improve")
    result = decide(250, forecast=[245, 248, 250, 252], zone="DE")
    print(f"  {result}")
    assert result["decision"] == "TRAIN", "Expected TRAIN when forecast doesn't improve enough"

    print("\nTest 4: Forecast-integrated decide() — no forecast, matches reactive")
    result = decide(250, forecast=None, zone="DE")
    print(f"  {result}")
    assert result["decision"] == "TRAIN"
    assert result["used_forecast"] is False

    print("\nTest 5: Forecast-integrated decide() — dirty current, forecast irrelevant")
    result = decide(350, forecast=[100, 90, 80], zone="DE")
    print(f"  {result}")
    assert result["decision"] == "PAUSE", "Current-hour dirty carbon must pause regardless of forecast"

    print("\nAll self-tests passed (synthetic fixtures only, not experimental results).")