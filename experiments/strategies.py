from scheduler.decision_engine import get_decision, get_adaptive_batch_size, decide
from scheduler.forecaster import get_forecast, get_best_forecaster
from scheduler.region_simulator import get_best_region_timeline
from scheduler import config

FIXED_BATCH_SIZE = 64


def decide_baseline(carbon, zone, sim_hour=None):
    return {
        "decision": "TRAIN",
        "reason": "Baseline strategy — no carbon awareness, always train",
        "batch_size": FIXED_BATCH_SIZE,
        "wait_hours": 0,
        "used_forecast": False,
        "effective_zone": zone,
        "effective_carbon": carbon,
    }


def decide_reactive(carbon, zone, sim_hour=None):
    decision, reason = get_decision(carbon, zone=zone)
    return {
        "decision": decision,
        "reason": reason,
        "batch_size": FIXED_BATCH_SIZE,
        "wait_hours": 0,
        "used_forecast": False,
        "effective_zone": zone,
        "effective_carbon": carbon,
    }


def decide_adaptive(carbon, zone, sim_hour=None):
    decision, reason = get_decision(carbon, zone=zone)
    batch_size, _ = get_adaptive_batch_size(carbon if carbon is not None else config.THRESHOLD_LOW)
    return {
        "decision": decision,
        "reason": reason,
        "batch_size": batch_size,
        "wait_hours": 0,
        "used_forecast": False,
        "effective_zone": zone,
        "effective_carbon": carbon,
    }


def decide_predictive(carbon, zone, sim_hour=None):
    model = get_best_forecaster(zone)
    forecast = get_forecast(zone, model=model) if model else None
    result = dict(decide(carbon, forecast=forecast, zone=zone))
    result["batch_size"] = FIXED_BATCH_SIZE
    result["effective_zone"] = zone
    result["effective_carbon"] = carbon
    return result


def decide_multiregion(carbon, zone, sim_hour=None):
    best_zone, best_carbon, readings = get_best_region_timeline(zone, sim_hour)
    decision, reason = get_decision(best_carbon, zone=best_zone)
    migrated = best_zone != zone
    tag = f"SIMULATED MIGRATION to {best_zone}" if migrated else f"STAYED in {zone}"
    return {
        "decision": decision,
        "reason": f"[{tag}] readings={readings} | {reason}",
        "batch_size": FIXED_BATCH_SIZE,
        "wait_hours": 0,
        "used_forecast": False,
        "effective_zone": best_zone,
        "effective_carbon": best_carbon,
    }


def decide_green_pipe(carbon, zone, sim_hour=None):
    best_zone, best_carbon, readings = get_best_region_timeline(zone, sim_hour)
    model = get_best_forecaster(best_zone)
    forecast = get_forecast(best_zone, model=model) if model else None
    result = dict(decide(best_carbon, forecast=forecast, zone=best_zone))
    batch_size, _ = get_adaptive_batch_size(best_carbon if best_carbon is not None else config.THRESHOLD_LOW)
    result["batch_size"] = batch_size
    migrated = best_zone != zone
    tag = f"SIMULATED MIGRATION to {best_zone}" if migrated else f"STAYED in {zone}"
    result["reason"] = f"[{tag}] readings={readings} | {result['reason']}"
    result["effective_zone"] = best_zone
    result["effective_carbon"] = best_carbon
    return result


DISPATCH = {
    "baseline": decide_baseline,
    "reactive": decide_reactive,
    "predictive": decide_predictive,
    "adaptive": decide_adaptive,
    "multiregion": decide_multiregion,
    "green_pipe": decide_green_pipe,
}


def decide_for(strategy, carbon, zone, sim_hour=None):
    if strategy not in DISPATCH:
        raise ValueError(f"Unknown strategy '{strategy}' — expected one of {list(DISPATCH.keys())}")
    return DISPATCH[strategy](carbon, zone, sim_hour=sim_hour)