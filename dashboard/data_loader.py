import os
import sys
import json
import glob
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scheduler import config
from scheduler.orchestrator import get_current_carbon
from scheduler.decision_engine import decide, get_adaptive_batch_size
from scheduler.forecaster import get_forecast, get_best_forecaster
from scheduler.region_simulator import get_best_region_live

DIRTY_EVENT_SEEDS = [42, 123, 7, 1, 2, 3, 4, 5, 6]
ANOMALY_THRESHOLD = 3.0


def get_thresholds():
    return {
        "low": config.THRESHOLD_LOW,
        "high": config.THRESHOLD_HIGH,
        "wait_horizon_hours": config.WAIT_HORIZON_HOURS,
        "wait_improvement_threshold": config.WAIT_IMPROVEMENT_THRESHOLD,
    }


def carbon_status_label(carbon):
    if carbon is None:
        return "UNKNOWN"
    if carbon <= config.THRESHOLD_LOW:
        return "CLEAN"
    if carbon >= config.THRESHOLD_HIGH:
        return "DIRTY"
    return "MODERATE"


def get_all_regions_status():
    rows = []
    for z in config.ZONES:
        carbon, source = get_current_carbon(z)
        rows.append({"zone": z, "carbon": carbon, "source": source, "status": carbon_status_label(carbon)})
    valid = [r for r in rows if r["carbon"] is not None]
    best_zone = min(valid, key=lambda r: r["carbon"])["zone"] if valid else None
    for r in rows:
        r["is_best"] = (r["zone"] == best_zone)
    return rows


def get_live_green_pipe_decision(zone):
    carbon, source = get_current_carbon(zone)
    best_zone, best_carbon, readings = get_best_region_live(zone, carbon)
    model = get_best_forecaster(best_zone)
    forecast = get_forecast(best_zone, model=model) if model else None
    result = decide(best_carbon, forecast=forecast, zone=best_zone)
    batch_size, _ = get_adaptive_batch_size(best_carbon if best_carbon is not None else config.THRESHOLD_LOW)
    return {
        "base_zone": zone, "base_carbon": carbon, "base_source": source, "readings": readings,
        "effective_zone": best_zone, "effective_carbon": best_carbon,
        "migrated": best_zone != zone, "forecaster_model": model, "forecast": forecast,
        "decision": result["decision"], "reason": result["reason"],
        "batch_size": batch_size, "wait_hours": result["wait_hours"],
    }


def parse_metrics_file(path):
    if not os.path.exists(path):
        return None
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
    return data


def get_forecasting_comparison():
    rows = []
    for zone in config.ZONES:
        arima = parse_metrics_file(os.path.join(config.RESULTS_DIR, f"arima_{zone}_metrics.txt"))
        lstm = parse_metrics_file(os.path.join(config.RESULTS_DIR, f"lstm_{zone}_metrics.txt"))
        rows.append({
            "zone": zone,
            "arima_mape": arima.get("MAPE") if arima else None,
            "arima_rmse": arima.get("RMSE") if arima else None,
            "lstm_mape": lstm.get("MAPE") if lstm else None,
            "lstm_rmse": lstm.get("RMSE") if lstm else None,
            "selected_model": get_best_forecaster(zone),
        })
    return rows


def get_carbon_history(zone, hours=168):
    path = config.resolve_zone_csv(zone)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["carbon_intensity"])
    return df.tail(hours).reset_index(drop=True)


def get_strategy_summary():
    path = os.path.join(config.RESULTS_DIR, "experiments", "summaries", "strategy_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def list_experiment_runs():
    raw_dir = os.path.join(config.RESULTS_DIR, "experiments", "raw")
    if not os.path.isdir(raw_dir):
        return []
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        with open(path, "r") as f:
            data = json.load(f)
        runs.append({"file": os.path.basename(path), "path": path, "strategy": data.get("strategy"), "seed": data.get("seed")})
    return runs


def load_run(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_raw_run(strategy, seed, zone="DE"):
    path = os.path.join(config.RESULTS_DIR, "experiments", "raw", f"{strategy}_{zone}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _detect_timing_anomaly(run):
    if run is None:
        return False, None
    train_epochs = [e for e in run.get("per_epoch", []) if e.get("energy_kwh", 0) > 0 and "duration_s" in e]
    if len(train_epochs) < 2:
        return False, None
    durations = [e["duration_s"] for e in train_epochs]
    median = statistics.median(durations)
    if median <= 0:
        return False, None
    for e in train_epochs:
        if e["duration_s"] >= ANOMALY_THRESHOLD * median:
            return True, f"epoch {e['epoch']} took {e['duration_s']:.1f}s vs run median {median:.1f}s ({e['duration_s']/median:.1f}x)"
    return False, None


def get_dirty_event_results():
    rows = []
    for seed in DIRTY_EVENT_SEEDS:
        baseline = _load_raw_run("baseline", seed)
        green_pipe = _load_raw_run("green_pipe", seed)
        if baseline is None or green_pipe is None:
            continue
        anomaly, anomaly_reason = _detect_timing_anomaly(green_pipe)
        base_co2 = baseline["co2_kg"]
        gp_co2 = green_pipe["co2_kg"]
        pct_change = ((gp_co2 - base_co2) / base_co2 * 100) if base_co2 else None
        rows.append({
            "seed": seed, "baseline_co2_kg": base_co2, "green_pipe_co2_kg": gp_co2,
            "pct_change": pct_change, "anomaly": anomaly, "anomaly_reason": anomaly_reason,
        })
    clean = [r["pct_change"] for r in rows if not r["anomaly"] and r["pct_change"] is not None]
    stats = {"n_total": len(rows), "n_excluded": sum(1 for r in rows if r["anomaly"]), "n_used": len(clean), "threshold": ANOMALY_THRESHOLD}
    if clean:
        stats["mean_pct_change"] = statistics.mean(clean)
        stats["std_pct_change"] = statistics.stdev(clean) if len(clean) > 1 else 0.0
        se = stats["std_pct_change"] / (len(clean) ** 0.5) if len(clean) > 1 else 0.0
        stats["ci_low"] = stats["mean_pct_change"] - 2 * se
        stats["ci_high"] = stats["mean_pct_change"] + 2 * se
        stats["inconclusive"] = stats["ci_low"] < 0 < stats["ci_high"]
    else:
        stats.update({"mean_pct_change": None, "std_pct_change": None, "ci_low": None, "ci_high": None, "inconclusive": None})
    return rows, stats


def get_dirty_event_timeline(seed):
    baseline = _load_raw_run("baseline", seed)
    green_pipe = _load_raw_run("green_pipe", seed)
    if baseline is None or green_pipe is None:
        return None, None, None, None
    gp_timeline = [{"sim_hour": e.get("sim_hour"), "carbon": e.get("carbon_intensity_g_per_kwh"),
                     "region": e.get("region"), "decision": e.get("decision")}
                    for e in green_pipe.get("per_epoch", [])]
    gp_df = pd.DataFrame(gp_timeline).drop_duplicates(subset="sim_hour").sort_values("sim_hour")
    base_trained = sorted(set(e.get("sim_hour") for e in baseline.get("per_epoch", []) if e.get("decision") == "TRAIN"))
    gp_trained = sorted(set(e.get("sim_hour") for e in green_pipe.get("per_epoch", []) if e.get("decision") == "TRAIN"))
    gp_paused = sorted(set(e.get("sim_hour") for e in green_pipe.get("per_epoch", []) if e.get("decision") == "PAUSE"))
    return gp_df, base_trained, gp_trained, gp_paused


def get_checkpoint_sweep_results():
    pattern = os.path.join(config.RESULTS_DIR, "experiments", "checkpoint_threshold", "sweep_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    by_tier = {}
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        for tier in data:
            by_tier.setdefault(tier["model"], []).append(tier)
    rows = []
    for tier_name, entries in by_tier.items():
        overheads = [e["total_overhead_pct_of_epoch"] for e in entries]
        rows.append({
            "tier": tier_name, "param_count": entries[0]["param_count"], "n_seeds": len(entries),
            "mean_overhead_pct": statistics.mean(overheads),
            "std_overhead_pct": statistics.stdev(overheads) if len(overheads) > 1 else 0.0,
            "mean_epoch_time_s": statistics.mean(e["avg_epoch_time_s"] for e in entries),
        })
    order = {"Tier1_Small": 0, "Tier2_Medium": 1, "Tier3_Large": 2}
    rows.sort(key=lambda r: order.get(r["tier"], 99))
    return rows


def get_hardware_usage():
    try:
        import psutil
        import platform
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_name": platform.processor() or "unknown",
            "available": True,
        }
    except ImportError:
        return {"available": False}

def get_live_reading_now(zone):
    carbon, source = get_current_carbon(zone)
    return {"carbon": carbon, "source": source}