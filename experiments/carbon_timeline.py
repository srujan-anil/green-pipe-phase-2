import os
import pandas as pd
from scheduler import config

_cache = {}


def load_timeline(zone):
    if zone in _cache:
        return _cache[zone]

    path = config.resolve_zone_csv(zone)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["carbon_intensity"])
    df = df.tail(720).reset_index(drop=True)

    values = df["carbon_intensity"].astype(float).tolist()
    timestamps = df["timestamp"].astype(str).tolist()

    _cache[zone] = (values, timestamps)
    return _cache[zone]


def get_timeline_carbon(zone, sim_hour):
    timeline = load_timeline(zone)
    if timeline is None:
        return None, None

    values, timestamps = timeline
    if sim_hour >= len(values):
        return None, None

    return values[sim_hour], timestamps[sim_hour]


def timeline_length(zone):
    timeline = load_timeline(zone)
    return len(timeline[0]) if timeline else 0