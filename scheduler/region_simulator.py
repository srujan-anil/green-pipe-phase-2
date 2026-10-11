from scheduler import config
from scheduler.carbon_client import fetch_live_carbon
from experiments.carbon_timeline import get_timeline_carbon


def get_best_region_timeline(zone, sim_hour, zones=None):
    zones = zones or config.ZONES
    readings = {}

    for z in zones:
        carbon, _ = get_timeline_carbon(z, sim_hour)
        readings[z] = carbon

    valid = {z: c for z, c in readings.items() if c is not None}
    if not valid:
        return zone, None, readings

    best_zone = min(valid, key=valid.get)
    return best_zone, valid[best_zone], readings

def get_best_region_live(zone, primary_carbon, zones=None):
    zones = zones or config.ZONES
    readings = {zone: primary_carbon}

    for z in zones:
        if z == zone:
            continue
        carbon, source = fetch_live_carbon(z)
        if carbon is None:
            from scheduler.orchestrator import get_current_carbon
            carbon, source = get_current_carbon(z)
        readings[z] = carbon

    valid = {z: c for z, c in readings.items() if c is not None}
    if not valid:
        return zone, primary_carbon, readings

    best_zone = min(valid, key=valid.get)
    return best_zone, valid[best_zone], readings