import os
from datetime import datetime, timezone
from codecarbon import EmissionsTracker

ENERGY_MEASUREMENT_SOURCE = "CodeCarbon"
CO2_CALCULATION_METHOD = "energy_kwh * carbon_intensity_g_per_kwh / 1000"


def track_epoch(fn, zone, carbon_intensity, carbon_source, epoch, output_dir, run_id):
    if carbon_intensity is None:
        raise ValueError(
            f"No carbon intensity available for zone={zone} — refusing to fabricate "
            f"a CO2 measurement basis. Check live API access or historical CSV fallback."
        )

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    tracker = EmissionsTracker(
        project_name=f"green_pipe_{run_id}",
        measure_power_secs=1,
        save_to_file=False,
        save_to_api=False,
        save_to_logger=False,
        log_level="error",
        allow_multiple_runs=True,
    )

    tracker.start()
    try:
        result = fn()
    finally:
        tracker.stop()

    data = tracker.final_emissions_data
    energy_kwh = float(data.energy_consumed)
    co2_kg = energy_kwh * float(carbon_intensity) / 1000.0

    metrics = {
        "epoch": epoch,
        "timestamp": timestamp,
        "region": zone,
        "carbon_intensity_g_per_kwh": float(carbon_intensity),
        "carbon_source": carbon_source,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "duration_s": float(data.duration),
        "energy_measurement_source": ENERGY_MEASUREMENT_SOURCE,
        "co2_calculation_method": CO2_CALCULATION_METHOD,
    }

    return result, metrics