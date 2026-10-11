"""
Causal forecast accuracy of the scheduler's forecasters, by horizon.

Every forecast at hour t uses only ci[:t+1]. Compares against the persistence
and seasonal-naive baselines that any forecaster must beat.

Run:  python -m experiments.forecast_accuracy
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config
from scheduler.deadline_scheduler import FORECASTERS
from experiments.schedule_sim import HISTORY_H, load_trace

HORIZONS = [1, 3, 6, 12, 24]


def mae_by_horizon(ci, forecaster, every=7):   # coprime with 24: all hours of day covered
    errs = {h: [] for h in HORIZONS}
    for t in range(HISTORY_H, len(ci) - max(HORIZONS) - 1, every):
        fc = forecaster(ci[: t + 1], max(HORIZONS))
        for h in HORIZONS:
            errs[h].append(abs(fc[h - 1] - ci[t + h]))
    return {h: float(np.mean(v)) for h, v in errs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="+", default=config.ZONES)
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "experiments",
                                                  "schedule_sim", "FORECAST_ACCURACY.md"))
    args = ap.parse_args()

    lines = ["# Forecast accuracy (causal, MAE in gCO2/kWh)\n",
             "Forecast issued every 7 h across 2025 after a 7-day warm-up (so every hour of the day is "
             "covered); lower is better.\n"]
    for z in args.zones:
        ci = load_trace(z)
        lines.append(f"\n## {z}  (mean intensity {ci.mean():.0f} gCO2/kWh)\n")
        lines.append("| forecaster | " + " | ".join(f"+{h}h" for h in HORIZONS) + " |")
        lines.append("|---|" + "---|" * len(HORIZONS))
        for name, f in FORECASTERS.items():
            m = mae_by_horizon(ci, f)
            lines.append(f"| {name} | " + " | ".join(f"{m[h]:.1f}" for h in HORIZONS) + " |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
