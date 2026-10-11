"""
Side-by-side view of how the forecaster choice changes the two forecast-driven
strategies. Reads the summary_<forecaster>.json files written by
experiments.schedule_sim and writes SENSITIVITY.md next to them.

Run:  python -m experiments.forecaster_sensitivity
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config

DIR = os.path.join(config.RESULTS_DIR, "experiments", "schedule_sim")
FORECASTERS = ["persistence", "seasonal_naive", "profile_7d", "profile_anomaly"]
STRATS = ["best_start", "deadline_aware"]


def main():
    data = {}
    for f in FORECASTERS:
        p = os.path.join(DIR, f"summary_{f}.json")
        if os.path.exists(p):
            data[f] = json.load(open(p))["summary"]
    if not data:
        sys.exit("No summary_<forecaster>.json files found — run experiments.schedule_sim first")

    any_rows = next(iter(data.values()))
    zones = sorted({r["zone"] for r in any_rows})
    cfgs = sorted({(r["duration_h"], r["deadline_h"]) for r in any_rows})
    cols = list(data)

    def get(f, z, c, s):
        return next(r for r in data[f] if r["zone"] == z and (r["duration_h"], r["deadline_h"]) == c
                    and r["strategy"] == s)

    L = ["# Forecaster sensitivity\n",
         "CO2 reduction vs run_now, % (paired mean over jobs). `oracle` and `run_now` do not depend "
         "on the forecaster. Deadline-met is 100% for these two strategies in every cell.\n"]
    for s in STRATS:
        L.append(f"\n## {s}\n")
        L.append("| zone | job (work h / deadline h) | " + " | ".join(cols) + " | oracle |")
        L.append("|---|---|" + "---|" * (len(cols) + 1))
        for z in zones:
            for c in cfgs:
                cells = []
                for f in cols:
                    r = get(f, z, c, s)
                    cells.append(f"{r['reduction_pct']:+.1f}")
                orc = get(cols[0], z, c, "oracle")
                L.append(f"| {z} | {c[0]} / {c[1]} | " + " | ".join(cells)
                         + f" | {orc['reduction_pct']:+.1f} |")
    out = os.path.join(DIR, "SENSITIVITY.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
