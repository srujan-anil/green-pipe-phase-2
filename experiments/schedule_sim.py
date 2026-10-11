"""
Scheduling simulation on real hourly carbon traces.

Why this exists
---------------
The per-epoch experiments in runner.py measure real energy with CodeCarbon, but
on shared hardware the energy of one run varies by roughly +/-50%, which swamps
the effect of choosing a cleaner hour. This simulation holds the job's energy
fixed (duration x power) and measures only what the scheduler controls: which
hours the job runs in, whether it meets its deadline, and the resulting gCO2.

Every strategy gets exactly the same jobs (same submission hour, same duration,
same deadline) over the full year of data, so results are paired.

Rules that keep it honest
-------------------------
* Causal: a strategy deciding at hour t only sees ci[:t+1].
* A job that is still unfinished at its deadline runs flat out afterwards, so
  its emissions are still counted and it is recorded as a deadline miss.
* Emissions include the extra energy of resuming from a checkpoint after a pause.
* "oracle" sees the real future. It is a lower bound, not a deployable strategy.

Run:  python -m experiments.schedule_sim            (from the repo root)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config
from scheduler.deadline_scheduler import (
    FORECASTERS, DeadlineAwareScheduler, Job, plan_contiguous, plan_resumable,
    schedule_cost,
)

STRATEGIES = [
    "run_now",          # no carbon awareness
    "threshold_abs",    # the repo's reactive rule: pause while ci >= THRESHOLD_HIGH
    "threshold_rel",    # pause while ci is above its trailing-7-day median (+ deadline guard)
    "best_start",       # pick one start time from a forecast, then run straight through
    "deadline_aware",   # re-plan every hour, can pause/resume, hard deadline
    "oracle",           # perfect foresight, same freedom (lower bound)
]

# (duration_h, deadline_h)
JOB_CONFIGS = [(2, 12), (8, 10), (8, 16), (8, 24), (8, 48), (8, 96), (24, 168)]

HISTORY_H = 168          # hours of history every strategy is allowed to look back on
# First choice was profile_7d. A quick run showed the hourly re-planner losing to the
# one-shot planner with it, and a forecast-error check (REPORT: forecast accuracy table)
# showed why: profile_7d ignores that today differs from the average day. Switched to
# profile_anomaly, whose only parameter (the decay rate) is estimated online from the
# trailing week. All forecasters are re-run as a sensitivity check.
DEFAULT_FORECASTER = "profile_anomaly"


# ───────────────────────── data ─────────────────────────

def load_trace(zone):
    path = config.resolve_zone_csv(zone)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").dropna(subset=["carbon_intensity"])
    steps = df["timestamp"].diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        raise ValueError(f"{zone}: trace has gaps; simulation needs contiguous hourly data")
    return df["carbon_intensity"].to_numpy(dtype=float)


# ───────────────────────── accounting ─────────────────────────

def account(ci, start, job, hours_run):
    """Turn a list of absolute run-hours into outcome metrics."""
    hours_run = sorted(hours_run)
    assert len(hours_run) == job.duration_h, "strategy did not run exactly duration_h hours"
    end = hours_run[-1] + 1
    span = np.zeros(end - start, dtype=bool)
    span[np.array(hours_run) - start] = True
    cost = schedule_cost(ci[start:end], span, started=False, was_running=False,
                         resume_overhead_h=job.resume_overhead_h)
    finish = end - start
    resumes = int(np.sum(span[1:] & ~span[:-1]))
    return {
        "gco2": job.power_kw * cost,
        "finish_h": finish,
        "delay_h": finish - job.duration_h,
        "met": finish <= job.deadline_h,
        "resumes": resumes,
    }


# ───────────────────────── strategies ─────────────────────────

def run_now(ci, start, job, forecaster):
    return list(range(start, start + job.duration_h))


def threshold_abs(ci, start, job, forecaster):
    hours, h = [], start
    while len(hours) < job.duration_h:
        late = (h - start) >= job.deadline_h
        if late or ci[h] < config.THRESHOLD_HIGH:
            hours.append(h)
        h += 1
    return hours


def threshold_rel(ci, start, job, forecaster, window=HISTORY_H, pct=50):
    hours, h = [], start
    while len(hours) < job.duration_h:
        remaining = job.duration_h - len(hours)
        slack_window = job.deadline_h - (h - start)
        thr = np.percentile(ci[max(0, h - window): h + 1], pct)
        if slack_window <= remaining or ci[h] <= thr:
            hours.append(h)
        h += 1
    return hours


def best_start(ci, start, job, forecaster):
    f = FORECASTERS[forecaster]
    costs = np.concatenate([[ci[start]], f(ci[: start + 1], job.deadline_h - 1)])
    off, _ = plan_contiguous(costs, job.duration_h)
    return list(range(start + off, start + off + job.duration_h))


def deadline_aware(ci, start, job, forecaster):
    sched = DeadlineAwareScheduler(forecaster)
    hours, h, was_running = [], start, False
    while len(hours) < job.duration_h:
        res = sched.decide(ci[: h + 1], job, len(hours), h - start, was_running)
        was_running = res["decision"] == "TRAIN"
        if was_running:
            hours.append(h)
        h += 1
    return hours


def oracle(ci, start, job, forecaster):
    costs = ci[start: start + job.deadline_h]
    mask, _ = plan_resumable(costs, job.duration_h, False, False, job.resume_overhead_h)
    return [start + int(i) for i in np.flatnonzero(mask)]


STRATEGY_FUNCS = {
    "run_now": run_now, "threshold_abs": threshold_abs, "threshold_rel": threshold_rel,
    "best_start": best_start, "deadline_aware": deadline_aware, "oracle": oracle,
}


# ───────────────────────── experiment ─────────────────────────

def valid_starts(n, job, stride):
    # leave room for HISTORY_H of look-back and for a late finish after the deadline
    last = n - job.deadline_h - job.duration_h - 1
    return list(range(HISTORY_H, last, stride))


def run_cell(args):
    """One (zone, job config): every strategy on every start time."""
    zone, duration, deadline, stride, forecaster = args
    ci = load_trace(zone)
    job = Job(duration_h=duration, deadline_h=deadline)
    starts = valid_starts(len(ci), job, stride)
    rows = {s: [] for s in STRATEGIES}
    for st in starts:
        for name in STRATEGIES:
            hours = STRATEGY_FUNCS[name](ci, st, job, forecaster)
            rows[name].append(account(ci, st, job, hours))
    return zone, duration, deadline, starts, rows


def block_bootstrap_ci(values, block, n_boot=2000, seed=0):
    """95% CI for the mean of an autocorrelated series using contiguous blocks."""
    v = np.asarray(values, dtype=float)
    nb = len(v) // block
    if nb < 2:
        return (float("nan"), float("nan"))
    blocks = v[: nb * block].reshape(nb, block)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    means = blocks[idx].mean(axis=(1, 2))
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(zone, duration, deadline, starts, rows, stride):
    base = np.array([r["gco2"] for r in rows["run_now"]])
    block = max(1, math.ceil(168 / stride))   # 7-day blocks
    out = []
    for name in STRATEGIES:
        g = np.array([r["gco2"] for r in rows[name]])
        red = 100.0 * (base - g) / base
        lo, hi = block_bootstrap_ci(red, block)
        out.append({
            "zone": zone, "duration_h": duration, "deadline_h": deadline,
            "strategy": name, "n_jobs": len(g),
            "mean_gco2_per_kw": float(g.mean()),
            "reduction_pct": float(red.mean()),
            "reduction_ci_lo": float(lo), "reduction_ci_hi": float(hi),
            "deadline_met_pct": 100.0 * float(np.mean([r["met"] for r in rows[name]])),
            "mean_delay_h": float(np.mean([r["delay_h"] for r in rows[name]])),
            "mean_resumes": float(np.mean([r["resumes"] for r in rows[name]])),
        })
    return out


def write_report(summary, path, meta):
    zones = sorted({s["zone"] for s in summary})
    L = []
    L.append("# Scheduling simulation results\n")
    L.append(f"Generated by `python -m experiments.schedule_sim` — forecaster for "
             f"`deadline_aware`/`best_start`: **{meta['forecaster']}**, one job submitted every "
             f"{meta['stride']} h across 2025 (after the first {HISTORY_H} h; submission hour of day "
             f"cycles through {24 // math.gcd(meta['stride'], 24)} of 24 hours), "
             f"{meta['seconds']:.0f}s runtime.\n")
    L.append("Reduction is the paired % drop in gCO2 versus `run_now` for the same job "
             "(mean over jobs, 95% CI from a 7-day block bootstrap). Positive = cleaner. "
             "Energy per job is identical for every strategy, so this isolates scheduling.\n")
    for z in zones:
        L.append(f"\n## {z}\n")
        L.append("### CO2 reduction vs run_now, % [95% CI]\n")
        L.append("| job (work h / deadline h) | " + " | ".join(STRATEGIES[1:]) + " |")
        L.append("|---|" + "---|" * (len(STRATEGIES) - 1))
        cfgs = sorted({(s["duration_h"], s["deadline_h"]) for s in summary if s["zone"] == z},
                      key=lambda c: (c[0], c[1]))
        for d, dl in cfgs:
            cells = []
            for name in STRATEGIES[1:]:
                s = next(x for x in summary if x["zone"] == z and x["duration_h"] == d
                         and x["deadline_h"] == dl and x["strategy"] == name)
                cells.append(f"{s['reduction_pct']:+.1f} [{s['reduction_ci_lo']:+.1f}, {s['reduction_ci_hi']:+.1f}]")
            L.append(f"| {d} / {dl} | " + " | ".join(cells) + " |")
        L.append("\n### Deadline met, % of jobs (mean delay h in brackets)\n")
        L.append("| job (work h / deadline h) | " + " | ".join(STRATEGIES) + " |")
        L.append("|---|" + "---|" * len(STRATEGIES))
        for d, dl in cfgs:
            cells = []
            for name in STRATEGIES:
                s = next(x for x in summary if x["zone"] == z and x["duration_h"] == d
                         and x["deadline_h"] == dl and x["strategy"] == name)
                cells.append(f"{s['deadline_met_pct']:.0f}% ({s['mean_delay_h']:.1f})")
            L.append(f"| {d} / {dl} | " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="+", default=config.ZONES)
    ap.add_argument("--stride", type=int, default=11,
                    help="hours between job submissions; keep it coprime with 24 so jobs "
                         "arrive at every hour of the day (default 11)")
    ap.add_argument("--forecaster", default=DEFAULT_FORECASTER, choices=list(FORECASTERS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="override job configs as duration:deadline, e.g. 8:24 8:48")
    ap.add_argument("--out-dir", default=os.path.join(config.RESULTS_DIR, "experiments", "schedule_sim"))
    args = ap.parse_args()

    if math.gcd(args.stride, 24) != 1:
        print(f"WARNING: stride {args.stride} shares a factor with 24, so jobs are only "
              f"submitted at {24 // math.gcd(args.stride, 24)} distinct hours of the day; "
              f"results will not represent the daily cycle. Use e.g. 7, 11 or 13.")

    cfgs = JOB_CONFIGS
    if args.configs:
        cfgs = [tuple(int(x) for x in c.split(":")) for c in args.configs]

    tasks = [(z, d, dl, args.stride, args.forecaster) for z in args.zones for d, dl in cfgs]
    t0 = time.time()
    with Pool(args.workers) as pool:
        cells = pool.map(run_cell, tasks, chunksize=1)

    summary = []
    for zone, d, dl, starts, rows in cells:
        summary.extend(summarize(zone, d, dl, starts, rows, args.stride))
    secs = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"summary_{args.forecaster}.json"), "w") as f:
        json.dump({"forecaster": args.forecaster, "stride": args.stride, "seconds": secs,
                   "summary": summary}, f, indent=2)
    with open(os.path.join(args.out_dir, f"summary_{args.forecaster}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    write_report(summary, os.path.join(args.out_dir, f"REPORT_{args.forecaster}.md"),
                 {"forecaster": args.forecaster, "stride": args.stride, "seconds": secs})
    print(f"Done in {secs:.0f}s — wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
