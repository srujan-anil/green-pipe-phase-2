"""
Deadline-aware carbon scheduler.

Given a training job (how many compute-hours it needs and by when it must be
finished), decide hour by hour whether to TRAIN or WAIT so the job finishes on
time while using the cleanest hours it can.

Design notes
------------
* Pure Python + numpy. No torch, no network, so it can be unit-tested anywhere.
* Causal: the only inputs are the carbon history up to and including the
  current hour. Nothing here ever sees a future reading.
* Receding horizon: every hour the scheduler re-plans with the newest reading,
  so forecast errors get corrected instead of accumulating.
* The deadline is a hard constraint. When the remaining slack reaches zero the
  scheduler always trains ("forced"), whatever the forecast says.
* The plan is exact (dynamic programming), not a heuristic, and it prices in
  the extra energy it costs to resume from a checkpoint after a pause.

Hour conventions
----------------
Hour 0 of a plan is the *current* hour. Its carbon value is the live reading
(known). Hours 1.. use forecasts.  `deadline_h` counts hours from submission:
the job must be finished by the end of hour `deadline_h - 1`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Job description
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Job:
    duration_h: int                    # compute-hours of work needed
    deadline_h: int                    # hours after submission by which it must be done
    power_kw: float = 1.0              # average power draw while training
    resume_overhead_h: float = 0.05    # extra compute-hours (at that hour's carbon)
                                       # paid each time training resumes after a pause

    def __post_init__(self):
        if self.duration_h < 1:
            raise ValueError("duration_h must be >= 1")
        if self.deadline_h < self.duration_h:
            raise ValueError("deadline_h must be >= duration_h")
        if self.resume_overhead_h < 0:
            raise ValueError("resume_overhead_h must be >= 0")


# ═══════════════════════════════════════════════════════════════════════
# Causal forecasters
#   f(history, horizon) -> array of `horizon` forecasts for the hours
#   *after* history[-1]. history[-1] is the current-hour reading.
# ═══════════════════════════════════════════════════════════════════════

def _season_index(t, horizon, period=24):
    """For future hour t+j (j=1..horizon) return the most recent index with the
    same hour-of-day that is already known (<= t)."""
    j = np.arange(1, horizon + 1)
    return t + j - period * np.ceil(j / period).astype(int)


def forecast_persistence(history, horizon):
    """Tomorrow looks like right now."""
    return np.full(horizon, float(history[-1]))


def forecast_seasonal_naive(history, horizon, period=24):
    """Each future hour looks like the same hour yesterday."""
    h = np.asarray(history, dtype=float)
    t = len(h) - 1
    idx = _season_index(t, horizon, period)
    out = np.where(idx >= 0, h[np.clip(idx, 0, None)], h[-1])
    return out.astype(float)


def forecast_profile(history, horizon, days=7, period=24):
    """Each future hour looks like the average of that hour-of-day over the
    last `days` days. Smooths out one-off spikes in yesterday's data."""
    h = np.asarray(history, dtype=float)
    t = len(h) - 1
    base = _season_index(t, horizon, period)
    total = np.zeros(horizon)
    count = np.zeros(horizon)
    for k in range(days):
        idx = base - period * k
        ok = idx >= 0
        total[ok] += h[idx[ok]]
        count[ok] += 1
    return np.where(count > 0, total / np.maximum(count, 1), h[-1])


def forecast_profile_anomaly(history, horizon, days=7, period=24):
    """Daily profile plus a decaying 'today is different' correction.

    On a windy day the whole day is cleaner than the average profile; on a calm
    day it is dirtier. A pure profile forecast ignores that, so a re-planner
    comparing a live reading against flat averages is biased. Here the current
    deviation from the profile is carried forward and fades at a rate (phi per
    hour) estimated from the trailing `days` days of the *same history* — no
    constants tuned on the evaluation data.
    """
    h = np.asarray(history, dtype=float)
    t = len(h) - 1
    prof = forecast_profile(h, horizon, days, period)

    n = min(days * period, t + 1)
    idx = np.arange(t - n + 1, t + 1)
    total = np.zeros(n)
    count = np.zeros(n)
    for k in range(1, days + 1):
        lag = idx - period * k
        ok = lag >= 0
        total[ok] += h[lag[ok]]
        count[ok] += 1
    valid = count > 0
    if valid.sum() < 2 * period:
        return prof                                   # not enough history yet
    anomaly = np.where(valid, h[idx] - total / np.maximum(count, 1), np.nan)
    a = anomaly[valid]
    denom = float(np.sum(a[:-1] ** 2))
    phi = float(np.clip(np.sum(a[1:] * a[:-1]) / denom, 0.0, 0.99)) if denom > 0 else 0.0
    current = float(anomaly[-1]) if valid[-1] else 0.0
    return prof + current * phi ** np.arange(1, horizon + 1)


FORECASTERS = {
    "persistence": forecast_persistence,
    "seasonal_naive": forecast_seasonal_naive,
    "profile_7d": forecast_profile,
    "profile_anomaly": forecast_profile_anomaly,
}


# ═══════════════════════════════════════════════════════════════════════
# Cost model (single source of truth for planning AND accounting)
# ═══════════════════════════════════════════════════════════════════════

def schedule_cost(costs, run_mask, started, was_running, resume_overhead_h):
    """Carbon cost (gCO2 per kW of job power) of running in the hours where
    run_mask is True. A resume (idle hour -> run hour after training has
    already started) costs `resume_overhead_h` extra compute-hours at that
    hour's carbon intensity."""
    total = 0.0
    prev = bool(was_running)
    began = bool(started)
    for c, r in zip(costs, run_mask):
        if r:
            total += float(c)
            if (not prev) and began:
                total += resume_overhead_h * float(c)
            began = True
        prev = bool(r)
    return total


# ═══════════════════════════════════════════════════════════════════════
# Planners
# ═══════════════════════════════════════════════════════════════════════

def plan_resumable(costs, remaining, started=False, was_running=False, resume_overhead_h=0.0):
    """Exact minimum-carbon plan for a pausable job.

    Choose `remaining` hours out of the window to run so total carbon (plus
    resume overhead) is minimal. Returns (run_mask, total_cost); if the window
    is too short, returns (None, inf).
    """
    costs = np.asarray(costs, dtype=float)
    W = len(costs)
    r = int(remaining)
    if r <= 0:
        return np.zeros(W, dtype=bool), 0.0
    if r > W:
        return None, math.inf

    inf = math.inf
    f0 = np.full(r + 1, inf)   # state: idle at the end of the previous hour
    f1 = np.full(r + 1, inf)   # state: running at the end of the previous hour
    if was_running:
        f1[0] = 0.0
    else:
        f0[0] = 0.0

    # back-pointers: was the previous hour a "run" hour in the best path?
    bp0 = np.zeros((W, r + 1), dtype=bool)
    bp1 = np.zeros((W, r + 1), dtype=bool)
    pen = 1.0 + resume_overhead_h
    began_before = np.arange(r + 1) > 0     # done > 0 within this plan
    if started:
        began_before = np.ones(r + 1, dtype=bool)

    for h in range(W):
        c = costs[h]
        # idle this hour
        n0 = np.minimum(f0, f1)
        bp0[h] = f1 < f0
        # run this hour: from state 1 (keep running) or state 0 (resume / first start)
        from_run = f1[:-1] + c
        resume_cost = np.where(began_before[:-1], c * pen, c)
        from_idle = f0[:-1] + resume_cost
        n1 = np.full(r + 1, inf)
        n1[1:] = np.minimum(from_run, from_idle)
        bp1[h, 1:] = from_run < from_idle
        f0, f1 = n0, n1

    best_state = 1 if f1[r] < f0[r] else 0
    best = float(min(f0[r], f1[r]))
    if not math.isfinite(best):
        return None, math.inf

    mask = np.zeros(W, dtype=bool)
    s, j = best_state, r
    for h in range(W - 1, -1, -1):
        if s == 1:
            mask[h] = True
            prev = int(bp1[h, j])
            j -= 1
            s = prev
        else:
            s = int(bp0[h, j])
    return mask, best


def plan_contiguous(costs, duration):
    """Best single uninterrupted block of `duration` hours. Returns (start, cost)."""
    costs = np.asarray(costs, dtype=float)
    if duration > len(costs):
        return None, math.inf
    csum = np.concatenate([[0.0], np.cumsum(costs)])
    window = csum[duration:] - csum[:-duration]
    best = float(window.min())
    # Ties (including float round-off between equal windows) go to the EARLIEST
    # start: with no information that separates two windows, don't wait.
    tol = 1e-9 * max(1.0, abs(best))
    start = int(np.flatnonzero(window <= best + tol)[0])
    return start, float(window[start])


# ═══════════════════════════════════════════════════════════════════════
# Online decision (same shape as scheduler.decision_engine.decide())
# ═══════════════════════════════════════════════════════════════════════

class DeadlineAwareScheduler:
    def __init__(self, forecaster="profile_anomaly", max_horizon_h=168):
        if forecaster not in FORECASTERS:
            raise ValueError(f"Unknown forecaster '{forecaster}' — expected one of {list(FORECASTERS)}")
        self.forecaster_name = forecaster
        self.forecaster = FORECASTERS[forecaster]
        self.max_horizon_h = max_horizon_h

    def decide(self, history, job, hours_done, hours_elapsed, was_running):
        """
        history        carbon readings up to and including the current hour
        hours_done     compute-hours already completed
        hours_elapsed  hours since submission (0 = the submission hour)
        was_running    did training run during the previous hour?
        """
        remaining = job.duration_h - hours_done
        if remaining <= 0:
            return self._result("DONE", "Job already complete", 0, False, False)

        slack_window = job.deadline_h - hours_elapsed      # hours left incl. this one
        if slack_window <= remaining:
            why = ("no slack left" if slack_window == remaining
                   else "deadline already unreachable — finish as soon as possible")
            return self._result("TRAIN", f"Forced: {why} "
                                f"({remaining}h of work, {slack_window}h until deadline)",
                                0, False, True)

        W = min(slack_window, max(self.max_horizon_h, remaining))
        current = float(history[-1])
        costs = np.concatenate([[current], self.forecaster(history, W - 1)])
        started = hours_done > 0
        mask, cost = plan_resumable(costs, remaining, started, was_running,
                                    job.resume_overhead_h)
        if mask is None:     # cannot happen when slack_window > remaining, but stay safe
            return self._result("TRAIN", "Planner infeasible — training now", 0, False, True)

        if mask[0]:
            return self._result(
                "TRAIN",
                f"Current hour ({current:.0f} gCO2/kWh) is in the cheapest plan "
                f"for {remaining}h of work with {slack_window}h to the deadline "
                f"({self.forecaster_name} forecast)",
                0, True, False, cost)
        nxt = int(np.argmax(mask))
        return self._result(
            "WAIT",
            f"Plan starts in {nxt}h: cleaner hours ahead "
            f"({slack_window - remaining}h of slack, {self.forecaster_name} forecast)",
            nxt, True, False, cost)

    @staticmethod
    def _result(decision, reason, wait_hours, used_forecast, forced, expected=None):
        return {
            "decision": decision,
            "reason": reason,
            "wait_hours": wait_hours,
            "used_forecast": used_forecast,
            "forced": forced,
            "expected_gco2_per_kw": expected,
        }
