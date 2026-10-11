"""
Tests for the deadline-aware scheduler and the scheduling simulation.

Standard library only:   python -m unittest discover -s tests -v    (from repo root)
"""

import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import deadline_scheduler as ds
from scheduler.deadline_scheduler import (
    DeadlineAwareScheduler, Job, plan_contiguous, plan_resumable, schedule_cost,
)
from experiments import schedule_sim as sim


def brute_force(costs, remaining, started, was_running, pen):
    best = (np.inf, None)
    W = len(costs)
    for combo in itertools.combinations(range(W), remaining):
        mask = np.zeros(W, dtype=bool)
        mask[list(combo)] = True
        c = schedule_cost(costs, mask, started, was_running, pen)
        if c < best[0] - 1e-12:
            best = (c, mask)
    return best


class PlannerTests(unittest.TestCase):
    def test_resumable_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            W = int(rng.integers(1, 11))
            r = int(rng.integers(1, W + 1))
            costs = rng.uniform(50, 600, W)
            started = bool(rng.integers(0, 2))
            was_running = started and bool(rng.integers(0, 2))
            pen = float(rng.choice([0.0, 0.05, 0.5, 2.0]))
            mask, cost = plan_resumable(costs, r, started, was_running, pen)
            bf_cost, _ = brute_force(costs, r, started, was_running, pen)
            self.assertEqual(int(mask.sum()), r)
            self.assertAlmostEqual(cost, bf_cost, places=6)
            # the returned mask must really cost what the planner says
            self.assertAlmostEqual(schedule_cost(costs, mask, started, was_running, pen),
                                   cost, places=6)

    def test_infeasible_window(self):
        mask, cost = plan_resumable([100, 200], 3)
        self.assertIsNone(mask)
        self.assertEqual(cost, np.inf)

    def test_large_overhead_prefers_one_block(self):
        # two cheap hours separated by an expensive one; resuming is very costly
        costs = [100, 400, 100, 400, 400]
        mask, _ = plan_resumable(costs, 2, False, False, resume_overhead_h=10.0)
        runs = int(np.sum(mask[1:] & ~mask[:-1])) + int(mask[0])
        self.assertEqual(runs, 1)

    def test_zero_overhead_picks_cheapest_hours(self):
        costs = np.array([5, 1, 9, 2, 8, 3.0])
        mask, cost = plan_resumable(costs, 3, False, False, 0.0)
        self.assertEqual(sorted(np.flatnonzero(mask)), [1, 3, 5])
        self.assertAlmostEqual(cost, 6.0)

    def test_contiguous(self):
        start, cost = plan_contiguous([9, 1, 1, 9, 9], 2)
        self.assertEqual((start, cost), (1, 2.0))

    def test_contiguous_ties_go_to_earliest_start(self):
        # identical costs, including values whose cumulative sums round differently
        for value in (100.0, 0.1, 237.31, 1e6 / 3):
            start, _ = plan_contiguous(np.full(40, value), 7)
            self.assertEqual(start, 0, value)


class ForecasterTests(unittest.TestCase):
    def test_shapes_and_causality(self):
        rng = np.random.default_rng(2)
        series = rng.uniform(100, 500, 400)
        for name, f in ds.FORECASTERS.items():
            for t in (5, 30, 200, 399):
                hist = series[: t + 1]
                out = f(hist, 50)
                self.assertEqual(out.shape, (50,), name)
                # changing the (non-existent) future cannot matter: recompute on a
                # copy whose later values are scrambled, truncated identically
                scrambled = series.copy()
                scrambled[t + 1:] = -999
                np.testing.assert_allclose(f(scrambled[: t + 1], 50), out)

    def test_persistence(self):
        np.testing.assert_allclose(ds.forecast_persistence([1, 2, 3.0], 4), [3, 3, 3, 3])

    def test_seasonal_naive_repeats_daily_pattern(self):
        day = np.arange(24, dtype=float)
        series = np.tile(day, 3)           # ends at hour-of-day 23
        out = ds.forecast_seasonal_naive(series, 30)
        # next hour after hour-of-day 23 is hour-of-day 0, then 1, 2, ... wrapping daily
        expected = np.array([day[j % 24] for j in range(30)])
        np.testing.assert_allclose(out, expected)

    def test_profile_equals_seasonal_on_periodic_data(self):
        series = np.tile(np.arange(24, dtype=float) * 10, 10)
        np.testing.assert_allclose(ds.forecast_profile(series, 48),
                                   ds.forecast_seasonal_naive(series, 48))

    def test_short_history_falls_back(self):
        for f in ds.FORECASTERS.values():
            out = f(np.array([100.0, 120.0]), 10)
            self.assertTrue(np.all(np.isfinite(out)))

    def test_anomaly_forecaster_equals_profile_on_perfectly_periodic_data(self):
        series = np.tile(np.arange(24, dtype=float) * 10 + 100, 12)
        np.testing.assert_allclose(ds.forecast_profile_anomaly(series, 30),
                                   ds.forecast_profile(series, 30), atol=1e-9)

    def test_anomaly_forecaster_beats_profile_when_days_differ(self):
        # daily shape + a slowly varying day-specific offset (AR(1), ~0.95/h)
        rng = np.random.default_rng(5)
        n = 24 * 40
        shape = np.tile(200 + 100 * np.sin(np.arange(24) / 24 * 2 * np.pi), n // 24)
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = 0.95 * a[i - 1] + rng.normal(0, 12)
        series = shape + a
        err_p, err_a = [], []
        for t in range(24 * 10, n - 4, 5):
            hist = series[: t + 1]
            err_p.append(abs(ds.forecast_profile(hist, 3)[2] - series[t + 3]))
            err_a.append(abs(ds.forecast_profile_anomaly(hist, 3)[2] - series[t + 3]))
        self.assertLess(np.mean(err_a), 0.8 * np.mean(err_p))


class SchedulerTests(unittest.TestCase):
    def test_forced_when_no_slack(self):
        s = DeadlineAwareScheduler()
        job = Job(duration_h=4, deadline_h=4)
        r = s.decide(np.full(200, 500.0), job, hours_done=0, hours_elapsed=0, was_running=False)
        self.assertEqual(r["decision"], "TRAIN")
        self.assertTrue(r["forced"])

    def test_forced_after_deadline(self):
        s = DeadlineAwareScheduler()
        job = Job(duration_h=4, deadline_h=10)
        r = s.decide(np.full(200, 500.0), job, hours_done=1, hours_elapsed=15, was_running=False)
        self.assertEqual((r["decision"], r["forced"]), ("TRAIN", True))

    def test_waits_for_cheaper_hours_when_slack_exists(self):
        # dirty now, a very clean daily dip 6h from now, plenty of slack
        day = np.full(24, 500.0)
        day[8:12] = 50.0
        hist = np.tile(day, 8)[: 24 * 7 + 2]       # current hour-of-day = 1
        s = DeadlineAwareScheduler("profile_7d")
        r = s.decide(hist, Job(duration_h=3, deadline_h=24), 0, 0, False)
        self.assertEqual(r["decision"], "WAIT")
        self.assertGreaterEqual(r["wait_hours"], 1)

    def test_trains_now_when_now_is_cleanest(self):
        day = np.full(24, 500.0)
        day[1:5] = 50.0
        hist = np.tile(day, 8)[: 24 * 7 + 2]       # current hour-of-day = 1 (clean)
        r = DeadlineAwareScheduler("profile_7d").decide(hist, Job(3, 24), 0, 0, False)
        self.assertEqual(r["decision"], "TRAIN")

    def test_done(self):
        r = DeadlineAwareScheduler().decide(np.ones(10), Job(2, 5), 2, 3, True)
        self.assertEqual(r["decision"], "DONE")

    def test_job_validation(self):
        with self.assertRaises(ValueError):
            Job(duration_h=5, deadline_h=4)
        with self.assertRaises(ValueError):
            Job(duration_h=0, deadline_h=4)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        hours = np.arange(24 * 60)
        # daily cycle + weekly drift + noise, like a real grid
        cls.ci = (300 + 120 * np.sin(2 * np.pi * (hours % 24) / 24)
                  + 40 * np.sin(2 * np.pi * hours / (24 * 7))
                  + rng.normal(0, 25, len(hours)))
        cls.ci = np.clip(cls.ci, 40, None)

    def run_all(self, job, stride=7):
        results = {s: [] for s in sim.STRATEGIES}
        for st in sim.valid_starts(len(self.ci), job, stride):
            for name in sim.STRATEGIES:
                hours = sim.STRATEGY_FUNCS[name](self.ci, st, job, "profile_7d")
                results[name].append(sim.account(self.ci, st, job, hours))
        return results

    def test_every_strategy_runs_exactly_duration_hours(self):
        job = Job(5, 20)
        for st in (200, 333):
            for name in sim.STRATEGIES:
                hours = sim.STRATEGY_FUNCS[name](self.ci, st, job, "profile_7d")
                self.assertEqual(len(hours), 5, name)
                self.assertEqual(len(set(hours)), 5, name)
                self.assertGreaterEqual(min(hours), st, name)

    def test_forecasters_only_ever_see_the_past(self):
        """Spy on every history a strategy hands to its forecaster: at hour h it
        may contain exactly ci[:h+1] — never a future reading."""
        seen = []

        def spy(history, horizon):
            seen.append(np.array(history, copy=True))
            return ds.forecast_persistence(history, horizon)

        ds.FORECASTERS["spy"] = spy
        try:
            st = 250
            # deadline_aware re-plans every hour: calls must be at h = st, st+1, ...
            job = Job(4, 24)
            seen.clear()
            hours = sim.deadline_aware(self.ci, st, job, "spy")
            self.assertGreater(len(seen), 0)
            for k, hist in enumerate(seen):
                np.testing.assert_array_equal(hist, self.ci[: st + k + 1])
            # the last call can be no later than the last hour that was actually run
            self.assertLessEqual(len(seen), hours[-1] - st + 1)
            # best_start plans once, at submission time
            seen.clear()
            sim.best_start(self.ci, st, job, "spy")
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(seen[0], self.ci[: st + 1])
        finally:
            del ds.FORECASTERS["spy"]

    def test_non_oracle_decisions_ignore_future_values(self):
        """Altering readings after the decision hour cannot change that decision."""
        job = Job(4, 24)
        st = 250
        altered = self.ci.copy()
        altered[st + 1:] = 1.0                      # absurd future
        self.assertEqual(sim.best_start(self.ci, st, job, "profile_7d"),
                         sim.best_start(altered, st, job, "profile_7d"))
        a = DeadlineAwareScheduler("profile_7d").decide(self.ci[: st + 1], job, 0, 0, False)
        b = DeadlineAwareScheduler("profile_7d").decide(altered[: st + 1], job, 0, 0, False)
        self.assertEqual(a["decision"], b["decision"])

    def test_deadline_aware_never_misses_feasible_deadline(self):
        for job in (Job(4, 6), Job(8, 10), Job(8, 24), Job(24, 168)):
            res = self.run_all(job, stride=23)
            self.assertTrue(all(r["met"] for r in res["deadline_aware"]), job)
            self.assertTrue(all(r["met"] for r in res["oracle"]), job)
            self.assertTrue(all(r["met"] for r in res["best_start"]), job)

    def test_oracle_is_a_lower_bound_among_on_time_schedules(self):
        for job in (Job(4, 12), Job(8, 24), Job(6, 48)):
            res = self.run_all(job, stride=17)
            n = len(res["oracle"])
            for name in sim.STRATEGIES:
                for i in range(n):
                    r = res[name][i]
                    if r["met"]:
                        self.assertLessEqual(res["oracle"][i]["gco2"], r["gco2"] + 1e-6,
                                             f"{name} beat the oracle on job {job}")

    def test_replanning_with_a_perfect_forecast_equals_the_oracle(self):
        """Validates the planner + hourly loop end to end: if the forecast were
        exact, receding-horizon re-planning must reach the oracle's cost."""
        ci = self.ci
        ds.FORECASTERS["truth"] = lambda history, horizon: ci[len(history): len(history) + horizon]
        try:
            for job in (Job(3, 8), Job(6, 24), Job(8, 48, resume_overhead_h=0.2)):
                for st in sim.valid_starts(len(ci), job, 61):
                    a = sim.account(ci, st, job, sim.deadline_aware(ci, st, job, "truth"))
                    b = sim.account(ci, st, job, sim.oracle(ci, st, job, "truth"))
                    self.assertAlmostEqual(a["gco2"], b["gco2"], places=6)
        finally:
            del ds.FORECASTERS["truth"]

    def test_run_now_accounting(self):
        job = Job(3, 10, power_kw=2.0)
        st = 200
        hours = sim.run_now(self.ci, st, job, "profile_7d")
        out = sim.account(self.ci, st, job, hours)
        self.assertAlmostEqual(out["gco2"], 2.0 * self.ci[st:st + 3].sum())
        self.assertEqual((out["finish_h"], out["delay_h"], out["resumes"]), (3, 0, 0))

    def test_resume_overhead_is_charged(self):
        job = Job(2, 10, resume_overhead_h=0.5)
        st = 200
        out = sim.account(self.ci, st, job, [st, st + 3])      # one pause
        expected = self.ci[st] + self.ci[st + 3] * 1.5
        self.assertAlmostEqual(out["gco2"], expected)
        self.assertEqual(out["resumes"], 1)

    def test_threshold_abs_misses_deadline_when_grid_never_clean(self):
        always_dirty = np.full(2000, 650.0)
        job = Job(4, 24)
        hours = sim.threshold_abs(always_dirty, 200, job, "profile_7d")
        out = sim.account(always_dirty, 200, job, hours)
        self.assertFalse(out["met"])
        self.assertEqual(out["finish_h"], job.deadline_h + job.duration_h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
