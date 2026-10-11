# Deadline-aware carbon scheduling

A scheduler that decides, hour by hour, whether a training job should **train or wait**, so that it
finishes *by its deadline* while using the cleanest hours it can. It complements the existing
`TRAIN / WAIT / PAUSE` rules in `scheduler/decision_engine.py`, which have no notion of a deadline.

## What was added

| file | purpose |
|---|---|
| `scheduler/deadline_scheduler.py` | the scheduler: `Job`, causal forecasters, exact planner, `DeadlineAwareScheduler.decide()` |
| `experiments/schedule_sim.py` | replays a full year of real hourly carbon data and compares strategies on identical jobs |
| `experiments/forecast_accuracy.py` | causal forecast error by horizon |
| `experiments/forecaster_sensitivity.py` | how the forecaster choice changes the results |
| `tests/test_deadline_scheduler.py` | 28 tests (standard library `unittest`) |
| `results/experiments/schedule_sim/` | generated reports and raw summaries |

Nothing existing was modified. The module has no torch dependency.

## How it works

1. A job is `Job(duration_h, deadline_h)`: compute-hours needed and the hour count from submission by which it must be finished.
2. Every hour the scheduler takes the live carbon reading, forecasts the rest of the window from the history it has seen (never the future), and computes the **exact** cheapest set of hours that still completes the job on time (dynamic programming). It trains now if the current hour is in that plan, otherwise it waits, and it re-plans next hour.
3. The deadline is a hard constraint: once slack reaches zero it always trains.
4. Resuming from a checkpoint costs extra energy (default 5% of an hour of compute, an assumption, see below), and the planner prices that in so it does not pause needlessly.

```python
from scheduler.deadline_scheduler import DeadlineAwareScheduler, Job

sched = DeadlineAwareScheduler()                       # forecaster="profile_anomaly"
job = Job(duration_h=8, deadline_h=48)
result = sched.decide(history=carbon_readings_so_far,  # last element = current hour
                      job=job, hours_done=3, hours_elapsed=10, was_running=True)
result["decision"]    # "TRAIN" | "WAIT" | "DONE"      (same shape as decision_engine.decide)
result["reason"], result["wait_hours"], result["forced"]
```

## Run it

```bash
python -m unittest discover -s tests -v                       # tests
python -m experiments.forecast_accuracy                       # forecast error table
python -m experiments.schedule_sim                            # ~5 min; main experiment
python -m experiments.schedule_sim --forecaster profile_7d    # repeat for each forecaster
python -m experiments.forecaster_sensitivity                  # side-by-side table
```

## Results (real 2025 hourly data for DE, IN, US-CAL-CISO)

One job submitted every 11 hours across the year (so every hour of the day is covered), about 760 to 790 jobs per
cell. Every strategy sees the same jobs; **energy per job is identical**, so this measures only what
scheduling controls. Reduction is the paired drop in gCO2 versus running immediately
(`run_now`); brackets are 95% intervals from a 7-day block bootstrap.

| zone | job (work / deadline) | repo rule `threshold_abs` | sensible baseline `threshold_rel` | **`deadline_aware`** | oracle (sees the future) | mean extra delay |
|---|---|---|---|---|---|---|
| DE | 8 h / 10 h | +8.1% [+6.4, +10.1] (28% on time) | +1.5% [+1.1, +1.9] | **+3.5% [+2.9, +4.1]** | +4.4% [+3.9, +5.0] | 1 h |
| DE | 8 h / 24 h | +14.2% [+11.5, +17.0] (55% on time) | +12.8% [+10.0, +15.9] | **+20.4% [+17.0, +24.2]** | +23.6% [+20.8, +26.7] | 10 h |
| DE | 8 h / 48 h | +18.1% [+15.0, +21.3] (73% on time) | +15.4% [+12.6, +18.3] | **+24.9% [+20.9, +29.2]** | +31.9% [+29.1, +35.0] | 30 h |
| DE | 24 h / 168 h | +25.9% [+22.0, +29.7] (83% on time) | +19.5% [+17.1, +22.4] | **+31.7% [+27.1, +36.7]** | +41.9% [+38.6, +45.3] | 119 h |
| IN | 8 h / 10 h | -1.5% [-1.7, -1.3] (0% on time) | +1.4% [+1.3, +1.5] | **+2.0% [+1.9, +2.2]** | +2.1% [+2.0, +2.2] | 1 h |
| IN | 8 h / 24 h | -0.0% [-0.2, +0.1] (0% on time) | +9.7% [+9.1, +10.2] | **+13.9% [+13.5, +14.3]** | +13.9% [+13.5, +14.3] | 10 h |
| IN | 8 h / 48 h | -0.1% [-0.3, +0.2] (0% on time) | +9.7% [+9.1, +10.2] | **+16.8% [+16.4, +17.2]** | +17.1% [+16.7, +17.5] | 30 h |
| IN | 24 h / 168 h | -0.1% [-1.0, +0.8] (0% on time) | +8.6% [+8.1, +9.2] | **+17.6% [+17.0, +18.1]** | +18.3% [+17.8, +18.8] | 127 h |
| US-CAL-CISO | 8 h / 10 h | +6.6% [+4.7, +8.7] (75% on time) | +4.1% [+3.9, +4.5] | **+6.6% [+6.3, +7.1]** | +6.8% [+6.5, +7.3] | 1 h |
| US-CAL-CISO | 8 h / 24 h | +8.5% [+6.1, +11.3] (99% on time) | +27.3% [+25.6, +29.4] | **+36.9% [+35.2, +38.8]** | +37.2% [+35.6, +39.2] | 10 h |
| US-CAL-CISO | 8 h / 48 h | +8.5% [+6.1, +11.3] (100% on time) | +27.9% [+26.2, +29.9] | **+40.4% [+38.6, +42.2]** | +44.0% [+42.3, +45.7] | 29 h |
| US-CAL-CISO | 24 h / 168 h | +8.4% [+6.0, +11.2] (100% on time) | +30.8% [+28.9, +33.4] | **+49.2% [+46.8, +51.7]** | +53.5% [+51.6, +55.5] | 112 h |

Reading the table:

* `deadline_aware` finished **100% of jobs on time in every cell**. The repo's absolute-threshold rule (pause while
  ≥ 300 gCO2/kWh) misses deadlines often (e.g. DE 8 h / 24 h: 55% on time) and in **IN never trains** within the
  deadline at all, because India's grid is above 300 gCO2/kWh in 100% of 2025 hours.
* `threshold_rel` is a fairer baseline (pause while above the trailing 7-day median, with a deadline guard).
  `deadline_aware` beats it by roughly 4 to 18 percentage points when the deadline is loose (24 h or more of
  slack for these jobs); the intervals mostly do not overlap there.
* With tight deadlines (8 h work in 10 h) there is little to gain for any strategy (about 2 to 7%).
* It captures about 76 to 78% (DE), 96 to 98% (IN) and 92% (US-CAL-CISO) of what a perfect-foresight oracle could save
  on the loose-deadline jobs. DE has the largest gap because its wind-driven forecast error is the largest.
* **The price is delay.** Savings come from using slack, so jobs finish later than if run immediately (last column).
  It never exceeds the deadline, but a job with a week of slack typically takes several days.

Full per-cell tables: `REPORT_profile_anomaly.md`. Forecaster comparison: `SENSITIVITY.md` and
`FORECAST_ACCURACY.md`.

## Limitations

* **A simulation of scheduling, not of training.** Jobs are abstract (constant power, pausable at hour granularity).
  The 5% resume overhead is an assumption; calibrate it from the repo's measured `checkpoint_save_seconds` /
  `checkpoint_restore_seconds`.
* **One year, three zones.** Results may differ in other years and grids. The block-bootstrap intervals capture
  sampling variation within this year only.
* **Forecaster choice.** The default started as `profile_7d`. A quick first run (which turned out to sample only one
  hour of the day) made it look weak, so it was replaced with `profile_anomaly` (daily profile plus a decaying
  deviation of today from that profile, decay estimated online). The corrected, full-day results show the gain is
  modest (average +21.3% vs +19.9%), and every forecaster is reported side by side in `SENSITIVITY.md`.
* **Not yet wired into `orchestrator.py` / `experiments/runner.py`.** Those pass only the current carbon value to a
  decision function, while this scheduler needs history, hours done and a deadline. That integration needs torch to
  test and was not attempted here.
* Carbon intensity is *average* grid intensity as supplied by the data; marginal emissions are not modelled.

## Notes on the existing experiments (found while reading the code)

* The `README.md` "Six-Strategy Comparison" column **"CO2 vs Baseline" lists increases**: Green-Pipe's mean was
  0.307 g versus 0.238 g for baseline (+29.4%), and energy and runtime rose by almost the same amounts
  (+29.5% and +46.9%). With n = 6 per strategy and a standard deviation of about 50% of the mean, those differences
  are not distinguishable from hardware noise. A reader will likely take the positive numbers as savings.
* In the replay, `predictive` and `green_pipe` call `forecast_next(zone)`, which has no time argument, so the same
  forecast is used at every simulated hour. For ARIMA it continues from the end of its training split (hour 576 of
  the 720-hour window being replayed); for the LSTM it starts from the end of that same 720-hour window. The models were also
  trained on data that overlaps the replayed hours. The forecast therefore is not what a scheduler would have known
  at that simulated hour.
