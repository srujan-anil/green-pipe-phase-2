# Green-Pipe — Frozen Baseline (pre-redesign)

Recorded before any code was modified in this pass. Repo state: single commit
`b6cc147` on `github.com/ganaviyd/green-pipe`. This file is historical record —
do not edit after the fact; append a new dated section if the baseline is
re-verified later.

## What is real and verified in the original repo

- `scheduler/carbon_client.py` — live Electricity Maps `/latest` call. Real.
- `scheduler/decision_engine.py` — `get_decision`, `get_adaptive_batch_size`
  work on **current** carbon only. `get_best_window`, `get_best_region`
  implemented but **not called** by the orchestrator (dead code).
- `scheduler/orchestrator.py` — real training loop, real `torch.save`/`load`
  checkpointing, current-carbon-only decisions. No forecast, no timing on
  checkpoint I/O, no CodeCarbon.
- `forecasting/arima_model.py`, `forecasting/lstm_model.py` — real, trained,
  produce genuine MAPE/RMSE (verified against `results/arima_*_metrics.txt`,
  `results/lstm_*_metrics.txt`, values match markdown). **Not connected** to
  the orchestrator. ARIMA model object was never persisted to disk. LSTM
  weights were persisted (`forecasting/best_lstm_{zone}.pt`) but the
  `MinMaxScaler` was not, so the saved weights alone can't produce a
  correctly-scaled forecast.
- `results/orchestrator_log.json` — one real logged run, carbon=120 constant,
  losses 0.1854 → 0.0219. This is the only verified live-orchestrator
  evidence in the repo. (Differs from numbers quoted in the historical
  markdown — flagged as a conflict in the prior audit turn, not reconciled.)

## What is NOT in the repo (confirmed by direct inspection, not assumption)

- `experiments/` directory does not exist. No `experiment_runner.py`, no
  `rerun_outliers.py`.
- `codecarbon` and `mlflow` are listed in `requirements.txt` but never
  imported anywhere in the codebase.
- `data/raw/*.csv`, `data/processed/*.csv` are gitignored and absent from
  the clone — no historical carbon time series survive in the repo.
- `checkpoints/*.pt` gitignored and absent.
- `.env` / API key absent from this environment.
- Live network access to `api.electricitymap.org` is blocked from this
  sandbox (`403 host_not_allowed`, confirmed by direct `curl`, not inferred).

## Consequence for this redesign pass

Given the above, this pass can rebuild and unit-verify the **architecture**
(forecast integration, decision logic, checkpoint timing, CodeCarbon wiring,
experiment scaffolding) but **cannot execute real experiments** requiring
carbon time series or live API access in this environment. Real experiment
runs happen on the user's machine once `.env` and `data/raw/*.csv` are
restored locally — see `RUNBOOK.md`.

Any number in this pass that is not traceable to an actual execution in
this sandbox is labeled `[SYNTHETIC — DEV FIXTURE ONLY, NOT A RESULT]` and
must never be copied into `results/`.