import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler import orchestrator as orch
from scheduler import config

FIXTURES = [
    (150, None),
    (250, [240, 230, 180, 170, 190, 210]),
    (250, [245, 248, 250, 252]),
    (350, [100, 90, 80]),
    (150, None),
]

_call_index = {"i": 0}


def fake_get_current_carbon(zone):
    carbon, _ = FIXTURES[_call_index["i"]]
    return carbon, "mocked"


def fake_get_zone_forecast(zone):
    _, forecast = FIXTURES[_call_index["i"]]
    _call_index["i"] += 1
    model = "mocked_arima" if forecast else None
    return forecast, model


orch.get_current_carbon = fake_get_current_carbon
orch.get_zone_forecast = fake_get_zone_forecast
orch.time.sleep = lambda s: None

checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "_dryrun")
log_path = os.path.join(config.RESULTS_DIR, "orchestrator_dryrun_log.json")

orch.clear_checkpoints(checkpoint_dir)

log = orch.run_orchestrator(
    zone="DE",
    num_epochs=3,
    use_forecast=True,
    checkpoint_dir=checkpoint_dir,
    log_path=log_path,
    seed=42,
)

decisions_seen = {entry["decision"] for entry in log}
print(f"\nDecisions exercised: {decisions_seen}")
assert "TRAIN" in decisions_seen
assert "WAIT" in decisions_seen
assert "PAUSE" in decisions_seen
print("Dry run covered TRAIN, WAIT, and PAUSE paths.")