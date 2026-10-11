import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config
from scheduler.orchestrator import SimpleCNN
from training.checkpoint_utils import save_checkpoint, load_latest_checkpoint, clear_checkpoints
from experiments.strategies import decide_for
from experiments.emissions_tracker import track_epoch, ENERGY_MEASUREMENT_SOURCE, CO2_CALCULATION_METHOD
from experiments.carbon_timeline import get_timeline_carbon, timeline_length

CARBON_SOURCE_LABEL = "experimental_timeline"


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_experiment(strategy, zone, seed, num_epochs, out_dir=None, checkpoint_root=None,
                    max_sim_hours=None, start_hour=0):
    out_dir = out_dir or os.path.join(config.RESULTS_DIR, "experiments", "raw")
    checkpoint_root = checkpoint_root or os.path.join(config.CHECKPOINT_DIR, "_experiments")
    run_id = f"{strategy}_{zone}_seed{seed}"
    checkpoint_dir = os.path.join(checkpoint_root, run_id)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    clear_checkpoints(checkpoint_dir)

    tl_len = timeline_length(zone)
    if tl_len == 0:
        raise RuntimeError(
            f"No historical timeline data for zone={zone} "
            f"(expected data/processed/{zone}_historical.csv). Cannot run experimental mode."
        )
    hour_cap = min(max_sim_hours, tl_len) if max_sim_hours is not None else tl_len

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    pause_count = 0
    resume_count = 0
    checkpoint_count = 0
    total_checkpoint_save_s = 0.0
    total_checkpoint_restore_s = 0.0
    per_epoch_log = []

    start_epoch, restore_time, ckpt = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    if ckpt:
        resume_count += 1
        total_checkpoint_restore_s += restore_time

    total_co2_kg = 0.0
    total_energy_kwh = 0.0
    final_loss = None
    completion_status = "incomplete"

    start_timestamp = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()

    epoch = start_epoch + 1
    batch_size_used = None
    sim_hour = start_hour
    timeline_exhausted = False

    while epoch <= num_epochs:
        if sim_hour >= start_hour + hour_cap:
            timeline_exhausted = True
            break

        carbon, real_timestamp = get_timeline_carbon(zone, sim_hour)
        if carbon is None:
            timeline_exhausted = True
            break

        decision_result = decide_for(strategy, carbon, zone, sim_hour=sim_hour)
        decision = decision_result["decision"]
        batch_size = decision_result["batch_size"]
        batch_size_used = batch_size
        effective_zone = decision_result.get("effective_zone", zone)
        effective_carbon = decision_result.get("effective_carbon", carbon)
        reason = decision_result.get("reason", "")

        if decision in ("PAUSE", "WAIT"):
            if decision == "PAUSE":
                pause_count += 1
                path, save_time = save_checkpoint(model, optimizer, epoch, effective_carbon, batch_size, checkpoint_dir)
                checkpoint_count += 1
                total_checkpoint_save_s += save_time

            per_epoch_log.append({
                "epoch": epoch,
                "sim_hour": sim_hour,
                "real_historical_timestamp": real_timestamp,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "region": effective_zone,
                "carbon_intensity_g_per_kwh": effective_carbon,
                "carbon_source": CARBON_SOURCE_LABEL,
                "decision": decision,
                "reason": reason,
                "energy_kwh": 0.0,
                "co2_kg": 0.0,
            })

            sim_hour += 1
            continue

        def train_one_epoch():
            shuffle_gen = torch.Generator()
            shuffle_gen.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=shuffle_gen)
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_loader)

        avg_loss, epoch_metrics = track_epoch(
            train_one_epoch, effective_zone, effective_carbon, CARBON_SOURCE_LABEL, epoch,
            os.path.join(config.EMISSIONS_DIR, run_id), run_id
        )
        total_co2_kg += epoch_metrics["co2_kg"]
        total_energy_kwh += epoch_metrics["energy_kwh"]
        final_loss = avg_loss

        path, save_time = save_checkpoint(model, optimizer, epoch, effective_carbon, batch_size, checkpoint_dir)
        checkpoint_count += 1
        total_checkpoint_save_s += save_time

        epoch_metrics["sim_hour"] = sim_hour
        epoch_metrics["real_historical_timestamp"] = real_timestamp
        epoch_metrics["decision"] = decision
        epoch_metrics["reason"] = reason
        epoch_metrics["checkpoint_save_s"] = save_time
        per_epoch_log.append(epoch_metrics)

        sim_hour += 1
        epoch += 1

    wall_runtime_s = time.perf_counter() - wall_start
    end_timestamp = datetime.now(timezone.utc).isoformat()

    if epoch > num_epochs:
        completion_status = "complete"
    elif timeline_exhausted:
        completion_status = "incomplete_timeline_exhausted"

    accuracy = evaluate(model, test_loader, device) if completion_status == "complete" else None

    regions_seen = {e["region"] for e in per_epoch_log}
    migrated_during_run = len(regions_seen) > 1

    result = {
        "strategy": strategy,
        "seed": seed,
        "region": zone,
        "migrated_during_run": migrated_during_run,
        "regions_seen": sorted(regions_seen),
        "experimental_mode": True,
        "carbon_data_source": "real historical CSV, replayed deterministically — NOT live",
        "start_hour": start_hour,
        "final_sim_hour": sim_hour,
        "model": "SimpleCNN",
        "batch_size": batch_size_used,
        "epochs": num_epochs,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "accuracy": accuracy,
        "final_loss": final_loss,
        "runtime_seconds": wall_runtime_s,
        "energy_kwh": total_energy_kwh,
        "co2_kg": total_co2_kg,
        "pause_count": pause_count,
        "resume_count": resume_count,
        "checkpoint_count": checkpoint_count,
        "checkpoint_save_seconds": total_checkpoint_save_s,
        "checkpoint_restore_seconds": total_checkpoint_restore_s,
        "completion_status": completion_status,
        "energy_measurement_source": ENERGY_MEASUREMENT_SOURCE,
        "co2_calculation_method": CO2_CALCULATION_METHOD,
        "carbon_intensity_source": CARBON_SOURCE_LABEL,
        "per_epoch": per_epoch_log,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--zone", default="DE")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-sim-hours", type=int, default=None)
    parser.add_argument("--start-hour", type=int, default=0)
    args = parser.parse_args()

    result, out_path = run_experiment(
        args.strategy, args.zone, args.seed, args.epochs,
        max_sim_hours=args.max_sim_hours, start_hour=args.start_hour
    )

    print(json.dumps(result, indent=2))
    print(f"\nWritten to: {out_path}")