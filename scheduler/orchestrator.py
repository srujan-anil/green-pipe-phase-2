import json
import time
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config
from scheduler.carbon_client import fetch_live_carbon
from scheduler.decision_engine import decide
from scheduler.forecaster import get_forecast, get_best_forecaster
from training.checkpoint_utils import save_checkpoint, load_latest_checkpoint, clear_checkpoints


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_current_carbon(zone):
    carbon, source = fetch_live_carbon(zone)
    if carbon is not None:
        return carbon, source

    path = config.resolve_zone_csv(zone)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return float(df["carbon_intensity"].iloc[-1]), "historical_fallback"

    return None, "unavailable"


def get_zone_forecast(zone):
    model = get_best_forecaster(zone)
    if model is None:
        return None, None
    forecast = get_forecast(zone, model=model, n_periods=config.FORECAST_HORIZON_HOURS)
    return forecast, model


def write_log(log, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def run_orchestrator(zone="DE", num_epochs=10, use_forecast=True,
                      checkpoint_dir=None, log_path=None, seed=None):
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    log_path = log_path or os.path.join(config.RESULTS_DIR, "orchestrator_log.json")

    if seed is not None:
        torch.manual_seed(seed)

    print(f"\n{'='*50}")
    print(f" Green-Pipe Orchestrator Starting")
    print(f"   Zone: {zone} | Epochs: {num_epochs} | Forecast: {use_forecast}")
    print(f"{'='*50}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_epoch, restore_time, ckpt = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    if ckpt:
        print(f"Resumed from epoch {start_epoch} in {restore_time:.4f}s")

    log = []

    epoch = start_epoch + 1
    while epoch <= num_epochs:
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        carbon, carbon_source = get_current_carbon(zone)
        forecast, forecast_model = (None, None)
        if use_forecast and carbon is not None:
            forecast, forecast_model = get_zone_forecast(zone)

        result = decide(carbon, forecast=forecast, zone=zone)
        decision = result["decision"]
        batch_size = result["batch_size"]

        print(f"Carbon: {carbon} ({carbon_source}) | Forecast model: {forecast_model}")
        print(f"Decision: {decision} | {result['reason']}")

        entry = {
            "epoch": epoch,
            "carbon": carbon,
            "carbon_source": carbon_source,
            "forecast_model": forecast_model,
            "decision": decision,
            "reason": result["reason"],
            "batch_size": batch_size,
            "wait_hours": result["wait_hours"],
            "used_forecast": result["used_forecast"],
        }

        if decision == "PAUSE":
            path, save_time = save_checkpoint(
                model, optimizer, epoch, carbon, batch_size, checkpoint_dir
            )
            entry["checkpoint_save_time"] = save_time
            entry["checkpoint_path"] = path
            entry["loss"] = None
            log.append(entry)
            write_log(log, log_path)
            print(f"Checkpoint saved in {save_time:.4f}s. Waiting before retry.")
            time.sleep(60)
            continue

        if decision == "WAIT":
            entry["loss"] = None
            log.append(entry)
            write_log(log, log_path)
            print(f"Waiting {result['wait_hours']}h forecasted improvement. Retrying shortly.")
            time.sleep(15)
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete — Loss: {avg_loss:.4f}")

        path, save_time = save_checkpoint(
            model, optimizer, epoch, carbon, batch_size, checkpoint_dir
        )
        entry["loss"] = avg_loss
        entry["checkpoint_save_time"] = save_time
        entry["checkpoint_path"] = path
        log.append(entry)
        write_log(log, log_path)

        epoch += 1

    print(f"\n{'='*50}")
    print(f" Green-Pipe Orchestrator Complete")
    print(f"{'='*50}\n")

    print(f"{'Epoch':<8}{'Carbon':<12}{'Batch':<8}{'Loss':<12}{'Decision'}")
    print("-" * 55)
    for e in log:
        loss_str = f"{e['loss']:.4f}" if e["loss"] is not None else "-"
        carbon_str = f"{e['carbon']:.1f}" if e["carbon"] is not None else "-"
        print(f"{e['epoch']:<8}{carbon_str:<12}{e['batch_size']:<8}{loss_str:<12}{e['decision']}")

    return log


if __name__ == "__main__":
    run_orchestrator(zone="DE", num_epochs=5, use_forecast=True, seed=42)