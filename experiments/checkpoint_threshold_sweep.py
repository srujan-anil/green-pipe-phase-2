import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import config
from experiments.models import MODEL_TIERS
from training.checkpoint_utils import save_checkpoint, load_latest_checkpoint, clear_checkpoints


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_tier(model_name, model_class, seed, epochs, batch_size, checkpoint_root):
    torch.manual_seed(seed)
    checkpoint_dir = os.path.join(checkpoint_root, model_name)
    clear_checkpoints(checkpoint_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []
    save_times = []
    restore_times = []
    checkpoint_size_bytes = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)

        path, save_time = save_checkpoint(model, optimizer, epoch, carbon=None, batch_size=batch_size, checkpoint_dir=checkpoint_dir)
        save_times.append(save_time)
        checkpoint_size_bytes = os.path.getsize(path)

        _, restore_time, _ = load_latest_checkpoint(model, optimizer, checkpoint_dir)
        restore_times.append(restore_time)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_save_time = sum(save_times) / len(save_times)
    avg_restore_time = sum(restore_times) / len(restore_times)

    return {
        "model": model_name,
        "param_count": count_params(model),
        "checkpoint_size_kb": checkpoint_size_bytes / 1024,
        "epochs": epochs,
        "batch_size": batch_size,
        "avg_epoch_time_s": avg_epoch_time,
        "avg_checkpoint_save_s": avg_save_time,
        "avg_checkpoint_restore_s": avg_restore_time,
        "save_pct_of_epoch": (avg_save_time / avg_epoch_time) * 100,
        "restore_pct_of_epoch": (avg_restore_time / avg_epoch_time) * 100,
        "total_overhead_pct_of_epoch": ((avg_save_time + avg_restore_time) / avg_epoch_time) * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_root = os.path.join(config.CHECKPOINT_DIR, "_threshold_sweep")
    out_dir = os.path.join(config.RESULTS_DIR, "experiments", "checkpoint_threshold")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for model_name, model_class in MODEL_TIERS.items():
        print(f"\nRunning {model_name}...")
        result = run_tier(model_name, model_class, args.seed, args.epochs, args.batch_size, checkpoint_root)
        results.append(result)
        print(json.dumps(result, indent=2))

    out_path = os.path.join(out_dir, f"sweep_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Model':<12}{'Params':<12}{'Ckpt KB':<10}{'Epoch s':<10}{'Save s':<10}{'Restore s':<11}{'Overhead %'}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<12}{r['param_count']:<12}{r['checkpoint_size_kb']:<10.1f}{r['avg_epoch_time_s']:<10.2f}{r['avg_checkpoint_save_s']:<10.4f}{r['avg_checkpoint_restore_s']:<11.4f}{r['total_overhead_pct_of_epoch']:.2f}%")

    print(f"\nWritten to: {out_path}")