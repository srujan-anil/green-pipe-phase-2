

import os
import time
import torch

ARCH_VERSION = "v1"


def save_checkpoint(model, optimizer, epoch, carbon, batch_size,
                     checkpoint_dir, prefix="checkpoint", extra=None):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pt")

    payload = {
        "arch_version": ARCH_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "carbon_at_pause": carbon,
        "batch_size": batch_size,
    }
    if extra:
        payload.update(extra)

    t0 = time.perf_counter()
    torch.save(payload, path)
    save_time = time.perf_counter() - t0

    return path, save_time


def load_latest_checkpoint(model, optimizer, checkpoint_dir, prefix="checkpoint"):
   
    if not os.path.isdir(checkpoint_dir):
        return 0, 0.0, None

    files = [f for f in os.listdir(checkpoint_dir)
             if f.startswith(f"{prefix}_epoch_") and f.endswith(".pt")]
    if not files:
        return 0, 0.0, None

    def epoch_num(fname):
        try:
            return int(fname.replace(f"{prefix}_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1

    latest = max(files, key=epoch_num)
    path = os.path.join(checkpoint_dir, latest)

    t0 = time.perf_counter()
    ckpt = torch.load(path, map_location="cpu")
    restore_time = time.perf_counter() - t0

    if ckpt.get("arch_version") != ARCH_VERSION:
        raise RuntimeError(
            f"Checkpoint arch_version mismatch: found "
            f"'{ckpt.get('arch_version')}', expected '{ARCH_VERSION}'. "
            f"Refusing to silently resume from an incompatible checkpoint."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt["epoch"], restore_time, ckpt


def clear_checkpoints(checkpoint_dir, prefix="checkpoint"):
   
    if not os.path.isdir(checkpoint_dir):
        return
    for f in os.listdir(checkpoint_dir):
        if f.startswith(f"{prefix}_epoch_") and f.endswith(".pt"):
            os.remove(os.path.join(checkpoint_dir, f))