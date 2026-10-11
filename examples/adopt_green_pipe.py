"""
======================================================================
 GREEN-PIPE ADOPTION EXAMPLE
======================================================================
 This file answers the real question: "How does a company actually
 use Green-Pipe?"

 ANSWER: They do NOT open a dashboard and click a button every time
 they want to train. They import two things from Green-Pipe into
 THEIR OWN existing training script:

   1. scheduler.decision_engine   -> tells them TRAIN or PAUSE
   2. a checkpoint save/load pattern -> lets them pause without
      losing progress, and resume later automatically

 Below is a BEFORE / AFTER comparison using a generic training loop
 (could be BERT, ResNet, or any PyTorch model a company already has).
 Only the lines marked with "# >>> GREEN-PIPE" are new. Everything
 else is the company's own existing code, untouched.
======================================================================
"""

# ----------------------------------------------------------------------
# BEFORE: a normal company training script (no carbon awareness)
# ----------------------------------------------------------------------
"""
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done, loss={loss.item():.4f}")
"""

# ----------------------------------------------------------------------
# AFTER: the SAME script, with Green-Pipe wrapped around it
# ----------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch

# >>> GREEN-PIPE: these two imports are the entire integration surface.
# Nothing else from Green-Pipe needs to be installed or understood.
from scheduler.decision_engine import get_decision, get_adaptive_batch_size
from scheduler.carbon_client import fetch_live_carbon  # thin wrapper over Electricity Maps API


def green_pipe_checkpoint_save(model, optimizer, epoch, carbon, path="checkpoints/company_job.pt"):
    """
    >>> GREEN-PIPE: drop-in checkpoint save. A company replaces
    `path` with wherever they already store checkpoints (S3, local
    disk, etc). The structure mirrors PyTorch's standard pattern,
    so it fits into any existing training script without changes
    to how the company stores/loads weights elsewhere.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "carbon_at_pause": carbon,
    }, path)


def green_pipe_checkpoint_load(model, optimizer, path="checkpoints/company_job.pt"):
    """>>> GREEN-PIPE: drop-in checkpoint resume."""
    import os
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]


def train_with_green_pipe(model, optimizer, criterion, train_loader_builder,
                           num_epochs, zone="DE"):
    """
    This function has THE SAME SHAPE as any standard PyTorch training
    function. A company takes their existing train() function and
    adds exactly 4 things, marked below. They do not change their
    model, their data loading, their loss function, or their
    optimizer in any way.
    """

    # >>> GREEN-PIPE [1 of 4]: resume from checkpoint if one exists
    start_epoch = green_pipe_checkpoint_load(model, optimizer)

    epoch = start_epoch
    while epoch < num_epochs:

        # >>> GREEN-PIPE [2 of 4]: ask the grid before training this epoch
        carbon, _source = fetch_live_carbon(zone)
        decision, reason = get_decision(carbon)

        if decision == "PAUSE":
            # >>> GREEN-PIPE [3 of 4]: save progress, wait, re-check later
            green_pipe_checkpoint_save(model, optimizer, epoch, carbon)
            print(f"[Green-Pipe] PAUSE at epoch {epoch} — {reason}. "
                  f"Checkpoint saved. Re-checking grid in 60s.")
            time.sleep(60)
            continue

        # >>> GREEN-PIPE [4 of 4]: adjust batch size to current grid cleanliness
        batch_size, _ = get_adaptive_batch_size(carbon)
        train_loader = train_loader_builder(batch_size=batch_size)

        # ---------------------------------------------------------
        # EVERYTHING BELOW THIS LINE IS THE COMPANY'S OWN, UNCHANGED
        # TRAINING LOOP. Green-Pipe does not touch model internals,
        # gradients, the loss function, or the optimizer step.
        # ---------------------------------------------------------
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete — loss={avg_loss:.4f} "
              f"— carbon={carbon:.0f}g — batch={batch_size} ({reason})")

        epoch += 1
        green_pipe_checkpoint_save(model, optimizer, epoch, carbon)

    print("[Green-Pipe] Training complete.")


# ======================================================================
# WHAT THE COMPANY ACTUALLY RUNS — this is the ONLY file they execute.
# Note this could just as easily be their BERT fine-tuning script or
# ResNet training script; nothing about Green-Pipe assumes MNIST.
# ======================================================================
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # --- This block is 100% the company's own model/data setup ---
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    class CompanyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.fc = nn.Sequential(nn.Linear(16 * 14 * 14, 128), nn.ReLU(), nn.Linear(128, 10))

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = CompanyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def build_loader(batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- This single call is the company's entire "adoption" of Green-Pipe ---
    train_with_green_pipe(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader_builder=build_loader,
        num_epochs=5,
        zone="DE",
    )