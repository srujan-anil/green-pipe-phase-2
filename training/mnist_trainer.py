import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ── Reproducibility ──────────────────────────────────────
torch.manual_seed(42)

# ── Device ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ── Model ─────────────────────────────────────────────────
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
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ── Checkpoint Functions ──────────────────────────────────
def save_checkpoint(epoch, loss):
    os.makedirs('checkpoints', exist_ok=True)
    path = f'checkpoints/epoch_{epoch}.pt'
    torch.save({
        'arch_version': 'v1',
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f" Checkpoint saved: {path}")

def load_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['arch_version'] != 'v1':
        print(" Architecture mismatch. Starting fresh.")
        return 0
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f" Resumed from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

# ── Training Loop ─────────────────────────────────────────
def train(num_epochs=5):
    print(f"\n Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] — Loss: {avg_loss:.4f}")

        # Save checkpoint after every epoch
        save_checkpoint(epoch, avg_loss)

    print("\n Training complete. All checkpoints saved in /checkpoints folder.")

train()