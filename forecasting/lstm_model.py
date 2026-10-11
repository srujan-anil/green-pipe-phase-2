import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

WINDOW = 24
SCALER_PATH_TEMPLATE = "forecasting/lstm_scaler_{zone}.pkl"
WEIGHTS_PATH_TEMPLATE = "forecasting/best_lstm_{zone}.pt"

# ── Reproducibility ───────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

def load_data(zone):
    path = f"data/processed/{zone}_historical.csv"
    if not os.path.exists(path):
        print(f" Processed data not found for {zone}")
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['carbon_intensity'])
    df = df.tail(720).reset_index(drop=True)
    print(f" Loaded {len(df)} records for {zone}")
    return df

# ── LSTM Model ────────────────────────────────────────
class CarbonLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(CarbonLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Create Sequences ──────────────────────────────────
def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def train_lstm(zone="DE"):
    # ── Load Data ─────────────────────────────────────
    df = load_data(zone)
    if df is None:
        return

    values = df['carbon_intensity'].values.reshape(-1, 1)

    # ── Scale Data ────────────────────────────────────
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # ── Create Sequences ──────────────────────────────
    X, y = create_sequences(scaled, window=WINDOW)

    # ── Split 80/20 ───────────────────────────────────
    split = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:split])
    y_train = torch.FloatTensor(y[:split])
    X_test  = torch.FloatTensor(X[split:])
    y_test  = torch.FloatTensor(y[split:])

    print(f"\n Training LSTM on {zone}...")
    print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── Model Setup ───────────────────────────────────
    model     = CarbonLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # ── Early Stopping Setup ──────────────────────────
    best_loss = float('inf')
    patience  = 10
    counter   = 0

    # ── Training Loop ─────────────────────────────────
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss   = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_test)
            val_loss   = criterion(val_output, y_test).item()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH_TEMPLATE.format(zone=zone))
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1} — Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    # ── Load Best Model ───────────────────────────────
    model.load_state_dict(torch.load(WEIGHTS_PATH_TEMPLATE.format(zone=zone)))

    # ── Persist scaler (original repo saved weights but not the
    #    scaler, so a loaded model alone could not produce a
    #    correctly-scaled forecast) ─────────────────────────────
    joblib.dump(scaler, SCALER_PATH_TEMPLATE.format(zone=zone))

    # ── Predict ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test).numpy()

    # ── Inverse Scale ─────────────────────────────────
    predictions = scaler.inverse_transform(predictions_scaled)
    actual      = scaler.inverse_transform(y_test.numpy())

    # ── Evaluate ──────────────────────────────────────
    mape = mean_absolute_percentage_error(actual, predictions) * 100
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    print(f"\n LSTM Results for {zone}:")
    print(f"   MAPE : {mape:.2f}%")
    print(f"   RMSE : {rmse:.2f} gCO2/kWh")

    # ── Save Metrics ──────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open(f"results/lstm_{zone}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone: {zone}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"RMSE: {rmse:.2f} gCO2/kWh\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n")

    # ── Plot ──────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", color="blue")
    plt.plot(predictions, label="LSTM Predicted",
             color="green", linestyle="--")
    plt.title(f"LSTM Forecast vs Actual — {zone}")
    plt.xlabel("Hours")
    plt.ylabel("Carbon Intensity (gCO2/kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/lstm_{zone}_forecast.png")
    plt.close()
    print(f" Graph saved to results/lstm_{zone}_forecast.png")

    return mape, rmse


def forecast_next(zone, n_periods=24):
    """
    Forecast the next n_periods hours of carbon intensity for `zone`
    using the persisted LSTM weights + scaler, autoregressively (each
    predicted value feeds back in as input for the next step).
    Returns None if no trained model/scaler exists for this zone —
    caller must handle this, not fabricate a forecast.
    """
    weights_path = WEIGHTS_PATH_TEMPLATE.format(zone=zone)
    scaler_path = SCALER_PATH_TEMPLATE.format(zone=zone)
    if not (os.path.exists(weights_path) and os.path.exists(scaler_path)):
        return None

    scaler = joblib.load(scaler_path)

    df = load_data(zone)
    if df is None or len(df) < WINDOW:
        return None

    model = CarbonLSTM()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    values = df['carbon_intensity'].values.reshape(-1, 1)
    scaled = scaler.transform(values)
    window = scaled[-WINDOW:].reshape(1, WINDOW, 1)

    preds_scaled = []
    current = window.copy()
    with torch.no_grad():
        for _ in range(n_periods):
            x = torch.FloatTensor(current)
            next_scaled = model(x).numpy()[0, 0]
            preds_scaled.append(next_scaled)
            current = np.roll(current, -1, axis=1)
            current[0, -1, 0] = next_scaled

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return list(preds.astype(float))


if __name__ == "__main__":
    for zone in ["DE", "IN", "US-CAL-CISO"]:
        print(f"\n{'='*40}")
        train_lstm(zone)