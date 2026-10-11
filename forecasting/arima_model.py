import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

MODEL_PATH_TEMPLATE = "forecasting/arima_{zone}_model.pkl"

def load_data(zone):
    path = f"data/processed/{zone}_historical.csv"
    if not os.path.exists(path):
        print(f" Processed data not found for {zone}")
        return None

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=['carbon_intensity'])
    df = df.tail(720).reset_index(drop=True)  # last 30 days
    print(f" Loaded {len(df)} records for {zone}")
    return df

def train_arima(zone="DE"):
    # ── Load Data ─────────────────────────────────────────
    df = load_data(zone)
    if df is None:
        return

    # ── Split 80% train 20% test ──────────────────────────
    split = int(len(df) * 0.8)
    train = df['carbon_intensity'][:split]
    test  = df['carbon_intensity'][split:]

    print(f"\n Training ARIMA on {zone}...")
    print(f"   Train size: {len(train)} | Test size: {len(test)}")

    # ── Auto ARIMA ────────────────────────────────────────
    model = auto_arima(
        train,
        seasonal=False,
        suppress_warnings=True,
        error_action='ignore',
        stepwise=True,
        max_p=3,
        max_q=3,
        max_d=2
    )

    print(f" Best ARIMA order: {model.order}")

    # ── Predict ───────────────────────────────────────────
    predictions = model.predict(n_periods=len(test))

    # ── Evaluate ──────────────────────────────────────────
    mape = mean_absolute_percentage_error(test, predictions) * 100
    rmse = np.sqrt(mean_squared_error(test, predictions))

    print(f"\n ARIMA Results for {zone}:")
    print(f"   MAPE : {mape:.2f}%")
    print(f"   RMSE : {rmse:.2f} gCO2/kWh")

    # ── Save Metrics ──────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open(f"results/arima_{zone}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone: {zone}\n")
        f.write(f"ARIMA Order: {model.order}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"RMSE: {rmse:.2f} gCO2/kWh\n")
        f.write(f"Train size: {len(train)}\n")
        f.write(f"Test size: {len(test)}\n")

    # ── Plot ──────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(test)), test.values,
             label="Actual", color="blue")
    plt.plot(range(len(test)), predictions,
             label="ARIMA Predicted", color="orange", linestyle="--")
    plt.title(f"ARIMA Forecast vs Actual — {zone}")
    plt.xlabel("Hours")
    plt.ylabel("Carbon Intensity (gCO2/kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/arima_{zone}_forecast.png")
    plt.close()
    print(f" Graph saved to results/arima_{zone}_forecast.png")

    # ── Persist model (original repo never saved this — orchestrator
    #    had nothing to load) ────────────────────────────────────────
    model_path = MODEL_PATH_TEMPLATE.format(zone=zone)
    joblib.dump(model, model_path)
    print(f" Model persisted to {model_path}")

    return model, mape, rmse


def load_arima_model(zone):
    """Load a previously trained+persisted ARIMA model. Returns None if
    no trained model exists — callers must handle this, not assume it."""
    path = MODEL_PATH_TEMPLATE.format(zone=zone)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def forecast_next(zone, n_periods=24):
    """
    Forecast the next n_periods hours of carbon intensity for `zone`
    using the persisted ARIMA model. Returns None if no model has been
    trained for this zone (caller must fall back, not fabricate values).
    """
    model = load_arima_model(zone)
    if model is None:
        return None
    predictions = model.predict(n_periods=n_periods)
    return list(np.asarray(predictions, dtype=float))


if __name__ == "__main__":
    for zone in ["DE", "IN", "US-CAL-CISO"]:
        print(f"\n{'='*40}")
        train_arima(zone)