# Green-Pipe

**A Carbon-Aware Machine Learning Training Orchestrator**

Green-Pipe is a carbon-aware ML training system that combines carbon-intensity data, forecasting, regional carbon comparison, scheduling decisions, checkpointing, and real energy/CO2 measurement.

## Technologies

* **Language:** Python
* **Machine Learning:** PyTorch, CNN, MNIST
* **Forecasting:** ARIMA, LSTM
* **Carbon Data:** Electricity Maps
* **Energy Measurement:** CodeCarbon
* **Dashboard:** Streamlit
* **System Monitoring:** psutil

## What It Does

Green-Pipe monitors carbon intensity before and during ML training and makes a rule-based decision to:

* **TRAIN** when conditions are favorable
* **WAIT** when cleaner conditions are expected
* **PAUSE** during high-carbon periods
* **SWITCH REGION** when another simulated region has lower carbon intensity

The system uses **ARIMA/LSTM forecasting**, **PyTorch**, **MNIST**, and **CodeCarbon** for energy and CO2 measurement.

## Pipeline

```text
Carbon Data
    ↓
ARIMA / LSTM Forecast
    ↓
Region Comparison
    ↓
TRAIN / WAIT / PAUSE Decision
    ↓
PyTorch Training
    ↓
Checkpointing
    ↓
Energy & CO2 Measurement
```

> Multi-region execution is simulated for evaluation; no physical cloud migration is performed.

## Prerequisites

Before installing Green-Pipe, make sure you have:

* **Python 3.10 or newer**
* **Git**
* **pip**
* An **Electricity Maps API key** for live carbon-intensity data
* A **CPU-capable machine** for running the training experiments

Check your installations:

```bash
python --version
git --version
pip --version
```

> A GPU is **not required**. The experiments were designed to run on CPU-based systems.

## Setup

Clone the repository:

```bash
git clone https://github.com/ganaviyd/green-pipe.git
cd green-pipe
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install psutil
```

Create a `.env` file in the project root:

```env
ELECTRICITY_MAPS_KEY=your_electricity_maps_api_key
```

## Run the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard provides:

* Live carbon intensity and scheduling decisions
* Historical high-carbon event replay
* Experiment run inspection
* Six-strategy comparison
* Forecasting results
* Checkpoint-overhead results

## Run Experiments

### Run a Strategy

```bash
python -m experiments.runner --strategy baseline --zone DE --seed 42 --epochs 5 --start-hour 3 --max-sim-hours 100
```

### Aggregate Results

```bash
python -m experiments.aggregate
```

### Checkpoint Threshold Sweep

```bash
python -m experiments.checkpoint_threshold_sweep --seed 42 --epochs 3
```

## Key Results

### Six-Strategy Comparison

| Strategy     | CO2 vs Baseline | Accuracy |
| ------------ | --------------- | -------- |
| Baseline     |            0.0% |   98.06% |
| Reactive     |           +3.0% |   98.06% |
| Predictive   |           +2.6% |   98.06% |
| Multi-region |          +12.1% |   98.06% |
| Adaptive     |          +21.2% |   98.24% |
| Green-Pipe   |          +29.4% |   98.24% |

### High-Carbon Event

The scheduling mechanism correctly detected and paused during the high-carbon period and selected a lower-carbon region before resuming in **9/9 trials**.

The measured CO2 change was **−11.3% on average across 6 analyzed trials**, but the result was **statistically inconclusive** because the 95% confidence interval crossed zero.

### Forecasting

| Region      | ARIMA MAPE | LSTM MAPE | Selected |
| ----------- | ---------- | --------- | -------- |
| DE          |     15.38% |    10.79% | LSTM     |
| IN          |      8.29% |     9.45% | ARIMA    |
| US-CAL-CISO |     31.63% |    26.68% | LSTM     |

## Project Structure

```text
green-pipe/
├── scheduler/       # Carbon fetching, forecasting, and decisions
├── training/        # Model training and checkpointing
├── experiments/     # Experiment runners and analysis
├── dashboard/       # Streamlit dashboard
├── data/            # Historical carbon data
└── results/         # Experiment outputs and metrics
```

## Limitations

* Evaluated using MNIST and small-to-moderate CPU-based models.
* Measurements were performed on shared, non-isolated hardware.
* Multi-region execution is simulated rather than actual cloud migration.
* Forecasting models use fixed historical training data and are not retrained live.
