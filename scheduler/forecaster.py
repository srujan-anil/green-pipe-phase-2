
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting import arima_model, lstm_model
from scheduler import config


def get_forecast(zone, model="arima", n_periods=None):
   
    if n_periods is None:
        n_periods = config.FORECAST_HORIZON_HOURS

    if model == "arima":
        return arima_model.forecast_next(zone, n_periods=n_periods)
    elif model == "lstm":
        return lstm_model.forecast_next(zone, n_periods=n_periods)
    else:
        raise ValueError(f"Unknown forecast model '{model}' — expected 'arima' or 'lstm'")


def get_best_forecaster(zone):
    
    def read_mape(path):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MAPE:"):
                    return float(line.split(":")[1].strip().rstrip("%"))
        return None

    arima_mape = read_mape(os.path.join(config.RESULTS_DIR, f"arima_{zone}_metrics.txt"))
    lstm_mape = read_mape(os.path.join(config.RESULTS_DIR, f"lstm_{zone}_metrics.txt"))

    if arima_mape is None and lstm_mape is None:
        return None
    if arima_mape is None:
        return "lstm"
    if lstm_mape is None:
        return "arima"
    return "arima" if arima_mape <= lstm_mape else "lstm"