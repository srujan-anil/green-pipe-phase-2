import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/DE_historical.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)

df = df.dropna(subset=["carbon_intensity"]).tail(720).reset_index(drop=True)

start_hour = 3
end_hour = 103

window = df.iloc[start_hour:end_hour].reset_index(drop=True)

sim_hours = list(range(start_hour, start_hour + len(window)))

carbon = window["carbon_intensity"].values

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(
    sim_hours,
    carbon,
    color="black",
    linewidth=1.4,
    label="Carbon intensity (DE)"
)

ax.axhline(
    300,
    color="gray",
    linestyle="--",
    linewidth=1.1,
    label="High threshold (300 gCO2/kWh)"
)

pause_start, pause_end = 6, 16

ax.axvspan(
    pause_start,
    pause_end,
    color="#cccccc",
    alpha=0.5,
    label="Paused / migrated region"
)

ax.set_xlabel("Simulated hour")
ax.set_ylabel("Carbon intensity (gCO2/kWh)")

ax.set_xlim(start_hour, min(sim_hours[-1], start_hour + 100))

# Clean, subtle gridlines
ax.grid(
    axis="both",
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

# Legend above the plotting area
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=9,
    handlelength=2.5,
    columnspacing=1.5
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "dirty_event_figure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "dirty_event_figure.pdf",
    bbox_inches="tight"
)

print("Saved dirty_event_figure.png and .pdf")