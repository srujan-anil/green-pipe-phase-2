import pandas as pd
import os

ZONES = {
    "DE": "data/raw/DE_historical.csv",
    "IN": "data/raw/IN_historical.csv",
    "US-CAL-CISO": "data/raw/US-CAL-CISO_historical.csv"
}

def preprocess(zone, filepath):
    processed_path = f"data/processed/{zone}_historical.csv"
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    print(f"\n Processing {zone}...")
    print(f"   Raw records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    # ── Rename columns ────────────────────────────────────
    df = df.rename(columns={
        'Datetime (UTC)': 'timestamp',
        'Carbon intensity gCO\u2082eq/kWh (Life cycle)': 'carbon_intensity'
    })

    # ── Keep only what we need ────────────────────────────
    df = df[['timestamp', 'carbon_intensity']]

    # ── Normalize timestamps ──────────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # ── Sort by time ──────────────────────────────────────
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── Drop duplicates ───────────────────────────────────
    df = df.drop_duplicates(subset='timestamp')

    # ── Set index and resample hourly ─────────────────────
    df = df.set_index('timestamp')
    df = df.resample('1h').mean(numeric_only=True)

    # ── Interpolate gaps up to 3 hours ────────────────────
    df['carbon_intensity'] = df['carbon_intensity'].interpolate(
        method='time', limit=3
    )

    # ── Drop remaining NaN ────────────────────────────────
    df = df.dropna(subset=['carbon_intensity'])
    df = df.reset_index()

    # ── Save ──────────────────────────────────────────────
    df.to_csv(processed_path, index=False)

    print(f"   Processed records: {len(df)}")
    print(f"   Carbon range: {df['carbon_intensity'].min():.1f} — {df['carbon_intensity'].max():.1f} gCO₂/kWh")
    print(f" Saved to {processed_path}")

def preprocess_all():
    print(" Preprocessing all zones...\n")
    for zone, filepath in ZONES.items():
        preprocess(zone, filepath)
    print("\n All zones preprocessed.")

preprocess_all()