import requests
import os
import csv
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_KEY")

ZONES = ["IN-KA", "DE", "US-CAL-CISO"]

def fetch_carbon(zone):
    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    headers = {"auth-token": API_KEY}
    params = {"zone": zone}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "zone": zone,
                "carbon_intensity": data.get("carbonIntensity"),
                "renewable_pct": data.get("renewablePercentage", "N/A"),
                "is_estimated": data.get("isEstimated", False)
            }
        else:
            print(f" Error fetching {zone}: {response.status_code}")
            return None

    except Exception as e:
        print(f" Exception for {zone}: {e}")
        return None

def save_to_csv(record, zone):
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{zone}.csv"
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "zone", "carbon_intensity", 
            "renewable_pct", "is_estimated"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f" Saved {zone}: {record['carbon_intensity']} gCO₂/kWh at {record['timestamp']}")

def collect_all_zones():
    print(f"\n Collecting carbon data for all zones...")
    for zone in ZONES:
        record = fetch_carbon(zone)
        if record:
            save_to_csv(record, zone)
        time.sleep(2)  # small delay between zone requests

def run_collector(hours=24):
    print(f" Starting data collector — running for {hours} hours")
    print(f" Data will be saved to data/raw/\n")

    for i in range(hours):
        collect_all_zones()
        if i < hours - 1:
            print(f" Waiting 1 hour for next collection... ({i+1}/{hours} done)\n")
            time.sleep(3600)  # wait 1 hour

    print("\n Data collection complete.")

# Run once immediately to test
collect_all_zones()