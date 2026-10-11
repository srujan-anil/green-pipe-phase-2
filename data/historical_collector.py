import requests
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_KEY")
ZONES = ["IN", "DE", "US-CAL-CISO"]

def fetch_historical(zone, days=30):
    print(f"\n Fetching {days} days of historical data for {zone}...")
    
    all_records = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    current = start_date
    
    while current < end_date:
        url = "https://api.electricitymap.org/v3/carbon-intensity/past"
        headers = {"auth-token": API_KEY}
        params = {
            "zone": zone,
            "datetime": current.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                all_records.append({
                    "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                    "zone": zone,
                    "carbon_intensity": data.get("carbonIntensity"),
                    "is_estimated": data.get("isEstimated", False)
                })
                print(f" {current.strftime('%Y-%m-%d %H:00')} — {data.get('carbonIntensity')} gCO₂/kWh")
            else:
                print(f" Skipped {current.strftime('%Y-%m-%d %H:00')} — Status: {response.status_code}")
        
        except Exception as e:
            print(f" Error: {e}")
        
        current += timedelta(hours=1)
    
    # Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(all_records)
    filepath = f"data/raw/{zone}_historical.csv"
    df.to_csv(filepath, index=False)
    print(f"\n Saved {len(all_records)} records to {filepath}")

for zone in ZONES:
    fetch_historical(zone, days=30)

print("\n Historical data collection complete for all zones!")