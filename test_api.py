import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_KEY")

def test_connection():
    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    headers = {"auth-token": API_KEY}
    params = {"zone": "DE"}  # DE works reliably on sandbox

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            carbon = data.get("carbonIntensity")
            timestamp = data.get("datetime")
            print("✅ API connection successful")
            print(f"Zone      : DE")
            print(f"Timestamp : {timestamp}")
            print(f"Carbon    : {carbon} gCO₂/kWh")

        elif response.status_code == 401:
            print("❌ Invalid API key.")

        elif response.status_code == 400:
            print("❌ Bad request — printing full response above for diagnosis.")

    except Exception as e:
        print(f"❌ Exception: {e}")

test_connection()